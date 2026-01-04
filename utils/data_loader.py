import os
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import config

from core.partition import separate_partitions


# ==========================================
# 1. 定义辅助类和你的划分算法
# ==========================================

class Node:
    """
    辅助类：用于封装节点信息，方便 separate_partitions 调用
    """

    def __init__(self, idx, lat, lng):
        self.original_index = idx  # 记录原始顺序 (0, 1, 2...)
        self.lat = lat
        self.lng = lng

# ==========================================
# 2. Dataset 定义 (保持不变)
# ==========================================

class TrafficDataset(Dataset):
    def __init__(self, raw_data, indices, input_len, pred_len, partition_idx=None, steps_per_day=96):
        # 注意：因为是 15分钟数据，steps_per_day = 24 * 4 = 96
        self.raw_data = raw_data
        self.indices = indices
        self.input_len = input_len
        self.pred_len = pred_len
        self.partition_idx = partition_idx  # 新增：用于预处理重排
        self.steps_per_day = steps_per_day

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        # 滑动窗口
        t = self.indices[index]
        x_end = t + self.input_len
        y_end = x_end + self.pred_len

        # 1. 物理特征 (Flow, Occ, Spd) -> Float
        # 这里只取前 3 维，确保不混入时间
        x_phys = torch.FloatTensor(self.raw_data[t: x_end, :, :3])
        y_phys = torch.FloatTensor(self.raw_data[x_end: y_end, :, :3])

        # 2. 生成时间特征 (Time Embedding) -> Long (整数索引)
        time_indices = np.arange(t, x_end)

        # [关键修改] 不要除以 steps_per_day，保留整数！
        tod = time_indices % self.steps_per_day  # range [0, 95]
        dow = (time_indices // self.steps_per_day) % 7  # range [0, 6]

        # 转为 Tensor
        tod = torch.LongTensor(tod)
        dow = torch.LongTensor(dow)

        # 3. 扩展维度以匹配物理特征: (T,) -> (T, N, 1)
        # 这样方便模型里的 gather 操作
        N = x_phys.shape[1]
        tod = tod.view(-1, 1, 1).expand(-1, N, 1)
        dow = dow.view(-1, 1, 1).expand(-1, N, 1)

        # [关键修改] 返回 4 个独立变量，不再拼接
        # 注意：数据已在预处理阶段完成重排，无需再次重排
        return x_phys, tod, dow, y_phys


# ==========================================
# 3. 主加载函数 get_dataloader
# ==========================================

def get_dataloader(args):
    """
    主入口：集成 Square Partition 算法
    """
    # -------------------------------------------------------
    # 1. 空间划分 (Square Partition) - 核心修改部分
    # -------------------------------------------------------
    meta_path = os.path.join(args.dataset_root,args.dataset_type,args.meta_file)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")

    # A. 读取 Meta 并转换为 Node 对象列表
    df_meta = pd.read_csv(meta_path)
    #  CSV 列名是大写的 'Lat' 和 'Lng'
    node_list = []
    for idx, row in df_meta.iterrows():
        # idx 就是原始的 0~715 索引
        node_list.append(Node(idx, row['Lat'], row['Lng']))

    print(f"正在对 {len(node_list)} 个节点执行 Square Partition (Capacity={args.patch_capacity})...")

    # B. 调用你的算法
    patches = separate_partitions(node_list, args.patch_capacity)
    num_patches = len(patches)
    print(f"划分完成，生成 {num_patches} 个 Patches。")

    # C. 扁平化索引 (移除 Padding 逻辑)
    # 由于 Eq. 7 的存在，对于 SD 数据集 (N=716, C=4)，每个 patch 的长度必然等于 capacity
    partition_idx = []
    for patch in patches:
        for node in patch:
            partition_idx.append(node.original_index)


    # -------------------------------------------------------
    # 2. 时序数据加载 (LargeST Benchmark) - 保持不变
    # -------------------------------------------------------
    data_year=os.path.join(args.dataset_root,args.dataset_type,args.year_folder)
    his_path = os.path.join(data_year,'his.npz')

    try:
        raw_data = np.load(his_path)['data']
    except:
        f = np.load(his_path)
        raw_data = f[list(f.keys())[0]]

    # 加载官方索引
    idx_train = np.load(os.path.join(data_year, 'idx_train.npy'))
    idx_val = np.load(os.path.join(data_year, 'idx_val.npy'))
    idx_test = np.load(os.path.join(data_year, 'idx_test.npy'))

    # [修改] 读取 his.npz
    data_dir = os.path.join(args.dataset_root, args.dataset_type, args.year_folder)
    his_path = os.path.join(data_dir, 'his.npz')
    
    raw = np.load(his_path, allow_pickle=True)
    # 根据你的 inspect 结果，key 包含 'data', 'mean', 'std'
    raw_data = raw['data'] # (35040, 716, 3)
    
    # [关键] 读取预存的 mean/std 用于后续反归一化
    # 注意：mean/std 的形状可能是 (1, 1, 3) 或 (3,)
    scaler_mean = raw['mean']
    scaler_std = raw['std']
    
    print(f"加载预处理数据: Mean shape {scaler_mean.shape}, Std shape {scaler_std.shape}")

    # [关键] 严禁在此处再次归一化！数据已经是 Z-Score 过的了！

    # [核心修改] 预处理阶段完成重排：根据partition_idx重排原始数据
    partition_idx_tensor = torch.LongTensor(partition_idx)
    raw_data = raw_data[:, partition_idx_tensor, :]  # 按分区索引重排数据
    
    # 封装 Dataset (注意 steps_per_day=96)，不再传递partition_idx，数据已重排
    train_set = TrafficDataset(raw_data, idx_train, args.input_len, args.pred_len, None, steps_per_day=96)
    val_set   = TrafficDataset(raw_data, idx_val,   args.input_len, args.pred_len, None, steps_per_day=96)
    test_set  = TrafficDataset(raw_data, idx_test,  args.input_len, args.pred_len, None, steps_per_day=96)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    scaler_info = {'mean': scaler_mean, 'std': scaler_std}

    # 返回处理好的 partition_idx 供模型使用
    return train_loader, val_loader, test_loader, partition_idx, scaler_info