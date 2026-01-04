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
# 2. Dataset 定义 (优化版本)
# ==========================================

class TrafficDataset(Dataset):
    def __init__(self, processed_data, indices, input_len, pred_len):
        # processed_data: 已处理的数据字典，包含 'flow', 'tod_idx', 'dow_idx'
        self.processed_data = processed_data
        self.indices = indices
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # 滑动窗口
        t = self.indices[index]
        x_end = t + self.input_len
        y_end = x_end + self.pred_len

        # 直接切片获取预处理好的数据
        # Flow: (T, N, 1) - 流量数据，浮点数
        x_flow = self.processed_data['flow'][t: x_end, :, :]
        y_flow = self.processed_data['flow'][x_end: y_end, :, :]

        # Time of Day Index: (T, N) - 时间索引，整数 (扩展为与节点数量匹配)
        x_tod = self.processed_data['tod_idx'][t: x_end, :]  # (input_len, N)
        y_tod = self.processed_data['tod_idx'][x_end: y_end, :]  # (pred_len, N)

        # Day of Week Index: (T, N) - 星期索引，整数 (扩展为与节点数量匹配)
        x_dow = self.processed_data['dow_idx'][t: x_end, :]  # (input_len, N)
        y_dow = self.processed_data['dow_idx'][x_end: y_end, :]  # (pred_len, N)

        # 扩展维度以匹配批次处理: (T, N) -> (T, N, 1)
        x_tod = x_tod.unsqueeze(-1)  # (T, N, 1)
        y_tod = y_tod.unsqueeze(-1)  # (T, N, 1)
        x_dow = x_dow.unsqueeze(-1)  # (T, N, 1)
        y_dow = y_dow.unsqueeze(-1)  # (T, N, 1)

        # 返回训练所需的数据 (x_flow, x_tod, x_dow, y_flow)
        return x_flow, x_tod, x_dow, y_flow


# ==========================================
# 3. 主加载函数 get_dataloader
# ==========================================

def get_dataloader(args):
    """
    主入口：集成 Square Partition 算法，优化数据加载效率
    """
    # -------------------------------------------------------
    # 1. 空间划分 (Square Partition) - 核心修改部分
    # -------------------------------------------------------
    meta_path = os.path.join(args.dataset_root, args.dataset_type, args.meta_file)
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


    patches = separate_partitions(node_list, args.patch_capacity)
    num_patches = len(patches)
    print(f"划分完成，生成 {num_patches} 个 Patches。")


    partition_idx = []
    for patch in patches:
        for node in patch:
            partition_idx.append(node.original_index)

    # -------------------------------------------------------
    # 2. 时序数据加载 (优化版)
    # -------------------------------------------------------
    data_year = os.path.join(args.dataset_root, args.dataset_type, args.year_folder)
    his_path = os.path.join(data_year, 'his.npz')

    try:
        raw = np.load(his_path, allow_pickle=True)
        raw_data = raw['data']  # (35040, 716, 3)
        scaler_mean = raw['mean']
        scaler_std = raw['std']
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise


    # 提取各维度数据
    flow_data = raw_data[:, :, 0:1]  # (35040, 716, 1) - Traffic Flow
    tod_data = raw_data[:, :, 1]     # (35040, 716) - Time of Day (0~1 normalized)
    dow_data = raw_data[:, :, 2]     # (35040, 716) - Day of Week (0~1 normalized)

    # 预处理时间索引：将归一化的浮点数还原为整数索引
    # 还原逻辑：TOD_Index = round(val * 96) % 96
    tod_indices = np.round(tod_data * 96) % 96  # (35040, 716)
    tod_indices = tod_indices.astype(np.int64)  # 转为 LongTensor

    # 还原逻辑：DOW_Index = round(val * 7) % 7
    dow_indices = np.round(dow_data * 7) % 7   # (35040, 716)
    dow_indices = dow_indices.astype(np.int64)  # 转为 LongTensor

    # 转换为 PyTorch Tensors 并常驻内存
    flow_tensor = torch.FloatTensor(flow_data)  # (35040, 716, 1)
    tod_tensor = torch.LongTensor(tod_indices)  # (35040, 716)
    dow_tensor = torch.LongTensor(dow_indices)  # (35040, 716)

    # [核心修改] 预处理阶段完成重排：根据partition_idx重排原始数据
    partition_idx_tensor = torch.LongTensor(partition_idx)
    flow_tensor = flow_tensor[:, partition_idx_tensor, :]  # 按分区索引重排数据
    tod_tensor = tod_tensor[:, partition_idx_tensor]        # 按分区索引重排时间索引
    dow_tensor = dow_tensor[:, partition_idx_tensor]        # 按分区索引重排星期索引

    # 加载官方索引
    idx_train = np.load(os.path.join(data_year, 'idx_train.npy'))
    idx_val = np.load(os.path.join(data_year, 'idx_val.npy'))
    idx_test = np.load(os.path.join(data_year, 'idx_test.npy'))

    # 准备处理好的数据字典
    processed_data_train = {
        'flow': flow_tensor,
        'tod_idx': tod_tensor,
        'dow_idx': dow_tensor
    }

    # 封装 Dataset - 只需要传入处理好的数据和索引
    train_set = TrafficDataset(processed_data_train, idx_train, args.input_len, args.pred_len)
    val_set = TrafficDataset(processed_data_train, idx_val, args.input_len, args.pred_len)
    test_set = TrafficDataset(processed_data_train, idx_test, args.input_len, args.pred_len)

    # 高效加载：设置 num_workers=0, pin_memory=False 避免Windows多进程死锁
    loader_args = dict(
        batch_size=args.batch_size, 
        num_workers=0,          # 改为 0，避免Windows多进程死锁
        pin_memory=False,       # 改为 False，CPU训练不需要锁页内存
        persistent_workers=False # num_workers=0 时必须为 False
    )
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # 确保 mean 和 std 是标量 (float)
    scaler_mean_scalar = float(scaler_mean) if np.isscalar(scaler_mean) or scaler_mean.size == 1 else scaler_mean[0]
    scaler_std_scalar = float(scaler_std) if np.isscalar(scaler_std) or scaler_std.size == 1 else scaler_std[0]

    scaler_info = {'mean': scaler_mean_scalar, 'std': scaler_std_scalar}

    # 返回处理好的 partition_idx 供模型使用
    return train_loader, val_loader, test_loader, partition_idx, scaler_info