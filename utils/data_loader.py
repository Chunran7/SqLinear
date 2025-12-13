import os
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, raw_data, indices, input_len, pred_len):
        self.raw_data = raw_data
        self.indices = indices
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        t = self.indices[index]
        x_end = t + self.input_len
        y_end = x_end + self.pred_len

        x = self.raw_data[t: x_end]
        y = self.raw_data[x_end: y_end]

        return torch.FloatTensor(x), torch.FloatTensor(y)


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
    meta_path = os.path.join(args.dataset_root,args.meta_file)
    print(f"正在读取元数据文件: {args.meta_file}")
    print(f"正在读取元数据文件: {meta_path}")
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

    # C. 扁平化并处理 Padding (关键步骤！)
    # 算法返回的是 [[Node, Node], [Node], ...]，我们需要展平为一维索引列表
    partition_idx = []
    for patch in patches:
        for node in patch:
            partition_idx.append(node.original_index)

        # --- Padding 逻辑 ---
        # 如果某个 Patch 的节点数少于 capacity，需要补齐
        # 否则模型 reshape 成 (Batch, P, C, D) 时会报错
        # 策略：用该 Patch 的最后一个节点重复填充 (或者填 0，视模型 Mask 机制而定)
        # 这里采用"重复填充"，影响较小
        num_to_pad = args.patch_capacity - len(patch)
        if num_to_pad > 0:
            last_node_idx = patch[-1].original_index
            partition_idx.extend([last_node_idx] * num_to_pad)

    # 此时 partition_idx 的长度一定是 num_patches * patch_capacity
    # 它包含了所有节点的重排索引，以及为了对齐而补的索引

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

    # -------------------------------------------------------
    # 3. 数据标准化
    # -------------------------------------------------------
    sample_indices = idx_train[::10]
    sample_data = []
    for t in sample_indices:
        if t + args.input_len <= raw_data.shape[0]:
            sample_data.append(raw_data[t: t + args.input_len])

    if len(sample_data) > 0:
        sample_stack = np.concatenate(sample_data, axis=0)
        mean = sample_stack.mean(axis=(0, 1), keepdims=True)
        std = sample_stack.std(axis=(0, 1), keepdims=True)
    else:
        mean = raw_data.mean()
        std = raw_data.std()

    raw_data_norm = (raw_data - mean) / std

    # -------------------------------------------------------
    # 4. 封装 DataLoader
    # -------------------------------------------------------
    train_set = TrafficDataset(raw_data_norm, idx_train, args.input_len, args.pred_len)
    val_set = TrafficDataset(raw_data_norm, idx_val, args.input_len, args.pred_len)
    test_set = TrafficDataset(raw_data_norm, idx_test, args.input_len, args.pred_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    scaler_info = {'mean': mean, 'std': std}

    # 返回处理好的 partition_idx 供模型使用
    return train_loader, val_loader, test_loader, partition_idx, scaler_info