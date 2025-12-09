import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import sys

# ---------------------------------------------------------
# 1. 引入你的核心算法 separate_partitions
# ---------------------------------------------------------
# 假设你的目录结构是:
# project/
#   core/partition.py (里面有 separate_partitions 函数)
#   utils/data_loader.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.partition import separate_partitions


class TrafficDataset(Dataset):
    def __init__(self, data_path, loc_path, capacity):
        """
        Args:
            data_path: .npz 文件路径 (例如 'data/sd/2019/train.npz')
            loc_path:  metadata.csv 文件路径 (例如 'data/sd/sd_meta.csv')
            capacity:  每个 Patch 的容量 C (论文中的 Leaf Capacity)
        """
        super().__init__()

        # ==========================================
        # 第一步：读取坐标 & 运行划分算法
        # ==========================================
        # 1. 读取元数据 CSV
        df = pd.read_csv(loc_path)

        # 2. 定义一个简单的节点类，用来喂给你的算法
        class SimpleNode:
            def __init__(self, id, lat, lng):
                self.id = id  # 原始索引 ID
                self.lat = lat  # 纬度
                self.lng = lng  # 经度

            # 方便调试打印
            def __repr__(self):
                return f"Node({self.id})"

        # 3. 把 DataFrame 转换成 Node 对象列表
        nodes = []
        for _, row in df.iterrows():
            nid = int(row['ID'])
            lat, lng = row['Lat'], row['Lng']
            nodes.append(SimpleNode(nid, lat, lng))

        print(f"Dataset: 加载了 {len(nodes)} 个节点，正在执行 separate_partitions (C={capacity})...")

        # 4. 调用你的算法！
        # patches 是一个列表的列表: [[Node1, Node2], [Node3, Node4]...]
        # 每个子列表代表一个分组，包含该分组中的所有节点
        self.patches = separate_partitions(nodes, capacity)
        self.num_patches = len(self.patches)
        self.num_nodes = len(nodes)

        print(f"Dataset: 划分完成，共生成 {self.num_patches} 个 Patch。")

        # ==========================================
        # 第二步：生成重排索引 (Reordering Index)
        # ==========================================
        # 我们要把原始数据里的节点顺序打乱，让同一个 Patch 的节点挨在一起
        self.ordered_indices = []
        for patch in self.patches:
            for node in patch:
                self.ordered_indices.append(node.id)

        # 转成 numpy 数组，防止报错
        self.ordered_indices = np.array(self.ordered_indices)

        # ==========================================
        # 第三步：读取流量数据 & 执行 Patching
        # ==========================================
        # 加载 .npz 文件
        # LargeST 生成的文件里通常有 'x' 和 'y' 两个数组
        raw_data = np.load(data_path)
        # x shape: (Samples, 12, Nodes, Features)
        self.x = raw_data['x']
        self.y = raw_data['y']

        # ------------------------------------------
        # 关键操作：Patching (数据重排)
        # 对应论文公式 (9): X_bar = Patching(Index, X)
        # ------------------------------------------
        # axis=2 是节点维度 (Samples, Time, Nodes, Feat)
        # 使用 np.take 提取并重组数据
        self.x = np.take(self.x, self.ordered_indices, axis=2)
        self.y = np.take(self.y, self.ordered_indices, axis=2)

        # 转为 PyTorch Tensor，节省显存可以用 FloatTensor
        self.x = torch.FloatTensor(self.x)
        self.y = torch.FloatTensor(self.y)

        print(f"Dataset: 数据加载完毕，最终形状 X: {self.x.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 返回一条样本：(历史数据, 未来标签)
        return self.x[idx], self.y[idx]


def get_dataloader(data_path, loc_path, capacity, batch_size, shuffle=True):
    """
    外部调用的接口函数
    """
    dataset = TrafficDataset(data_path, loc_path, capacity)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # 把 dataset 也返回出去，因为后面模型初始化需要用到 dataset.num_patches 等属性
    return loader, dataset