import torch
import numpy as np
import os
import sys

# 确保 Python 能找到 utils 和 core 模块
sys.path.append(os.getcwd())

from utils.data_loader import get_dataloader


class MockArgs:
    """
    模拟 config.py 中的参数，方便独立测试
    请确保这里的路径与你实际的文件结构一致
    """

    def __init__(self):
        # --- 路径配置 (请根据你的实际情况修改) ---
        self.dataset_root = './data'  # 假设 SD 文件夹在 data 下
        self.dataset_type = 'sd'
        self.year_folder = '2019'  # 年份文件夹
        self.meta_file = 'sd/sd_meta.csv'  # 元数据文件名

        # --- 模型与数据参数 ---
        self.input_len = 12
        self.pred_len = 12
        self.batch_size = 4  # 测试时用小一点的 batch
        self.patch_capacity = 4  # 设为 4，方便肉眼检查 padding

        # --- 调试用 ---
        self.num_nodes = 716  # SD 数据集节点数


def test():
    print("====== 开始测试 Data Loader ======")
    args = MockArgs()
    #
    # 1. 检查路径是否存在
    full_meta_path = os.path.join(args.dataset_root, args.meta_file)
    data_year=os.path.join(args.dataset_root,args.dataset_type,args.year_folder)
    full_his_path = os.path.join(data_year, 'his.npz')

    if not os.path.exists(full_meta_path):
        print(f"[错误] 找不到 Meta 文件: {full_meta_path}")
        return
    if not os.path.exists(full_his_path):
        print(f"[错误] 找不到 Traffic 文件: {full_his_path}")
        return

    print(f"[检查] 路径检查通过。")

    # 2. 调用核心加载函数
    try:
        train_loader, val_loader, test_loader, partition_idx, scaler = get_dataloader(args)
        print("[成功] get_dataloader 调用成功。")
    except Exception as e:
        print(f"[失败] get_dataloader 运行时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 验证空间划分 (Square Partition) 结果
    print("\n--- 验证 1: 空间划分 (Partition) ---")
    print(f"原始节点数: {args.num_nodes}")
    print(f"Patch 容量 (Capacity): {args.patch_capacity}")
    print(f"生成的 partition_idx 总长度: {len(partition_idx)}")

    # 验证整除性 (Padding 检查)
    if len(partition_idx) % args.patch_capacity == 0:
        num_patches = len(partition_idx) // args.patch_capacity
        print(f"[通过] Padding 正常。共生成 {num_patches} 个 Patches。")
    else:
        print(f"[失败] Partition 长度 ({len(partition_idx)}) 不能被 Capacity 整除！Padding 逻辑可能有误。")

    # 4. 验证 DataLoader 输出形状
    print("\n--- 验证 2: Tensor 形状 ---")
    try:
        # 从训练集取一个 Batch
        x, y = next(iter(train_loader))

        print(f"Batch Size 设置为: {args.batch_size}")
        print(f"输入 X 的形状: {x.shape}")  # 预期: (Batch, 12, N, D)
        print(f"标签 Y 的形状: {y.shape}")  # 预期: (Batch, 12, N, D)

        # 检查是否包含了 NaN
        if torch.isnan(x).any() or torch.isnan(y).any():
            print("[警告] 数据中包含 NaN (空值)！")
        else:
            print("[通过] 数据无 NaN。")

    except Exception as e:
        print(f"[失败] 无法从 DataLoader 读取数据: {e}")
        return

    # 5. 验证标准化信息
    print("\n--- 验证 3: 标准化参数 ---")
    print(f"Mean: {scaler['mean']}")
    print(f"Std : {scaler['std']}")

    print("\n====== 测试完成 ======")


if __name__ == "__main__":
    test()