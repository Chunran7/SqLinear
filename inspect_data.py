import numpy as np
import os


def inspect_largest_data(data_dir):
    """
    检查 LargeST 数据集文件的形状和内容统计
    data_dir: 包含 his.npz 和 idx_*.npy 的文件夹路径 (例如 './data/SD/2019')
    """
    print(f"====== 正在检查目录: {data_dir} ======")

    # ---------------------------------------------
    # 1. 检查 his.npz (原始流量数据)
    # ---------------------------------------------
    his_path = os.path.join(data_dir, 'his.npz')
    if os.path.exists(his_path):
        try:
            # allow_pickle=True 是为了防止某些旧版本 numpy 的安全限制报错
            raw = np.load(his_path, allow_pickle=True)

            print(f"\n[his.npz 分析]")
            print(f"包含的键 (Keys): {raw.files}")

            # 通常键名是 'data'，但为了保险，我们动态获取
            key = 'data' if 'data' in raw.files else raw.files[0]
            data_array = raw[key]

            print(f"数据形状 (Shape): {data_array.shape}")
            print(f"  -> 时间步数 (T): {data_array.shape[0]}")
            print(f"  -> 节点数 (N):   {data_array.shape[1]}")
            print(f"  -> 特征数 (D):   {data_array.shape[2]}")

            # 简单统计
            print(
                f"数据统计: Mean={np.mean(data_array):.4f}, Max={np.max(data_array):.4f}, Min={np.min(data_array):.4f}")

        except Exception as e:
            print(f"读取 his.npz 失败: {e}")
    else:
        print(f"[警告] 找不到 {his_path}")

    # ---------------------------------------------
    # 2. 检查 idx_*.npy (划分索引)
    # ---------------------------------------------
    print(f"\n[数据集划分索引分析]")
    split_files = ['idx_train.npy', 'idx_val.npy', 'idx_test.npy']

    indices = {}

    for file_name in split_files:
        path = os.path.join(data_dir, file_name)
        if os.path.exists(path):
            idx = np.load(path)
            indices[file_name] = idx
            print(f"{file_name}:")
            print(f"  -> 样本数量: {idx.shape[0]}")
            print(f"  -> 索引范围: {idx.min()} ~ {idx.max()}")
        else:
            print(f"[警告] 找不到 {file_name}")

    # ---------------------------------------------
    # 3. 验证是否符合 'Chronological Split' (时间顺序划分)
    # ---------------------------------------------
    # 论文提到必须是 Train -> Val -> Test，不能乱序
    if len(indices) == 3:
        train_max = indices['idx_train.npy'].max()
        val_min = indices['idx_val.npy'].min()
        val_max = indices['idx_val.npy'].max()
        test_min = indices['idx_test.npy'].min()

        print(f"\n[逻辑验证]")
        if train_max < val_min and val_max < test_min:
            print("✅ 验证通过：数据集是严格按时间顺序划分的 (Train -> Val -> Test)。")
        else:
            print("❌ 验证失败：数据集索引存在重叠或乱序！")


# --- 运行配置 ---
if __name__ == "__main__":
    # 请修改为你实际存放数据的路径
    # 例如你的结构是 data/SD/2019，那就填这里
    target_dir = './data/sd/2019'

    if os.path.exists(target_dir):
        inspect_largest_data(target_dir)
    else:
        print(f"错误: 路径 {target_dir} 不存在，请修改脚本中的 target_dir 变量。")