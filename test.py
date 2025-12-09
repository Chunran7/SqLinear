from utils.data_loader import get_dataloader

# 配置路径 (根据你实际的文件位置修改)
history_data_path = "data/sd/2019/his.npz"  # 包含所有样本的历史数据
index_path = "data/sd/2019/idx_train.npy"   # 用于训练的样本索引
loc_path = "data/sd/sd_meta.csv"           # 节点元数据
capacity = 4                               # SD 数据集通常设为 4
batch_size = 32

print("=== 开始测试 DataLoader ===")
try:
    # 注意：现在需要传入两个数据路径
    loader, dataset = get_dataloader(history_data_path, index_path, loc_path, capacity, batch_size)

    # 打印一些关键信息
    print(f"\n✅ 数据集加载成功！")
    print(f"总 Patch 数量: {dataset.num_patches}")
    print(f"总节点数量: {dataset.num_nodes}")

    # 尝试拿一个 Batch 的数据
    for x, y in loader:
        print(f"\n✅ 成功读取一个 Batch:")
        print(f"Input Shape: {x.shape} (Batch, Time, Nodes, Feat)")
        print(f"Label Shape: {y.shape}")

        # 验证节点数是否能被 Patch Size 整除 (如果不整除，后面要补0，先看看情况)
        if x.shape[2] % capacity != 0:
            print(f"⚠️ 注意：节点数 {x.shape[2]} 不能被 {capacity} 整除，后面写模型可能需要 padding。")
        else:
            print(f"完美！节点数可以被整除。")
        break

except Exception as e:
    print(f"\n❌ 出错了: {e}")
