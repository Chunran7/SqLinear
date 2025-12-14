import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='SqLinear Project Configuration')

    # --- 数据集路径配置 ---
    # 假设你的目录结构是: data/sd/2019/his.npz 和 data/SD/sd_meta.csv
    parser.add_argument('--dataset_root', type=str, default='./data', help='数据集根目录')
    parser.add_argument('--dataset_type', type=str, default='sd', help='数据集类型')
    parser.add_argument('--year_folder', type=str, default='2019', help='年份文件夹')
    parser.add_argument('--meta_file', type=str, default='sd_meta.csv', help='元数据文件名')

    # --- 数据处理参数 (LargeST Benchmark Setting) ---
    parser.add_argument('--input_len', type=int, default=12, help='历史时间步长度')
    parser.add_argument('--pred_len', type=int, default=12, help='预测时间步长度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_nodes', type=int, default=716, help='节点(传感器)数量，SD=716')

    # --- SqLinear 模型核心参数 ---
    # [cite: 190, 207] Patch Capacity: 每个正方形小块包含的最大节点数
    parser.add_argument('--patch_capacity', type=int, default=4, help='Leaf node capacity (C)')
    # [重要修改] 输入维度改为 5 
    # 3 (Flow, Occ, Spd) + 1 (Time of Day) + 1 (Day of Week)
    parser.add_argument('--input_dim', type=int, default=5, help='输入特征维度: Flow,Occ,Spd,ToD,DoW')
    parser.add_argument('--hidden_dim', type=int, default=64, help='模型隐藏层维度')
    # 确保 output_dim 仍然是 1 (我们只预测流量)
    parser.add_argument('--output_dim', type=int, default=1, help='输出特征维度')
    parser.add_argument('--num_layers', type=int, default=3, help='层数')

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    return args