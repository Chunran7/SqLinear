import os
import time
import torch
import torch.optim as optim
import numpy as np
from config import get_args
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae, masked_rmse, masked_mape
from model.sqlinear import SqLinear


def main():
    # 1. 加载配置
    args = get_args()
    device = torch.device(args.device)

    # 创建保存目录
    save_dir = os.path.join('./checkpoints', args.dataset_type)
    os.makedirs(save_dir, exist_ok=True)

    print(f"--- Running Training strictly following SqLinear Paper ---")
    print(f"Device: {device}")
    print(f"Patch Capacity: {args.patch_capacity}")

    # 2. 加载数据 (自动处理 Partition 和 Scaling)
    train_loader, val_loader, test_loader, partition_idx, scaler_info = get_dataloader(args)

    # [修改] 获取标量 mean/std
    # 不需要转 Tensor，因为它们是标量，PyTorch 支持 Tensor 和 scalar 运算
    mean = scaler_info['mean']
    std = scaler_info['std']

    # 3. 初始化模型
    # 必须将 partition_idx 转为 LongTensor 并移至 GPU
    partition_idx_tensor = torch.LongTensor(partition_idx).to(device)

    model = SqLinear(
        args.num_nodes,
        args.patch_capacity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        partition_idx=partition_idx_tensor,  # <--- 核心：空间划分索引
        num_layers=args.num_layers,
        input_len=args.input_len,
        output_len=args.pred_len,
    ).to(device)

    # 辅助函数：只处理 Flow 通道的重排
    def reorder_target_flow(y_full, partition_idx):
        # y_full: (B, T, N, 3)
        # 1. 只取 Flow (Channel 0)
        y_flow = y_full[..., 0:1] # (B, T, N, 1)
        
        # 2. 重排
        B, T, N, D = y_flow.shape
        y_flow = y_flow.permute(0, 1, 3, 2) # (B, T, 1, N)
        
        idx = partition_idx.view(1, 1, 1, -1).expand(B, T, D, -1)
        y_ordered = torch.gather(y_flow, -1, idx) # (B, T, 1, N_padded)
        
        return y_ordered.permute(0, 1, 3, 2) # (B, T, N_padded, 1)

    # 4. 优化器配置 (参考论文常见设置)
    # 通常使用 Adam，学习率 1e-3，权重衰减 1e-4
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 学习率调度器 (MultiStepLR 是时序预测标准配置)
    # 在 20, 40, 70 epoch 衰减学习率，或者根据你的收敛情况调整
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 70], gamma=0.1)

    # 5. 训练主循环
    best_val_mae = float('inf')
    patience_count = 0
    max_patience = 15  # Early Stopping 耐心值

    print(f"Start Training: {args.epochs} epochs")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Params for logging
        train_loss_list = []

        # --- Training Step ---
        model.train()
        for x, y in train_loader:
            x = x.to(device) # (B, 12, N, 5)
            y = y.to(device) # (B, 12, N, 3)
            
            optimizer.zero_grad()
            
            # Forward: 模型输出的是 Flow 预测 (B, 12, N_p, 1)
            preds = model(x)
            
            # Target: 提取 Flow 并重排 (B, 12, N_p, 1)
            y_target = reorder_target_flow(y, partition_idx_tensor)
            
            # 计算 Loss (标准化空间)
            loss = masked_mae(preds, y_target, null_val=0.0)
            
            loss.backward()
            # 梯度裁剪 (Gradient Clipping) - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            train_loss_list.append(loss.item())

        # 更新学习率
        scheduler.step()

        # --- Validation Step (Strictly on Unscaled Data) ---
        # 验证集必须反归一化后计算，才是真实的物理误差
        model.eval()
        val_mae_list = []
        val_rmse_list = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                
                preds = model(x)
                y_target = reorder_target_flow(y, partition_idx_tensor)
                
                # [关键] 反归一化
                # 因为 mean/std 是标量，这里直接运算即可，不需要考虑维度广播
                preds_real = preds * std + mean
                y_real = y_target * std + mean
                
                # 计算真实误差
                v_mae = masked_mae(preds_real, y_real, null_val=0.0)
                v_rmse = masked_rmse(preds_real, y_real, null_val=0.0)

                val_mae_list.append(v_mae.item())
                val_rmse_list.append(v_rmse.item())

        train_loss = np.mean(train_loss_list)
        val_mae = np.mean(val_mae_list)
        val_rmse = np.mean(val_rmse_list)

        end_time = time.time()

        print(f"Epoch {epoch + 1:03d} | Time: {end_time - start_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # --- Early Stopping ---
        # 基于验证集 MAE (真实流量误差) 来决定模型好坏
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_count = 0
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> Best MAE updated. Model saved.")
        else:
            patience_count += 1
            if patience_count >= max_patience:
                print(f"  >>> Early stopping triggered at epoch {epoch + 1}")
                break

    # --- Final Test Step ---
    print("\n--- Starting Final Testing ---")
    # 加载最优模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()

    test_mae = []
    test_rmse = []
    test_mape = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            y_target = reorder_target_flow(y, partition_idx_tensor)

            # 反归一化
            preds_real = preds * std + mean
            y_real = y_target * std + mean

            t_mae = masked_mae(preds_real, y_real, null_val=0.0)
            t_rmse = masked_rmse(preds_real, y_real, null_val=0.0)
            t_mape = masked_mape(preds_real, y_real, null_val=0.0)

            test_mae.append(t_mae.item())
            test_rmse.append(t_rmse.item())
            test_mape.append(t_mape.item())

    print(f"Final Test Results ({args.dataset_type}):")
    print(f"MAE : {np.mean(test_mae):.4f}")
    print(f"RMSE: {np.mean(test_rmse):.4f}")
    print(f"MAPE: {np.mean(test_mape):.4f}")


if __name__ == "__main__":
    main()