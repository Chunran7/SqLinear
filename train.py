import os
import time
import torch
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime
from config import get_args
from utils.data_loader import get_dataloader
from utils.metrics import masked_mae, masked_rmse, masked_mape
from model.sqlinear import SqLinear
from tqdm import tqdm

def main():
    # 1. 加载配置
    args = get_args()

    device = torch.device('cpu')
    print("使用 CPU 进行训练")


    # 创建保存目录和日志文件
    save_dir = os.path.join('./checkpoints', args.dataset_type)
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    log_filename = os.path.join(save_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 配置 Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename), # 写入文件
            logging.StreamHandler()            # 输出到控制台
        ]
    )
    
    def log_string(string):
        logging.info(string)

    log_string(f"--- Running Training strictly following SqLinear Paper ---")
    log_string(f"Log file saved to: {log_filename}")
    log_string(f"Device: {device}")
    log_string(f"Hyperparameters: {args}")
    log_string(f"Patch Capacity: {args.patch_capacity}")

    # 2. 加载数据 (自动处理 Partition 和 Scaling)
    train_loader, val_loader, test_loader, partition_idx, scaler_info = get_dataloader(args)


    mean = scaler_info['mean']
    std = scaler_info['std']

    # 3. 初始化模型
    # 必须将 partition_idx 转为 LongTensor 并移至 CPU
    partition_idx_tensor = torch.LongTensor(partition_idx).to(device)

    model = SqLinear(
        original_num_nodes=args.num_nodes,
        patch_size=args.patch_capacity,
        input_dim=args.input_dim,   # 1
        
        # 传入新参数
        token_dim=args.token_dim,     # 64
        day_dim=args.day_dim,         # 32
        week_dim=args.week_dim,       # 32
        spatial_dim=args.spatial_dim, # 32

        
        num_layers=args.num_layers,
        input_len=args.input_len,
        output_len=args.pred_len,
        partition_idx=partition_idx_tensor
    ).to(device)

    # 4. 优化器配置 (遵循论文)
    # 使用 AdamW 优化器，提供更好的权重衰减处理
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

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

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

        for x_flow, x_tod, x_dow, y_flow in train_loader:
            x_flow = x_flow.to(device)  # Flow: Float
            x_tod = x_tod.to(device)    # Time of Day Index: Long
            x_dow = x_dow.to(device)    # Day of Week Index: Long
            y_flow = y_flow.to(device)  # Flow: Float

            optimizer.zero_grad()

            # [修改] 传入 3 个参数
            preds = model(x_flow, x_tod, x_dow)
            
            # Target: 提取 Flow 通道 (B, 12, N, 1) - 数据已在预处理阶段重排
            y_target = y_flow  # 因为 y_flow 已经是 flow 数据了
            
            # 计算 Loss (标准化空间)
            loss = masked_mae(preds, y_target, null_val=0.0)
            
            loss.backward()
            # 梯度裁剪 (Gradient Clipping) - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            train_loss_list.append(loss.item())

            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 更新学习率
        scheduler.step()

        # --- Validation Step (Strictly on Unscaled Data) ---
        # 验证集必须反归一化后计算，才是真实的物理误差
        model.eval()
        val_mae_list = []
        val_rmse_list = []

        with torch.no_grad():
            for x_flow, x_tod, x_dow, y_flow in val_loader:
                x_flow = x_flow.to(device)  # Flow: Float
                x_tod = x_tod.to(device)    # Time of Day Index: Long
                x_dow = x_dow.to(device)    # Day of Week Index: Long
                y_flow = y_flow.to(device)  # Flow: Float
                
                preds = model(x_flow, x_tod, x_dow)
                # Target: 提取 Flow 通道 - 数据已在预处理阶段重排
                y_target = y_flow  # 因为 y_flow 已经是 flow 数据了
                
                # [关键] 反归一化
                preds_real = preds * std + mean
                y_real = y_flow * std + mean  # 使用 y_flow 而不是 y_target
                
                # 计算真实误差
                v_mae = masked_mae(preds_real, y_real, null_val=0.0)
                v_rmse = masked_rmse(preds_real, y_real, null_val=0.0)

                val_mae_list.append(v_mae.item())
                val_rmse_list.append(v_rmse.item())

        train_loss = np.mean(train_loss_list)
        val_mae = np.mean(val_mae_list)
        val_rmse = np.mean(val_rmse_list)

        end_time = time.time()

        # 使用格式化字符串记录每一轮的详细信息
        log_string(
            f"Epoch {epoch + 1:03d} | Time: {end_time - start_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}"
        )

        # --- Early Stopping ---
        # 基于验证集 MAE (真实流量误差) 来决定模型好坏
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_count = 0
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            log_string(f"  >>> Best Val MAE updated: {val_mae:.4f}. Model saved.")
        else:
            patience_count += 1
            if patience_count >= max_patience:
                log_string(f"  >>> Early stopping triggered at epoch {epoch + 1}")
                break

    # --- Final Test Step ---
    log_string("\n" + "="*30)
    log_string("Starting Final Testing on Test Set")
    # 加载最优模型
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()

    test_mae = []
    test_rmse = []
    test_mape = []

    with torch.no_grad():
        for x_flow, x_tod, x_dow, y_flow in test_loader:
            x_flow = x_flow.to(device)  # Flow: Float
            x_tod = x_tod.to(device)    # Time of Day Index: Long
            x_dow = x_dow.to(device)    # Day of Week Index: Long
            y_flow = y_flow.to(device)  # Flow: Float

            preds = model(x_flow, x_tod, x_dow)
            # Target: 提取 Flow 通道 - 数据已在预处理阶段重排
            y_target = y_flow  # 因为 y_flow 已经是 flow 数据了

            # 反归一化
            preds_real = preds * std + mean
            y_real = y_flow * std + mean  # 使用 y_flow 而不是 y_target

            t_mae = masked_mae(preds_real, y_real, null_val=0.0)
            t_rmse = masked_rmse(preds_real, y_real, null_val=0.0)
            t_mape = masked_mape(preds_real, y_real, null_val=0.0)

            test_mae.append(t_mae.item())
            test_rmse.append(t_rmse.item())
            test_mape.append(t_mape.item())

    avg_mae = np.mean(test_mae)
    avg_rmse = np.mean(test_rmse)
    avg_mape = np.mean(test_mape)

    log_string(f"Final Test Results ({args.dataset_type}):")
    log_string(f"MAE  : {avg_mae:.4f}")
    log_string(f"RMSE : {avg_rmse:.4f}")
    log_string(f"MAPE : {avg_mape:.4f}")
    log_string("="*30)


if __name__ == "__main__":
    main()