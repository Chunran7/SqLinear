import torch
import numpy as np


def masked_mae(preds, labels, null_val=0.0):
    """
    计算掩码平均绝对误差 (Masked MAE)
    只计算 labels != null_val (通常是0) 的部分
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean((mask))  # 归一化 mask，保证 loss 数值量级正确
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=0.0):
    """计算掩码均方根误差 (Masked RMSE)"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds, labels, null_val=0.0):
    """计算掩码平均绝对百分比误差 (Masked MAPE)"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)

    # MAPE 分母不能为0，通常我们会把分母的0替换成一个小值或者直接mask掉
    loss = torch.abs(preds - labels) / (torch.abs(labels) + 1e-5)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = loss * mask
    return torch.mean(loss)