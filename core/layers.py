import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 time_of_day_size=288, day_of_week_size=7):
        """
        对应论文 4.1 节
        """
        super().__init__()

        # 1. 历史流量嵌入 (Token Embedding) - 对应 Eq. (2)
        # 输入: 流量数值 -> 输出: 隐藏向量
        self.token_emb = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)

        # 2. 时间嵌入 (Temporal Embedding) - 对应 Eq. (3)
        # 输入: 时间索引 -> 输出: 时间向量
        # 虽然论文提到了 Linear，但对离散索引的标准实现是 Embedding
        self.day_emb = nn.Embedding(time_of_day_size, hidden_dim)
        self.week_emb = nn.Embedding(day_of_week_size, hidden_dim)

        # 3. 空间嵌入 (Spatial Embedding) - 对应 Eq. (4)
        # 【关键点】这里是 "Adaptive Embedding"，输入是流量 X_H
        # 所以必须用 Linear，而不是 Embedding(num_nodes)
        self.spatial_linear = nn.Linear(input_dim, hidden_dim)

        # 记录总维度 (用于后续拼接 Eq. 5)
        # Total = d_h + d_d + d_w + d_s
        self.output_dim = hidden_dim * 4

    def forward(self, x, t_day, t_week):
        """
        x: (Batch, Time, Nodes, In_Dim) -> 原始流量
        t_day: (Batch, Time, 1) -> 每天的时间索引 (0-287)
        t_week: (Batch, Time, 1) -> 每周的时间索引 (0-6)
        注意：不需要传入 node_idx，因为是自适应空间嵌入
        """
        batch, time, nodes, _ = x.shape

        # -------------------------------
        # 1. 流量嵌入 (Data)
        # -------------------------------
        # Conv2d 需要 (B, C, H, W) 格式 -> (B, In_Dim, T, N)
        # 主要是调整各个维度的顺序，适配 Conv2d 的输入要求
        x_perm = x.permute(0, 3, 1, 2)
        h_data = self.token_emb(x_perm).permute(0, 2, 3, 1)  # -> (B, T, N, Hidden)

        # -------------------------------
        # 2. 时间嵌入 (Temporal)
        # -------------------------------
        # t_day: (B, T, 1) -> squeeze -> (B, T) -> emb -> (B, T, Hidden)
        h_day = self.day_emb(t_day.squeeze(-1))
        # 广播到每个节点: (B, T, 1, H) -> (B, T, N, H)
        h_day = h_day.unsqueeze(2).expand(-1, -1, nodes, -1)

        h_week = self.week_emb(t_week.squeeze(-1))
        h_week = h_week.unsqueeze(2).expand(-1, -1, nodes, -1)

        # -------------------------------
        # 3. 空间嵌入 (Spatial - Adaptive)
        # -------------------------------
        # 对应公式 (4): sigma(W * X + b)
        # x: (B, T, N, 1) -> Linear -> (B, T, N, Hidden)
        h_spatial = self.spatial_linear(x)
        h_spatial = F.relu(h_spatial)  # sigma 激活函数

        # -------------------------------
        # 4. 融合 (Concatenate)
        # -------------------------------
        # 对应公式 (5)
        hidden = torch.cat([h_data, h_day, h_week, h_spatial], dim=-1)

        return hidden

    # ==========================================
    # 积木 2: 分层线性交互块 (HLIBlock)
    # ==========================================
    class HLIBlock(nn.Module):
        """
        对应论文 4.3 节: Hierarchical Linear Interaction (HLI)
        包含: Inter-Patch (宏观/店长会) + Intra-Patch (微观/店内会)
        """

        def __init__(self, hidden_dim, num_patches, patch_size, rank=32):
            super().__init__()

            # ------------------------------------------------
            # 1. Inter-Patch Interaction (宏观: 处理 P 维度)
            # ------------------------------------------------
            self.inter_norm = nn.LayerNorm(hidden_dim)
            # 层归一化：把这排数据重新调整一下，让它们的平均值接近 0，方差接近 1。

            self.inter_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # 顺序容器：打包这三个操作，有数据来了就按顺序执行这三个操作

            # 低秩投影: P -> rank -> P (对应 Eq. 11)
            self.low_rank_1 = nn.Linear(num_patches, rank)
            self.low_rank_2 = nn.Linear(rank, num_patches)

            # FFN
            self.inter_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            # 这里是先放大再缩小

            # ------------------------------------------------
            # 2. Intra-Patch Interaction (微观: 处理 C 维度)
            # ------------------------------------------------
            self.intra_norm = nn.LayerNorm(hidden_dim)
            self.intra_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # 补丁内全连接: C -> C (对应 Eq. 13)
            self.intra_mixer = nn.Linear(patch_size, patch_size)

            # FFN
            self.intra_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )

        def forward(self, x):
            # 输入 x 形状: (Batch, Time, P, C, Hidden)

            # ===========================
            # Part 1: Inter-Patch (P)
            # ===========================
            residual = x
            h = self.inter_norm(x)
            h = self.inter_mlp(h)

            # 【关键】把 P (dim=2) 换到最后，因为 Linear 只处理最后一维
            # (B, T, P, C, H) -> (B, T, C, H, P)
            h = h.permute(0, 1, 3, 4, 2)
            h = self.low_rank_1(h)  # P -> rank
            h = self.low_rank_2(h)  # rank -> P
            # 换回来 -> (B, T, P, C, H)
            h = h.permute(0, 1, 4, 2, 3)

            x = residual + self.inter_ffn(h)  # 残差连接

            # ===========================
            # Part 2: Intra-Patch (C)
            # ===========================
            residual = x
            h = self.intra_norm(x)
            h = self.intra_mlp(h)

            # 【关键】把 C (dim=3) 换到最后
            # (B, T, P, C, H) -> (B, T, P, H, C)
            h = h.permute(0, 1, 2, 4, 3)
            h = self.intra_mixer(h)  # C -> C
            # 换回来 -> (B, T, P, C, H)
            h = h.permute(0, 1, 2, 4, 3)

            x = residual + self.intra_ffn(h)  # 残差连接

            return x