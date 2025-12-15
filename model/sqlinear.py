import torch
from core.layers import SpatioTemporalEmbedding, HLIBlock
import torch.nn as nn

class SqLinear(nn.Module):
    def __init__(self, original_num_nodes, patch_size,
                 input_dim=1, output_dim=1,
                 hidden_dim=64, num_layers=4,
                 input_len=12, output_len=12,
                 partition_idx=None):
        """
        Args:
            original_num_nodes: 原始节点数 (e.g., 716)
            partition_idx: 包含 Padding 的重排索引 (长度 >= 716)
        """
        super().__init__()
        self.patch_size = patch_size
        
        # [修改 1] 处理 Partition Index 和 Padding 后的节点数
        if partition_idx is not None:
            # 注册为 buffer，自动随模型移动到 GPU
            self.register_buffer('partition_idx', torch.LongTensor(partition_idx))
            # 模型的有效节点数 = 索引的长度 (包含了 Padding)
            self.effective_num_nodes = len(partition_idx)
        else:
            self.partition_idx = None
            self.effective_num_nodes = original_num_nodes

        # 计算 Patch 数量 (基于填充后的节点数)
        self.num_patches = self.effective_num_nodes // patch_size
        
        print(f"Model Init: Original Nodes={original_num_nodes}, "
              f"Effective Nodes (Padded)={self.effective_num_nodes}, "
              f"Patches={self.num_patches}")

        # 1. 嵌入层
        # 注意：Embedding层最后会拼接 4 个特征。
        # 为了让拼接后的总维度等于 hidden_dim，我们这里传入 hidden_dim // 4
        # 比如 hidden_dim=64, 那每个子特征就是 16, 16*4=64
        emb_dim = hidden_dim // 4
        physical_feature_dim = 3


        self.embedding = SpatioTemporalEmbedding(
            input_dim=physical_feature_dim,
            hidden_dim=emb_dim,

            # num_nodes=self.effective_num_nodes # 如果你的 Embedding 需要节点 ID，用这个
        )

        # 2. 核心层 (堆叠 L 层 HLI)
        self.layers = nn.ModuleList([
            HLIBlock(hidden_dim, self.num_patches, patch_size)
            for _ in range(num_layers)
        ])

        # 3. 输出层 (论文 4.4 节)
        # 3.1 特征降维: Hidden(64) -> Output(1)
        # 这里用 1x1 卷积或者 Linear 都可以
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # 3.2 时间预测: Input_Len(12) -> Output_Len(12)
        # 直接用 Linear 映射时间轴
        self.time_proj = nn.Linear(input_len, output_len)

    def forward(self, x_in):
        """
        Args:
            x_in: (B, T, N_orig, 5) -> [Flow, Occ, Spd, ToD, DoW]
        Returns:
            out: (B, T, N_padded, 1) -> 预测结果是重排且包含 Padding 的
        """
        # [修改 3] 切片提取需要的特征
        # Channel 0: Flow (输入特征)
        # Channel 3: Time of Day
        # Channel 4: Day of Week
        val = x_in[..., 0:3]    # (B, T, N_orig, 1)
        t_day = x_in[..., 3:4]  # (B, T, N_orig, 1)
        t_week = x_in[..., 4:5] # (B, T, N_orig, 1)

        # [修改 4] 空间重排 (Reordering & Padding)
        # 将数据从 N_orig 映射到 N_padded
        if self.partition_idx is not None:
            B, T, _, D = val.shape
            # partition_idx: (N_padded,) -> 扩展为 (1, 1, N_padded, 1)
            idx = self.partition_idx.view(1, 1, -1, 1)
            
            # 扩展 idx 以匹配 Batch 和 Time
            # 注意：gather 的 dim=2 是节点维度
            idx_val = idx.expand(B, T, -1, D)
            idx_t = idx.expand(B, T, -1, 1)

            # 执行 Gather: 输出形状变为 (B, T, N_padded, 1)
            val = torch.gather(val, 2, idx_val)
            t_day = torch.gather(t_day, 2, idx_t)
            t_week = torch.gather(t_week, 2, idx_t)

        # ====================================
        # Step 1: Embedding (特征增强)
        # ====================================
        # Out: (B, T, N, Hidden)
        x = self.embedding(val, t_day, t_week)

        # ====================================
        # Step 2: Patching (变形)
        # ====================================
        # 对应论文 Eq. 9 的 Patching 操作
        # 我们的 DataLoader 已经把节点按顺序排好了，
        # 所以这里只需要简单的 Reshape (View)
        # (B, T, N, H) -> (B, T, P, C, H)
        B, T, N_pad, H = x.shape
        x = x.view(B, T, self.num_patches, self.patch_size, H)

        # ====================================
        # Step 3: HLI Processing (核心计算)
        # ====================================
        for layer in self.layers:
            x = layer(x)

        # ====================================
        # Step 4: Output (输出层)
        # ====================================
        # 4.1 Unpatch (还原形状) - 对应 Eq. 15
        # (B, T, P, C, H) -> (B, T, N, H)
        x = x.view(B, T, N_pad, H)

        # 4.2 预测特征 (Regression) - 对应 Eq. 16
        # 先把特征维变成 1: (B, T, N, H) -> (B, T, N, 1)
        x = self.output_proj(x)

        # 4.3 预测时间
        # (B, T_in, N, 1) -> (B, N, 1, T_in) -> Linear -> (B, N, 1, T_out)
        x = x.permute(0, 2, 3, 1)
        x = self.time_proj(x)

        # 最终调整回 (B, T_out, N, 1)
        x = x.permute(0, 3, 1, 2)

        return x