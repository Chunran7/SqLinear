from core.layers import SpatioTemporalEmbedding, HLIBlock
import torch.nn as nn

class SqLinear(nn.Module):
    def __init__(self, num_nodes, patch_size,
                 input_dim=1, output_dim=1,
                 hidden_dim=64, num_layers=4,
                 input_len=12, output_len=12,
                 partition_idx=None):
        """
        Args:
            num_nodes: 716
            num_patches: 179 (716/4)
            patch_size: 4
            input_dim: 1 (原始流量)
            output_dim: 1 (预测流量)
            hidden_dim: 模型内部特征维度 (比如 64)
            num_layers: HLI 层数 (比如 4)
            partition_idx: 空间划分索引
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_nodes//patch_size
        
        # 保存分区索引（如果提供了的话）
        if partition_idx is not None:
            self.partition_idx = partition_idx

        # 1. 嵌入层
        # 注意：Embedding层最后会拼接 4 个特征。
        # 为了让拼接后的总维度等于 hidden_dim，我们这里传入 hidden_dim // 4
        # 比如 hidden_dim=64, 那每个子特征就是 16, 16*4=64
        emb_dim = hidden_dim // 4
        self.embedding = SpatioTemporalEmbedding(
            input_dim=input_dim,
            hidden_dim=emb_dim,
            #num_nodes=num_nodes
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

    def forward(self, x, t_day, t_week):
        # 输入 x: (Batch, Time, Nodes, 1)

        # ====================================
        # Step 1: Embedding (特征增强)
        # ====================================
        # Out: (B, T, N, Hidden)
        x = self.embedding(x, t_day, t_week)

        # ====================================
        # Step 2: Patching (变形)
        # ====================================
        # 对应论文 Eq. 9 的 Patching 操作
        # 我们的 DataLoader 已经把节点按顺序排好了，
        # 所以这里只需要简单的 Reshape (View)
        # (B, T, N, H) -> (B, T, P, C, H)
        B, T, N, H = x.shape
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
        x = x.view(B, T, N, H)

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