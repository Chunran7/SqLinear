import torch
from core.layers import SpatioTemporalEmbedding, HLIBlock
import torch.nn as nn

class SqLinear(nn.Module):
    def __init__(self, original_num_nodes, patch_size,
                 input_dim=3,          # 物理特征维度 (Flow, Occ, Spd)
                 token_dim=64,         # 新增参数
                 day_dim=32,           # 新增参数
                 week_dim=32,          # 新增参数
                 spatial_dim=32,       # 新增参数
                 output_dim=1,
                 num_layers=4,
                 input_len=12, output_len=12,
                 partition_idx=None):
        """
        Args:
            original_num_nodes: 原始节点数 (e.g., 716)
            partition_idx: 包含 Padding 的重排索引 (长度 >= 716)
        """
        super().__init__()
        self.patch_size = patch_size
        
        # 论文强调无填充，所以这里索引长度应直接等于原始节点数
        if partition_idx is not None:
            # 注册为 buffer，自动随模型移动到 GPU
            self.register_buffer('partition_idx', torch.LongTensor(partition_idx))
            self.num_nodes = original_num_nodes
        else:
            self.partition_idx = None
            self.num_nodes = original_num_nodes

        # 直接计算 Patch 数量
        self.num_patches = self.num_nodes // patch_size
        
        # 打印信息保持专业
        print(f"Model Init: Nodes={self.num_nodes}, Patches={self.num_patches} (Padding-free)")

        # 1. 嵌入层
        self.embedding = SpatioTemporalEmbedding(
            input_dim=input_dim,
            token_dim=token_dim,
            day_dim=day_dim,
            week_dim=week_dim,
            spatial_dim=spatial_dim
        )
        
        # [关键] 计算 HLI Block 需要的总维度
        # 64 + 32 + 32 + 32 = 160
        self.total_hidden_dim = self.embedding.output_dim

        print(f"Model Config: Token={token_dim}, Time={day_dim}+{week_dim}, Spatial={spatial_dim}")
        print(f"Total Hidden Dim for HLI: {self.total_hidden_dim}")

        # 2. 核心层 (堆叠 L 层 HLI)
        # 使用较小的rank值以符合低秩投影要求
        self.layers = nn.ModuleList([
            HLIBlock(self.total_hidden_dim, self.num_patches, patch_size, rank=min(3, patch_size//2))
            for _ in range(num_layers)
        ])

        # 3. 输出层 (论文 4.4 节)
        # 3.1 特征降维: Hidden(64) -> Output(1)
        # 这里用 1x1 卷积或者 Linear 都可以
        self.output_proj = nn.Linear(self.total_hidden_dim, output_dim)

        # 3.2 时间预测: Input_Len(12) -> Output_Len(12)
        # 直接用 Linear 映射时间轴
        self.time_proj = nn.Linear(input_len, output_len)

    def forward(self, x_phys, t_day, t_week):  # <--- 接收 3 个输入
        """
        Args:
            x_phys: (B, T, N, 3) Float  # 已在数据加载时预处理重排
            t_day:  (B, T, N, 1) Long   # 已在数据加载时预处理重排
            t_week: (B, T, N, 1) Long   # 已在数据加载时预处理重排
        """
        # 直接进入 Embedding，无需任何重排操作
        x = self.embedding(x_phys, t_day, t_week)

        # Step 2: Patching (变形)
        # 对应论文 Eq. 9 的 Patching 操作
        # 数据加载阶段已完成预处理重排，
        # 所以这里只需要简单的 Reshape (View)
        # (B, T, N, H) -> (B, T, P, C, H)
        B, T, N, H = x.shape  # N 现在是原始节点数，无填充
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