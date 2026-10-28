import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sam2.sam2_model.Module.HeterogeneousFusionAggregationModule import HFAM

class SAMFusion(nn.Module):
    def __init__(self, c_1, c_2, dropout=0.1, hga_layers=3):
        """
        Args:
            c_sam (int): SAM 当前 stage 的通道数 (C_sam)。
            c_vssm (int): VMamba 当前 stage 的通道数 (C_vssm)。
            dropout (float): Dropout 概率。
            hga_layers (int): HGA 模块的迭代层数。
        """
        super().__init__()
        self.c_1 = c_1
        self.c_2 = c_2
        self.dropout = dropout

        # 1. 通道对齐层 (C_2 -> C_1)
        self.align_conv = nn.LazyConv2d(c_1, kernel_size=1)

        # 2. HGA 融合模块
        # 这里的 HGA 将处理 N=3 个节点：[SAM, VM_RGB, VM_Edge]
        self.hga = HFAM(
            in_channels=c_1, 
            out_channels=c_1, 
            num_layers=hga_layers,
        )

        # 3. 最终输出的平滑层 (Residual Projection)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_1, c_1, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # 初始化：将输出层初始化为 0，确保在训练初期不破坏 SAM 的预训练权重
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.out_conv[0].weight, 0)
        nn.init.constant_(self.out_conv[0].bias, 0)

    def forward(self, x1, x2, stage_idx=None):
        """
        Args:
            x1 (torch.Tensor): 特征图, shape: [B, C_1, H, W]
            x2 (torch.Tensor): VMamba 的x模态特征图, shape: [B, C_2, H, W]
            stage_idx (int, optional): 当前 stage 的索引，预留用于后续可能的 stage-specific 逻辑
        Returns:
            delta_feat (torch.Tensor): 融合后的增量特征, shape: [B, C_1, H, W]
        """
        # Step 1: 通道对齐 (C_2 -> C_1)
        x2_aligned = self.align_conv(x2)

        # Step 2: 融合
        fused_feat = self.hga([x1, x2_aligned])

        # Step 3: 映射为增量特征 (Delta)
        # 经过零初始化的 out_conv，保证 delta_feat 在训练开始时接近 0
        delta_feat = self.out_conv(fused_feat)

        return delta_feat


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟参数
    B, H, W = 2, 96, 96
    C_sam = 112   # SAM 的通道数
    C_vm = 96     # VMamba 的通道数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化融合模块
    fusion = SAMFusion(c_1=C_sam, c_2=C_vm).to(device)

    # 模拟输入
    sam_feat = torch.randn(B, C_sam, H, W).to(device)
    vm_feat = torch.randn(B, C_vm, H, W).to(device)

    # 前向传播
    delta = fusion(sam_feat, vm_feat, stage_idx=0)

    print(f"--- Test Result ---")
    print(f"SAM feature shape:    {sam_feat.shape}")
    
    print(f"VM feature shape:     {vm_feat.shape}")
    print(f"SAMFusion delta shape:  {delta.shape}")
    

