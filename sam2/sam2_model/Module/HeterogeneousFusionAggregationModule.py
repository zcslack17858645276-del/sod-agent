import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEnhancementModule(nn.Module):
    """
    Feature Enhancement Module

    Local branch:
        - spatial details
        - boundary information

    Global branch:
        - channel-wise semantic enhancement

    Output:
        same size as input
    """

    def __init__(self, channels, reduction=4):
        super().__init__()

        # local enhancement branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # global enhancement branch
        hidden_dim = channels // reduction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_branch = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.gelu = nn.GELU()


    def forward(self, x):
        identity = x

        # ---------------------
        # Local enhancement
        # ---------------------
        local_feat = self.local_branch(x)

        # ---------------------
        # Global enhancement
        # ---------------------
        global_weight = self.global_pool(x)
        global_weight = self.global_branch(global_weight)
        global_feat = x * global_weight

        # ---------------------
        # Fusion
        # ---------------------
        enhance_feat = torch.cat([local_feat, global_feat], dim=1)
        enhance_feat = self.fusion(enhance_feat)

        # ---------------------
        # Residual refinement
        # ---------------------
        out = identity + enhance_feat
        out = self.gelu(out)

        return out

class HFAM(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers=3):
        super(HFAM, self).__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.enhance_module = FeatureEnhancementModule(out_channels)
        
        # 1. 初始化 中心节点 特征的 MLP (Linear -> ReLU -> Linear)
        self.agg_mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

        # 2. 共享权重 W
        self.shared_weight = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        #self.shared_weight = nn.Conv2d()

        # 3. 注意力向量 a
        # 它负责计算 [W*H0, W*Hk] 的注意力得分
        self.attention_net = nn.Conv2d(2 * out_channels, 1, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, modality_features):
        """
        Args:
            modality_features: List[Tensor], 长度为 N 的列表，每个 Tensor 形状为 (B, C, H, W)
        Returns:
            f4: 聚合后的多模态特征 (B, C, H, W)
        """
        # N 是模态数量
        N = len(modality_features)
        B, C, H, W = modality_features[0].shape

        # --- Step 1: 初始化 Hub 特征 H0 ---
        # 将所有模态特征经过增强后求平均或拼接后通过 MLP 得到 H0
        modality_ef = []
        for feat in modality_features:
            modality_ef.append(self.enhance_module(feat))
        stacked_features = torch.stack(modality_ef, dim=0) # (N, B, C, H, W)
        avg_features = torch.mean(stacked_features, dim=0)      # (B, C, H, W)
        max_features, _ = torch.max(stacked_features, dim=0)    # (B, C, H, W)
        combined_features = (avg_features + max_features) * 0.5
        H0 = self.agg_mlp(combined_features)                   # (B, C, H, W)

        H_spokes = []
        for feat in modality_features:
            if feat.shape[1] != self.out_channels:
                H_spokes.append(F.interpolate(feat, size=(H, W)))
            else:
                H_spokes.append(feat)

        # --- Step 2: 迭代更新 L 层 ---
        for l in range(self.num_layers):
            # 先计算所有节点的共享权重变换 W*H
            W_H0 = self.shared_weight(H0)
            W_H_spokes = [self.shared_weight(hk) for hk in H_spokes]

            # 用于存储所有注意力系数的列表
            # alpha_list[0] 是 alpha_0,0, alpha_list[1] 是 alpha_0,1 ...
            alpha_list = []

            # 计算 H0 与每个 Hk 的注意力
            # 我们需要计算 alpha_{0,k} 对于所有 k=0...N
            # 为了实现 softmax，我们需要先收集所有的 logits
            all_logits = []

            # 计算 k=0 (H0 与自身的交互)
            concat_0 = torch.cat([W_H0, W_H0], dim=1) # (B, 2C, H, W)
            logit_0 = self.attention_net(concat_0)    # (B, 1, H, W)
            all_logits.append(logit_0)

            # 计算 k=1...N (H0 与 Hk 的交互)
            for k in range(1, N + 1):
                concat_k = torch.cat([W_H0, W_H_spokes[k-1]], dim=1) # (B, 2C, H, W)
                logit_k = self.attention_net(concat_k)               # (B, 1, H, W)
                all_logits.append(logit_k)

            # 执行 Softmax 归一化
            logits_tensor = torch.stack(all_logits, dim=0) 
            alphas = F.softmax(logits_tensor, dim=0) # (N+1, B, 1, H, W)

            # --- Step 3: 更新 H0 ---
            
            # 第一部分: alpha_0,0 * W_H0
            term_hub = alphas[0] * W_H0
            
            # 第二部分: sum(alpha_0,k * W_Hk)
            term_spokes = 0
            for k in range(1, N + 1):
                term_spokes += alphas[k] * W_H_spokes[k-1]
            
            H0 = H0 + self.relu(term_hub + term_spokes)

        return H0


# --- 测试代码 ---
if __name__ == "__main__":
    B, C, H, W = 2, 96, 96, 96
    # 模拟 3 个模态的输入 (例如 RGB, Depth, Thermal)
    modality_inputs = [
        torch.randn(B, C, H, W),
        torch.randn(B, C, H, W),
        #torch.randn(B, C, H, W)
    ]
    hfam = HFAM(in_channels=96, out_channels=96)
    output = hfam(modality_inputs)

    print(f"Input modality shape: {modality_inputs[0].shape}")
    print(f"Output HFAM shape: {output.shape}") # 应该是 (2, 96, 96, 96)