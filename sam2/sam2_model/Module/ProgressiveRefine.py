from sam2.sam2_model.Module.ConvBNReLU import ConvBNReLU
import torch
import torch.nn as nn
import torch.nn.functional as F
from Vmamba.models.encoders.vmamba import VSSBlock


class ProgressiveRefine(nn.Module):
    def __init__(self, in_channels, high_res_channels=None, hidden_dim=192, mlp_ratio=2.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1×1 Conv 降通道（concat 后用）
        self.reduce_highres = None
        if high_res_channels is not None:
            self.reduce_highres = nn.Conv2d(high_res_channels, hidden_dim//2, 1, 1, 0)

        # stage1: 低分辨率 -> 中分辨率
        self.stage1_conv = nn.Sequential(
            ConvBNReLU(in_channels + (hidden_dim//2 if high_res_channels else 0), hidden_dim, 3, 1, 1),
            ConvBNReLU(hidden_dim, hidden_dim, 3, 1, 1)
        )

        # 中间尺度 VSSBlock
        self.vss1 = VSSBlock(hidden_dim, mlp_ratio=mlp_ratio)

        # stage2: 中分辨率 -> 高分辨率
        self.branch0 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0,1)),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1,0)),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=3, dilation=3),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 5), padding=(0,2)),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 1), padding=(2,0)),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=5, dilation=5),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 7), padding=(0,3)),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(7, 1), padding=(3,0)),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=7, dilation=7),
        )

        
        self.conv_cat = nn.Conv2d(hidden_dim * 4, hidden_dim, 3, padding=1)
        self.conv_res = nn.Conv2d(hidden_dim, hidden_dim, 1)

        # 高分辨率输出
        self.pred = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, low_res_features, high_res_features=None):
        '''
        print(f"DEBUG: Low-res feature: {low_res_features.shape}")
        if high_res_features is not None:
            for i, feat in enumerate(high_res_features):
                print(f"DEBUG: High-res feature {i} shape: {feat.shape}")
        '''
        # --- Step1: 上采样到中分辨率 ---
        x = F.interpolate(low_res_features, scale_factor=2, mode="bilinear", align_corners=False)
        
        if high_res_features is not None:
            high_feat = high_res_features[0]
            high_feat = F.interpolate(high_feat, size=x.shape[2:], mode="bilinear", align_corners=False)
            if self.reduce_highres:
                high_feat = self.reduce_highres(high_feat)
            x = torch.cat([x, high_feat], dim=1)

        # stage1 conv 精细化
        x = self.stage1_conv(x)

        # VSSBlock 处理 (B,H,W,C)
        x = x.permute(0,2,3,1)  # [B,H,W,C]
        x = self.vss1(x)
        x = x.permute(0,3,1,2)  # [B,C,H,W]

        # --- Step2: 上采样到高分辨率 ---
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x0, x1, x2, x3], dim=1)
        x_cat = self.conv_cat(x_cat)
        x = self.relu(x_cat + self.conv_res(x))
        pre = x

        # 输出单通道 Mask
        x = self.pred(x)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 64, 96, 96)

    model = DoubleTransposedUpsample(
        in_channels=64,
        out_channels=32
    )

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)