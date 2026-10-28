import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from Vmamba.models.encoders.vmamba import Backbone_VSSM

from sam2.sam2_model.Module.HeterogeneousFusionAggregationModule import HFAM as FusedMambaBlock

class RGBXMamba(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=128,
                 pretrained=None,
                 mlp_ratio=0.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[448, 448],
                 patch_size=4,
                 drop_path_rate=0.6,
                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm_r = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.pred_saliency = nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, bias=False)

        self.modality_mamba1 = FusedMambaBlock(
            in_channels=768, 
            out_channels=768, 
            num_layers=3,
        )

        self.modality_mamba2 = FusedMambaBlock(
            in_channels=384, 
            out_channels=384, 
            num_layers=3,
        )

        self.modality_mamba3 = FusedMambaBlock(
            in_channels=192, 
            out_channels=192, 
            num_layers=3,
        )

        self.modality_mamba4 = FusedMambaBlock(
            in_channels=96, 
            out_channels=96, 
            num_layers=3,
        )
        
        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

    def forward(self, x_rgb, x_e):
        B = x_rgb.shape[0]
        outs_fused = []
        
        outs_rgb = self.vssm_r(x_rgb) # B x C x H x W
        outs_e = self.vssm_r(x_e) # B x C x H x W
        out_fused = self.modality_mamba4([outs_rgb[0], outs_e[0]])
        outs_fused.append(out_fused)
        out_fused = self.modality_mamba3([outs_rgb[1], outs_e[1]])
        outs_fused.append(out_fused)
        out_fused = self.modality_mamba2([outs_rgb[2], outs_e[2]])
        outs_fused.append(out_fused)
        out_fused = self.modality_mamba1([outs_rgb[3], outs_e[3]])
        outs_fused.append(out_fused)
        return outs_fused

class vssm_small(RGBXMamba):
    def __init__(self, cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained=cfg.vmamba_pretrained_model,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )