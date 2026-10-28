# vmamba只使用encoder结构，然后从encoder得到多尺度的融合结果，这个结果输送到adapter中进行再次融合
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.modeling.backbones.hieradet import do_pool, window_partition, window_unpartition
from sam2.sam2_model.Module.GRCM import SAMFusion as HFusion

from Vmamba.models.Vmamba_encoder_dual import Model as VmambaModel
from sam2.sam2_model.Module.ProgressiveRefine import ProgressiveRefine
from sam2.sam2_model.Module.CPD import CPD_VMamba as VMambaDecoder


class Model(nn.Module):

    def __init__(self, cfg, verbose=True) -> None:
        super().__init__()
        self.cfg = cfg
        self.freeze_dformer = getattr(cfg, "freeze_dformer", True)
        project_root = Path(__file__).resolve().parents[2]
        sam2_pkg_root = Path(__file__).resolve().parents[1]

        sam2_cfg = self._normalize_hydra_config_name(self.cfg.sam2_cfg, sam2_pkg_root)
        sam2_ckpt = self._resolve_project_path(self.cfg.sam2_ckpt, project_root)

        self.model = build_sam2(sam2_cfg, sam2_ckpt, verbose=verbose)
        self.vmamba = VmambaModel(cfg=self.cfg)

        block_nums = {
            'hiera_b+': [2, 3, 16,3],
            'hiera_l': [2, 6, 36, 4],
        }

        channels_sam = {
            'hiera_b+': [112, 224, 448, 896],
            'hiera_l': [144, 288, 576, 1152],
        }

        channels_vssm = {
            'vssm_small': [96, 192, 384, 768],
        }

        backbone_name = self.cfg.backbone_variant

        self.lenth_stage1 = block_nums[backbone_name][0]
        self.lenth_stage2 = block_nums[backbone_name][1]
        self.lenth_stage3 = block_nums[backbone_name][2]
        self.lenth_stage4 = block_nums[backbone_name][3]
        
        self.block_index = [
            self.lenth_stage1,
            self.lenth_stage1 + self.lenth_stage2,
            self.lenth_stage1 + self.lenth_stage2 + self.lenth_stage3,
            self.lenth_stage1 + self.lenth_stage2 + self.lenth_stage3 + self.lenth_stage4,
        ]
        self.stage_channels = channels_sam[backbone_name]
        self.stage_adapters = nn.ModuleList([
          HFusion(self.stage_channels[i], channels_vssm['vssm_small'][i]) for i in range(4)
        ])
        self.stage3_adapter_every = int(getattr(self.cfg, "stage3_adapter_every", 0))
        if self.stage3_adapter_every < 1:
            self.stage3_adapter_every = 0
        stage3_channels = self.stage_channels[2]
        stage3_vssm_channels = channels_vssm['vssm_small'][2]
        stage3_inner_count = 0
        if self.stage3_adapter_every:
            stage3_inner_count = max((self.lenth_stage3 - 1) // self.stage3_adapter_every, 0)
        self.stage3_inner_adapters = nn.ModuleList(
            [HFusion(stage3_channels, stage3_vssm_channels) for _ in range(stage3_inner_count)]
        )

        self.refine = ProgressiveRefine(in_channels=32, high_res_channels=32, hidden_dim=64)
        self.vmamba_decoder = VMambaDecoder(channel=32)

    @staticmethod
    def _resolve_project_path(path_value, project_root: Path):
        if not path_value:
            return path_value

        path = Path(path_value)
        if path.is_absolute() and path.exists():
            return str(path)

        candidate_relative = project_root / str(path_value).lstrip("/")
        if candidate_relative.exists():
            return str(candidate_relative)

        return str(path)

    @staticmethod
    def _normalize_hydra_config_name(config_value, sam2_pkg_root: Path):
        if not config_value:
            return config_value

        raw = str(config_value)
        raw_path = Path(raw)

        if raw_path.is_absolute():
            try:
                return raw_path.resolve().relative_to(sam2_pkg_root.resolve()).as_posix()
            except ValueError:
                return raw_path.name

        normalized = raw.lstrip("/").replace("\\", "/")
        if normalized.startswith("sam2/"):
            candidate = normalized[len("sam2/"):]
            if (sam2_pkg_root / candidate).exists():
                return candidate

        if (sam2_pkg_root / normalized).exists():
            return normalized

        fallback_name = Path(normalized).name
        if (sam2_pkg_root / fallback_name).exists():
            return fallback_name

        return normalized

    def setup(self):
        if self.cfg.freeze_image_encoder:
            print("冻结编码器")
            for name, param in self.model.named_parameters():
                if "sam_mask_decoder" not in name:
                    param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        '''
        if self.freeze_dformer:
            # Keep DFormer in eval mode so BN stats are not updated during training.
            self.dformer.eval()
        '''
        return self
        
    def forward(self, rgb, modality=None, return_intermediates: bool = False):
        # print(f"DEBUG: Initial rgb shape: {rgb.shape}") 
        #print('shape of rgb and modality: ', rgb.shape, modality.shape)
        outs = self.vmamba(rgb, modality)
        vssm = outs
        input_hw = rgb.shape[-2:]
        self.model.image_size = self.cfg.input_size
        target_size = self.model.image_size
        if isinstance(target_size, int):
            target_hw = (target_size, target_size)
        else:
            target_hw = tuple(target_size)
        if input_hw != target_hw:
            rgb = F.interpolate(rgb, size=target_hw, mode="bilinear", align_corners=False)

        rgb = self.model.image_encoder.trunk.patch_embed(rgb)
        rgb = rgb + self.model.image_encoder.trunk._get_pos_embed(rgb.shape[1:3])

        rgb_outputs = []
        stage_adapter_inputs = []
        stage_adapter_outputs = []
        stage_end_to_idx = {
            end_idx: stage_idx
            for stage_idx, end_idx in enumerate(self.model.image_encoder.trunk.stage_ends)
        }
        stage3_start = self.model.image_encoder.trunk.stage_ends[1] + 1
        stage3_end = self.model.image_encoder.trunk.stage_ends[2]

        for i, blk in enumerate(self.model.image_encoder.trunk.blocks):
            rgb_shortcut = rgb  # B, H, W, C
            rgb = blk.norm1(rgb)
            # Skip connection
            if blk.dim != blk.dim_out:
                rgb_shortcut = do_pool(blk.proj(rgb), blk.pool)
            # Window partition
            window_size = blk.window_size
            if window_size > 0:
                Hb, Wb = rgb.shape[1], rgb.shape[2]
                rgb, pad_hw = window_partition(rgb, window_size)

            # Window Attention + Q Pooling (if stage change)
            B, H, W, _ = rgb.shape
            # qkv with shape (B, H * W, 3, nHead, C)
            qkv = blk.attn.qkv(rgb).reshape(B, H * W, 3, blk.attn.num_heads, -1)
            # q, k, v with shape (B, H * W, nheads, C)
            q, k, v = torch.unbind(qkv, 2)
            # Q pooling (for downsample at stage changes)
            if blk.attn.q_pool:
                q = do_pool(q.reshape(B, H, W, -1), blk.attn.q_pool)
                H, W = q.shape[1:3]  # downsampled shape
                q = q.reshape(B, H * W, blk.attn.num_heads, -1)
            # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
            rgb = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
            # Transpose back
            rgb = rgb.transpose(1, 2)
            rgb = rgb.reshape(B, H, W, -1)
            rgb = blk.attn.proj(rgb)

            if blk.q_stride:
                # Shapes have changed due to Q pooling
                window_size = blk.window_size // blk.q_stride[0]
                Hb, Wb = rgb_shortcut.shape[1:3]

                pad_h = (window_size - Hb % window_size) % window_size
                pad_w = (window_size - Wb % window_size) % window_size
                pad_hw = (Hb + pad_h, Wb + pad_w)
            # Reverse window partition
            if blk.window_size > 0:
                rgb = window_unpartition(rgb, window_size, pad_hw, (Hb, Wb))

            rgb = rgb_shortcut + blk.drop_path(rgb)
            adapter_output_nchw = None
            if i in stage_end_to_idx:
                current_stage = stage_end_to_idx[i]
                # HFusion expects NCHW; SAM trunk uses BHWC.
                rgb_nchw = rgb.permute(0, 3, 1, 2)
                adapter_output_nchw = self.stage_adapters[current_stage](rgb_nchw, vssm[current_stage], current_stage)
                adapter_output = adapter_output_nchw.permute(0, 2, 3, 1)
                mlp_output = blk.drop_path(blk.mlp(blk.norm2(rgb + adapter_output)))
                rgb = rgb + adapter_output + mlp_output
                #rgb = adapter_output + mlp_output
                stage_adapter_inputs.append(rgb_nchw)
                stage_adapter_outputs.append(adapter_output_nchw)
            elif stage3_start <= i < stage3_end:
                rgb_nchw = rgb.permute(0, 3, 1, 2)
                stage3_local_idx = i - stage3_start
                if self.stage3_adapter_every and (stage3_local_idx + 1) % self.stage3_adapter_every == 0:
                    adapter_idx = stage3_local_idx // self.stage3_adapter_every
                    adapter_output_nchw = self.stage3_inner_adapters[adapter_idx](rgb_nchw, vssm[2], 2)
                    adapter_output = adapter_output_nchw.permute(0, 2, 3, 1)
                    mlp_output = blk.drop_path(blk.mlp(blk.norm2(rgb + adapter_output)))
                    rgb = rgb + adapter_output + mlp_output
                else:
                    mlp_output = blk.drop_path(blk.mlp(blk.norm2(rgb)))
                    rgb = rgb + mlp_output
            else:
                mlp_output = blk.drop_path(blk.mlp(blk.norm2(rgb)))
                rgb = rgb + mlp_output

            

            # [1, 7, 43, 47]
            if (i == self.model.image_encoder.trunk.stage_ends[-1]) or (
                    i in self.model.image_encoder.trunk.stage_ends and self.model.image_encoder.trunk.return_interm_layers
            ):
                feats_rgb = rgb.permute(0, 3, 1, 2)
                rgb_outputs.append(feats_rgb)
        
        rgb_feats, pos = self.model.image_encoder.neck(rgb_outputs)

        sam_before_fusion = [
            feat.clone()
            for feat in rgb_feats
        ]   

        if self.model.image_encoder.scalp > 0:
            # Discard the lowest resolution features
            rgb_feats = rgb_feats[: -self.model.image_encoder.scalp]

        rgb_out = {
            "backbone_fpn": rgb_feats,
        }

        if self.model.use_high_res_features_in_sam:
            rgb_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
                rgb_out["backbone_fpn"][0]
            )
            rgb_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
                rgb_out["backbone_fpn"][1]
            )
        rgb_out = rgb_out.copy()

        rgb_feature_maps = rgb_out["backbone_fpn"][-self.model.num_feature_levels:]

        # flatten NxCxHxW to HWxNxC
        rgb_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in rgb_feature_maps]

        if self.model.directly_add_no_mem_embed:
            rgb_vision_feats[-1] = rgb_vision_feats[-1] + self.model.no_mem_embed
        b = rgb_feature_maps[0].shape[0]
        rgb_feat_sizes = [(x.shape[-2], x.shape[-1]) for x in rgb_feature_maps]

        rgb_feats_after = [
            feat.permute(1, 2, 0).view(b, -1, *feat_size)
            for feat, feat_size in zip(rgb_vision_feats[::-1], rgb_feat_sizes[::-1])
        ][::-1]

        sam_after_fusion = [
            feat.clone()
            for feat in rgb_feats_after
        ]

        rgb_end_features = {"image_embed": rgb_feats_after[-1], "high_res_feats": rgb_feats_after[:-1]}

        rgb_high_res_features = [
            feat_level
            for feat_level in rgb_end_features["high_res_feats"]
        ]
        
        # NOTE: Prompt encoder path is disabled in this file.
        # Provide empty prompt embeddings so we can still run the SAM2 decoder.
        b, c, h, w = rgb_end_features["image_embed"].shape
        device = rgb_end_features["image_embed"].device
        sparse_embeddings = torch.zeros((b, 0, c), device=device)
        dense_embeddings = torch.zeros((b, c, h, w), device=device)
        image_pe = torch.zeros((1, c, h, w), device=device)
        
        low_res_masks_rgb, iou_predictions_rgb, _, _, upscaled_embedding = self.model.sam_mask_decoder(
            image_embeddings=rgb_end_features["image_embed"],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=rgb_high_res_features,
        )
        sam_out = self.refine(upscaled_embedding, rgb_high_res_features)

        vssm_out_prediction, vssm_out_detection = self.vmamba_decoder(vssm[1], vssm[2], vssm[3])

        if return_intermediates:
            return (
                sam_out,
                vssm,
                stage_adapter_inputs,
                stage_adapter_outputs,
                low_res_masks_rgb,
                vssm_out_prediction,
                vssm_out_detection
            )

        return (
            sam_out,
            vssm,
            vssm_out_prediction,
            vssm_out_detection
        )

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    # fp32情况下每个参数4Byte
    model_size_mb = total_params * 4 / 1024 / 1024

    print("\n" + "=" * 60)
    print(f"Total Params      : {total_params:,}")
    print(f"Trainable Params  : {trainable_params:,}")
    print(f"Frozen Params     : {frozen_params:,}")
    print(f"Model Size (FP32) : {model_size_mb:.2f} MB")
    print("=" * 60 + "\n")

    return total_params, trainable_params, frozen_params, model_size_mb

if __name__ == "__main__":
    from option import get_args_parser

    parser = argparse.ArgumentParser("Model shape test", add_help=True)
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--height", type=int, default=384, help="Input height")
    parser.add_argument("--width", type=int, default=384, help="Input width")
    args, _ = parser.parse_known_args()

    cfg = get_args_parser()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = Model(cfg)

    model.to(device)
    rgb = torch.randn(args.batch, 3, args.height, args.width, device=device)
    modality = torch.randn(args.batch, 3, args.height, args.width, device=device)

    with torch.no_grad():
        _ = model(rgb, modality)

    total_params, trainable_params, frozen_params, model_size_mb = count_parameters(model)

    model.eval()
    model.to(device)

    rgb = torch.randn(args.batch, 3, args.height, args.width, device=device)
    modality = torch.randn(args.batch, 3, args.height, args.width, device=device)

    #print(model)
    with open("model_structure.txt", "w") as f:
        f.write(str(model))
    with torch.no_grad():
        (
            sam_out,
            vssm,
            stage_adapter_inputs,
            stage_adapter_outputs,
            low_res_masks_rgb,
            vssm_out_prediction,
            vssm_out_detection,
        ) = model(rgb, modality, return_intermediates=True)

    print("sam_out:", sam_out.shape)
    #print("rgb_vssm:", [tuple(x.shape) for x in rgb_vssm])
    #print("e_vssm:", [tuple(x.shape) for x in e_vssm])
    #print("fused_vssm:", [tuple(x.shape) for x in fused_vssm])
    print("vssm:", [tuple(x.shape) for x in vssm])
    #print("edge_out:", tuple(edge_out.shape))
    print("low_res_masks_rgb:", tuple(low_res_masks_rgb.shape))
    print("vssm_out_prediction:", tuple(vssm_out_prediction.shape))
    print("vssm_out_detection:", tuple(vssm_out_detection.shape))

    for idx, (pre, post) in enumerate(zip(stage_adapter_inputs, stage_adapter_outputs)):
        print(f"stage {idx} adapter_in:", tuple(pre.shape))
        print(f"stage {idx} adapter_out:", tuple(post.shape))

