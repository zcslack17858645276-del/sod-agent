import torch
from torch import nn
from typing import List, Optional, Tuple, Type

# 假设这些工具类是从 sam2.modeling.sam2_utils 导入的
# 如果你在本地测试，请确保这些路径正确
from sam2.modeling.sam2_utils import LayerNorm2d, MLP

class PromptDecoder(nn.Module):
    """
    通用的提示解码器：既可以作为正提示解码器，也可以作为负提示解码器。
    
    输出：
        1. mask: 用于计算损失函数的像素级掩码 [B, 1, H, W]
        2. latent_token: 用于传递给下一个解码器的高维语义特征 [B, 1, C]
    """
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 1, # 前两个解码器通常只需要一个最强的token
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            use_high_res_features: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        # 基础 Token 嵌入
        self.iou_token = nn.Embedding(1, transformer_dim)
        # 我们只需要输出一个主要的 Mask Token 来作为下一级的引导
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 上采样模块 (将特征图放大到原图尺寸)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

        # Hypernetwork: 将 Mask Token 转化为预测 Mask 的权重
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_embeddings: [B, C, H, W]
            image_pe: [1, C, H, W]
            sparse_prompt_embeddings: [B, N, C] (来自点、框或预设的背景Prompt)

        Returns:
            mask: [B, 1, H, W] 用于监督学习
            latent_token: [B, 1, C] 用于传递给第三个解码器
        """
        # 1. 准备 Tokens
        # 构造 [iou_token, mask_token_0, mask_token_1, ...]
        output_tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight
        ], dim=0) # [1 + num_mask_tokens, C]
        
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        
        # 拼接输出 Token 和 输入的 Prompt Token
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 2. 准备图像特征
        src = image_embeddings
        b, c, h, w = src.shape
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # 3. Transformer 推理
        hs, src = self.transformer(src, pos_src, tokens)

        # 4. 提取关键信息
        # hs 的结构: [Batch, 1(iou) + num_mask_tokens + num_prompt_tokens, C]
        # 我们取第一个 mask token 作为引导信号 (latent_token)
        mask_tokens_out = hs[:, 1 : 1 + self.num_mask_tokens, :]
        latent_token = mask_tokens_out[:, 0:1, :] # [B, 1, C]

        # 5. 生成像素级 Mask (用于监督)
        # 使用 Hypernetwork 将 token 转化为权重
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 上采样特征图
        src_trans = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src_trans)
        else:
            # 如果使用了高分辨率特征，这里需要传入 high_res_features，逻辑略复杂，此处简化
            raise NotImplementedError("High-res features not implemented in this simplified version.")

        # 矩阵乘法生成 Mask: [B, Num_Masks, C] @ [B, C, H*W] -> [B, Num_Masks, H*W]
        b_up, c_up, h_up, w_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b_up, c_up, h_up * w_up)).view(b_up, -1, h_up, w_up)

        # 我们只返回第一个 mask 用于监督 (index 0)
        return masks[:, 0:1, :, :], latent_token