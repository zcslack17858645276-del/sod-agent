# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    verbose=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, verbose=verbose)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model



def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def _load_checkpoint(model, ckpt_path, verbose=True):
    """
    Load checkpoint into model.
    Args:
        model:
            target model
        ckpt_path:
            checkpoint path
        verbose:
            True:
                print missing/unexpected keys for debugging
            False:
                silent loading (recommended for inference)
    Notes:
        - strict=False allows newly added modules to be initialized randomly
        - useful when loading SAM pretrained weights into modified architecture
    """
    if ckpt_path is None:
        if verbose:
            logging.warning(
                "No checkpoint path provided. "
                "Training from scratch."
            )
        return

    # load checkpoint
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True
    )

    # support different checkpoint formats
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # load weights
    incompatible = model.load_state_dict(
        state_dict,
        strict=False
    )

    missing_keys = incompatible.missing_keys
    unexpected_keys = incompatible.unexpected_keys

    if verbose:

        if len(missing_keys) > 0:
            logging.warning(
                "The following keys are missing in checkpoint "
                "and will be randomly initialized:"
            )

            for k in missing_keys:
                logging.warning(f"  {k}")

        if len(unexpected_keys) > 0:
            logging.warning(
                "The following keys in checkpoint are not used "
                "by the model and are ignored:"
            )

            for k in unexpected_keys:
                logging.warning(f"  {k}")

        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            logging.info("Checkpoint loaded successfully.")

    else:
        # inference mode only print success
        logging.info("Checkpoint loaded successfully.")