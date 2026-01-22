# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.hub import load_state_dict_from_url

from common.utils import LOGGER
from pathlib import Path
from urllib.parse import urlparse

def load_pretrained_weights(model, checkpoint_url, device='cpu'):
    parsed = urlparse(checkpoint_url)
    # Check if this is a URL (http/https)
    if parsed.scheme in ("http", "https"):
        pretrained_dict = load_state_dict_from_url(
        checkpoint_url,
        progress=True,
        check_hash=True,
        map_location=device,
    )
    else:
        ckpt_path = Path(checkpoint_url)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        pretrained_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(pretrained_dict, dict):
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        elif "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]
    
    load_state_dict_partial(model, pretrained_dict)
    print(f"Loaded weights from {checkpoint_url}")
    return model


def load_state_dict_partial(model, pretrained_dict):
    """
    Loads matching keys from pretrained_dict into model, ignoring mismatched layers.
    """
    model_dict = model.state_dict()
    matched = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    skipped = [k for k in pretrained_dict.keys() if k not in matched]
    model_dict.update(matched)
    model.load_state_dict(model_dict)

    LOGGER.info(
        f"Loaded {len(matched)}/{len(model_dict)} layers from checkpoint. "
        f"Skipped {len(skipped)} layers."
    )


def fuse_blocks(model: torch.nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'fuse'):
            module.fuse()
    return model