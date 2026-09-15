"""Ultrasound encoder E_US (README.md section 15.1), backed by USFM's pretrained ViT-B/16.

USFM ("UltraSound Foundation Model", github.com/openmedlab/USFM) ships only a
single pretrained backbone: a ViT-B/16 trained with masked image modeling on
ultrasound images normalized with ImageNet statistics after being replicated to
3 channels (see USFM/usdsgen/data/datasets.py).

Requires the sibling USFM checkout's `usdsgen` package importable, e.g.:
    pip install --no-deps -e /path/to/USFM
(--no-deps: usdsgen's own requirements.txt pulls in mmsegmentation/lightning/etc,
none of which this file's import path needs).
"""

import logging

import torch
from torch import nn
from usdsgen.modules.backbone.vision_transformer import VisionTransformer
from usdsgen.utils.modelutils import remap_pretrained_keys_vit

_logger = logging.getLogger(__name__)

# Fixed by the released USFM_latest.pth checkpoint itself (patch_embed.proj.weight,
# rel_pos_bias table, per-block gamma_1/gamma_2, mlp fc1 width all match ViT-B/16);
# only img_size varies with the caller's Config, and the relative position bias
# is geometrically interpolated to match by remap_pretrained_keys_vit.
_EMBED_DIM = 768
_DEPTH = 12
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class USFMEncoder(nn.Module):
    """Per-frame USFM ViT-B/16 features, concatenated over the observation history."""

    def __init__(self, config, pretrained=None, freeze=True):
        super().__init__()
        self.history = config.history
        self._freeze = freeze
        self.backbone = VisionTransformer(
            img_size=config.image_size,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=_EMBED_DIM,
            depth=_DEPTH,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            init_values=0.1,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            # False keeps the pretrained cls-token + its pretrained final norm;
            # True would leave fc_norm randomly initialized and silently drop
            # the checkpoint's own norm.weight/bias (verified against
            # USFM_latest.pth: use_mean_pooling=False loads every relevant key).
            use_mean_pooling=False,
        )
        if pretrained is not None:
            self._load_pretrained(pretrained)
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.project = nn.Sequential(
            nn.Linear(config.history * _EMBED_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, config.feature_dim),
            nn.SiLU(),
        )

    def _load_pretrained(self, path):
        checkpoint = remap_pretrained_keys_vit(
            self.backbone, torch.load(path, map_location="cpu"), _logger
        )
        missing, unexpected = self.backbone.load_state_dict(checkpoint, strict=False)
        # relative_position_index is a deterministic non-persistent buffer
        # (recomputed from window_size, never checkpointed); mask_token is the
        # MAE decoder's, unused by a bare encoder.
        allowed_missing = {f"blocks.{i}.attn.relative_position_index" for i in range(_DEPTH)}
        allowed_unexpected = {"mask_token"}
        if set(missing) - allowed_missing or set(unexpected) - allowed_unexpected:
            raise ValueError(
                f"USFM checkpoint at {path} does not match the expected ViT-B/16 "
                f"architecture: missing={missing} unexpected={unexpected}"
            )

    def train(self, mode=True):
        super().train(mode)
        if self._freeze:
            self.backbone.eval()  # frozen backbone: never enable dropout/stochastic depth
        return self

    def forward(self, image):
        b, history, h, w = image.shape
        if history != self.history:
            raise ValueError(f"Expected {self.history} history frames, got {history}")
        frames = image.reshape(b * history, 1, h, w).repeat(1, 3, 1, 1)
        frames = (frames - self.mean) / self.std
        features = self.backbone(frames)
        return self.project(features.reshape(b, history * _EMBED_DIM))
