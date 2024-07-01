# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from models.encoder.mlae_vit import MLAE_ViT, TransformerBlock
from models.mtae import MTAE


__all__ = ['MLAE', 'mlae_vit_small_dec256d4b', 'mlae_vit_base_dec256d4b']


class MLAE(MTAE):
    def __init__(self,
                 seq_len: int = 2250,
                 patch_size: int = 1,
                 num_leads: int = 12,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: int = 4,
                 qkv_bias: bool = True,
                 norm_layer: nn.Module = nn.LayerNorm,
                 norm_pix_loss: bool = False):
        super(MTAE, self).__init__()
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'embed_dim': embed_dim,
                           'depth': depth,
                           'num_heads': num_heads,
                           'decoder_embed_dim': decoder_embed_dim,
                           'decoder_depth': decoder_depth,
                           'decoder_num_heads': decoder_num_heads,
                           'mlp_ratio': mlp_ratio,
                           'qkv_bias': qkv_bias,
                           'norm_layer': str(norm_layer),
                           'norm_pix_loss': norm_pix_loss}
        self.patch_size = patch_size
        self.num_patches = num_leads // patch_size
        # --------------------------------------------------------------------
        # MAE encoder specifics
        self.encoder = MLAE_ViT(seq_len=seq_len,
                                patch_size=patch_size,
                                num_leads=num_leads,
                                width=embed_dim,
                                depth=depth,
                                mlp_dim=mlp_ratio * embed_dim,
                                heads=num_heads,
                                qkv_bias=qkv_bias)
        self.to_patch_embedding = self.encoder.to_patch_embedding
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # MAE decoder specifics
        self.to_decoder_embedding = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([TransformerBlock(input_dim=decoder_embed_dim,
                                                              output_dim=decoder_embed_dim,
                                                              hidden_dim=decoder_embed_dim * mlp_ratio,
                                                              heads=decoder_num_heads,
                                                              dim_head=64,
                                                              qkv_bias=qkv_bias)
                                             for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, seq_len)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def patchify(self, series):
        """
        series: (batch_size, num_leads, seq_len)
        x: (batch_size, n, patch_size * seq_len)
        """
        p = self.patch_size
        assert series.shape[2] % p == 0
        x = rearrange(series, 'b (n p) t -> b n (p t)', p=p)
        return x

    def unpatchify(self, x):
        """
        x: (batch_size, n, patch_size * seq_len)
        series: (batch_size, num_leads, seq_len)
        """
        series = rearrange(x, 'b n (p t) -> b (n p) t')
        return series


def mlae_vit_small_dec256d4b(**kwargs):
    model = MLAE(embed_dim=384,
                 depth=12,
                 num_heads=6,
                 decoder_embed_dim=256,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 **kwargs)
    return model


def mlae_vit_base_dec256d4b(**kwargs):
    model = MLAE(embed_dim=768,
                 depth=12,
                 num_heads=12,
                 decoder_embed_dim=256,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 **kwargs)
    return model
