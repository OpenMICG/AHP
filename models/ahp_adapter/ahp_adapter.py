# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .base.vit import TIMMVisionTransformer
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

# from adapters.adapter_controller import AdapterController
# from adapters.adapter_configuration import AdapterConfig

import numpy as np


_logger = logging.getLogger(__name__)

class AHPAdapter(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, *args, **kwargs):

        super().__init__(num_heads=num_heads, *args, **kwargs)

        # self.num_classes = 80
        self.num_features = 768
        # self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(1, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])
        # self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)

        # self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        self.linear_1 = nn.Linear(768 * 2, 768)
        self.linear_2 = nn.Linear(768 * 2, 768)
        self.linear_3 = nn.Linear(768 * 2, 768)
        self.linear_4 = nn.Linear(768 * 2, 768)
        self.linear_cls = nn.Linear(768, 768)

        self.activation = nn.Tanh()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self,c):
        c = c + self.level_embed[0]
        return c.clone()


    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c = self.spm(x)
        c = self._add_level_embed(c)
        # c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        # add [CLS] Token
        x_cls_token = self.cls_token.expand(
            x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat([x_cls_token, x], dim=1)
        # c = torch.cat([cls_token, c3], dim=1)

        # pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + self.pos_embed)


        # Interaction
        register_blk=-1
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if i == 3:
                register_blk=2
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W, register_blk=register_blk)

        if self.add_vit_feature:
            x = x + c
        # Final Norm
        fx = F.normalize(x, dim=-1)
        return fx
