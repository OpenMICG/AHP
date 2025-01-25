import logging
from functools import partial

import torch
import torch.nn as nn
from models.ahp_adapter.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(1, 1),
                                      # (h // 8, w // 8),
                                      (h // 16, w // 16)],
                                      # (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    extra_reference_point = torch.zeros((1, 1, 1, 2), device=x.device)
    reference_points = torch.cat([extra_reference_point, reference_points], dim=1)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(1, 1),(h // 16, w // 16 )], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(1, 1),
                                             # (h // 8, w // 8),
                                             (h // 16, w // 16)],
                                             # (h // 32, w // 32)],
                                            x.device)
    # print("reference_points_2.shape:",reference_points.shape)

    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # print("N.shape:",N)
        n = (N - 1) // 21
        # print("N.shape:",n)

        x_cls = x[:, :1, :].transpose(1, 2).view(B, C, 1, 1).contiguous()
        x1 = x[:, 1:, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x_cls = self.dwconv(x_cls).flatten(2).transpose(1, 2)
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x = torch.cat([x_cls, x1], dim=1)
        return x

lass Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=2, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=2, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, n_levels=2, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, register_blk=-1):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W, register_blk==idx)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        # self.conv4 = nn.Sequential(*[
        #     nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=True),
        #     nn.SyncBatchNorm(4 * inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

        # self.conv1_emb = nn.Sequential(*[
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=4, stride=4, padding=1, bias=True),
        #     nn.SyncBatchNorm(embed_dim),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv2_emb = nn.Sequential(*[
        #     nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
        #     nn.SyncBatchNorm(embed_dim),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv4_emb = nn.Sequential(*[
        #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2, bias=True),
        #     nn.SyncBatchNorm(embed_dim),
        #     nn.ReLU(inplace=True)
        # ])

        # self.dense1 = nn.Linear(embed_dim, embed_dim)
        # self.dense2 = nn.Linear(embed_dim, embed_dim)
        self.dense3 = nn.Linear(embed_dim, embed_dim)
        # self.dense4 = nn.Linear(embed_dim, embed_dim)
        # self.activation1 = nn.Tanh()
        # self.activation2 = nn.Tanh()
        self.activation3 = nn.Tanh()
        # self.activation4 = nn.Tanh()
        # self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dim))


    def forward(self, x):
        # cls_token1 = self.cls_token1.expand(
        #     x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # cls_token2 = self.cls_token2.expand(
        #     x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token3 = self.cls_token3.expand(
            x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # cls_token4 = self.cls_token4.expand(
        #     x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        # c4 = self.conv4(c3)

        # c1 = self.fc1(c1)
        # c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        # c4 = self.fc4(c4)

        # c1 = self.conv1_emb(c1)
        # c2 = self.conv2_emb(c2)
        # c4 = self.conv4_emb(c4)

        bs, dim, _, _ = c3.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)
        # c2 = c2.view(bs, dim, -1).transpose(1, 2)
        c3 = c3.view(bs, dim, -1).transpose(1, 2)
        # c4 = c4.view(bs, dim, -1).transpose(1, 2)

        # pooled_output1 = self.dense1(cls_token1)
        # pooled_output1 = self.activation1(pooled_output1)
        # pooled_output2 = self.dense2(cls_token2)
        # pooled_output2 = self.activation2(pooled_output2)
        pooled_output3 = self.dense3(cls_token3)
        pooled_output3 = self.activation3(pooled_output3)
        # pooled_output4 = self.dense4(cls_token4)
        # pooled_output4 = self.activation4(pooled_output4)

        # c1 = torch.cat([pooled_output1, c1], dim=1)
        # c2 = torch.cat([pooled_output2, c2], dim=1)
        c3 = torch.cat([pooled_output3, c3], dim=1)
        # c4 = torch.cat([pooled_output4, c4], dim=1)

        # print("a1.shape:", c1.shape)
        # print("a2.shape:", c2.shape)
        # print("a3.shape:", c3.shape)
        # print("a4.shape:", c4.shape)

        return c3
