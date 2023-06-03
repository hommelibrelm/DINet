import os
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from collections import OrderedDict

from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead
from timm.models.vision_transformer import trunc_normal_
from ...utils.positional_encoding import get_2d_sincos_pos_embed

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x  
    

class Interactive_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.act = nn.GELU()

        self.aux_proj = nn.Linear(self.head_dim, dim, bias=True)
        self.aux_norm = nn.LayerNorm(self.head_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_heads, dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N - 1))
        aux_dim = C // self.num_heads

        # Add positional embeddings
        ori_x = x
        pos_embed = self.pos_embed.expand(B, -1, -1)
        x = x.reshape(B, N, self.num_heads, aux_dim).permute(0, 2, 1, 3)
        x = x.mean(dim=2)
        x = self.aux_proj(x).reshape(B, -1, self.num_heads, aux_dim)
        x = self.act(self.aux_norm(x)).flatten(2)
        x = x + pos_embed
        x = torch.cat([ori_x, x], dim=1)

        # Compute query, key, and value
        qkv = self.qkv(x).reshape(B, N + self.num_heads, 3, self.num_heads, aux_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention weights and apply dropout
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute output and apply dropout
        x = (attn @ v).transpose(1, 2).reshape(B, N + self.num_heads, C)
        x = self.proj(x)
        cls, patch, aux = torch.split(x, [1, N - 1, self.num_heads], dim=1)
        cls = cls + torch.mean(aux, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)
        x = self.proj_drop(x)

        return x

class Interactive_Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., qk_scale=None, 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Interactive_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_hidden_dim = mlp_hidden_dim
        

    def forward(self, x):
        B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
@HEADS.register_module()
class InterBBoxHead(BBoxHead):
    def __init__(self,
                 in_channels,
                 img_size=224,
                 patch_size=16, 
                 embed_dim=256, 
                 depth=4,
                 num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # MAE decoder specifics
        self.norm = norm_layer(in_channels)
        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList([
                Interactive_Block(
                    embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
        ])
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_box_norm = norm_layer(embed_dim)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(embed_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = nn.Linear(embed_dim, out_dim_reg)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def init_weights(self, pretrained):
        logger = get_root_logger()
        if os.path.isfile(pretrained):
            logger.info('loading checkpoint for {}'.format(self.__class__))
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # TODO: match the decoder blocks, norm and head in the state_dict due to the different prefix
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('patch_embed') or k.startswith('blocks'):
                    continue
                elif k in ['pos_embed']:
                    continue
                else:
                    new_state_dict[k] = v
            load_state_dict(self, new_state_dict, strict=False, logger=logger)
        else:
            raise ValueError(f"checkpoint path {pretrained} is invalid")

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.decoder_pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.decoder_pos_embed
        class_pos_embed = self.decoder_pos_embed[:, 0]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward(self, x):
        B, C, W, H = x.shape

        # Flatten and transpose input tensor
        x = x.flatten(2).transpose(1, 2)

        # Apply normalization and embedding layers
        x = self.norm(x)
        x = self.decoder_embed(x)

        # Add positional encoding
        pos_encoding = self.interpolate_pos_encoding(x, W * self.patch_size, H * self.patch_size)[:, 1:, :]
        x = x + pos_encoding

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        # Compute class scores and bounding box predictions
        x = self.decoder_box_norm(x.mean(dim=1))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None

        return cls_score, bbox_pred
