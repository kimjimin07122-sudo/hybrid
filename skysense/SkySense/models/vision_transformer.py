import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# 1. MMEngine 기반 필수 모듈
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init, kaiming_init
from mmengine.runner import load_state_dict

# 2. MMCV-Lite 기반 필수 레이어
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

# 3. MMSegmentation 1.x 호환성 해결 (PatchEmbed & resize)
try:
    # MMSegmentation 1.x 표준 경로
    from mmseg.models.utils import PatchEmbed
    from mmseg.utils import resize
    from mmengine.logging import MMLogger
    def get_root_logger():
        return MMLogger.get_current_instance().logger
except ImportError:
    # 설치 환경에 따른 예외 처리
    try:
        from mmcv.cnn.bricks.transformer import PatchEmbed
        from torch.nn.functional import interpolate as resize
    except ImportError:
        from mmengine.model import PatchEmbed
        resize = F.interpolate

    import logging
    def get_root_logger():
        return logging.getLogger()

# 4. 기타 유틸리티
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """내부적으로 trunc_normal_init을 활용한 헬퍼 함수"""
    with torch.no_grad():
        return trunc_normal_init(tensor, mean=mean, std=std, a=a, b=b)

# --- 모델 클래스 정의 ---

class TransformerEncoderLayer(BaseModule):
    """SkySense 위성 영상 처리를 위한 최적화된 Transformer Encoder Layer"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(embed_dims=embed_dims,
                 num_heads=num_heads,
                 attn_drop=attn_drop_rate,
                 proj_drop=drop_rate,
                 batch_first=batch_first,
                 bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(embed_dims=embed_dims,
                 feedforward_channels=feedforward_channels,
                 num_fcs=num_fcs,
                 ffn_drop=drop_rate,
                 dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                 if drop_path_rate > 0 else None,
                 act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            x = _inner_forward(x)
        return x


class VisionTransformer(BaseModule):
    """SkySense 백본: 위성 영상 특화 ViT 구조"""

    def __init__(self,
                 img_size=64,
                 patch_size=4,
                 in_channels=10,
                 embed_dims=1024,
                 num_layers=24,
                 num_heads=16,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        
        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
        )

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        else:
            self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims=embed_dims,
                                        num_heads=num_heads,
                                        feedforward_channels=mlp_ratio * embed_dims,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_rate=drop_rate,
                                        drop_path_rate=dpr[i],
                                        num_fcs=num_fcs,
                                        qkv_bias=qkv_bias,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg,
                                        with_cp=with_cp,
                                        batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    def init_weights(self):
        from mmengine.runner.checkpoint import load_checkpoint
        if self.init_cfg and self.init_cfg.get('type') == 'Pretrained':
            logger = get_root_logger()
            load_checkpoint(self, self.init_cfg['checkpoint'], map_location='cpu', logger=logger)
        else:
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            super(VisionTransformer, self).init_weights()

    def forward(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
            
            if i in self.out_indices:
                if self.with_cls_token:
                    out = x[:, 1:]
                else:
                    out = x
                
                # YOLOv26 입력을 위해 [B, C, H, W]로 변환
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            pos_h = self.img_size[0] // self.patch_size
            pos_w = self.img_size[1] // self.patch_size
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, 1:].reshape(1, pos_h, pos_w, -1).permute(0, 3, 1, 2)
        pos_embed_weight = resize(pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)