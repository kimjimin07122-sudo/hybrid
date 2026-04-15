import torch
import torch.nn as nn
import sys
import os

# 윈도우 경로 설정 (C:\hybrid 기준)
SKYSENSE_PATH = r"C:/hybrid/skysense/SkySense"
if SKYSENSE_PATH not in sys.path:
    sys.path.insert(0, SKYSENSE_PATH)

from models.swin_transformer_v2 import SwinTransformerV2

class SkySense_Backbone_with_Lateral(nn.Module):
    def __init__(self, weights_path=r"C:/hybrid/weights/skysense_model_backbone_hr.pth"):
        super().__init__()
        
        # 1. 거인족(HR) 아키텍처 프리셋 (에러 로그 기반 확정값)
        custom_arch = {
            'embed_dims': 352,
            'depths': [2, 2, 18, 2],
            'num_heads': [8, 16, 32, 64],
            'extra_norm_every_n_blocks': 6
        }

        # 2. 백본 생성
        self.backbone = SwinTransformerV2(
            arch=custom_arch, 
            img_size=640, 
            out_indices=(1, 2, 3), 
            drop_path_rate=0.2
        )
        
        # 3. 채널 매칭용 1x1 Conv (Lateral Layers)
        # SkySense HR [704, 1408, 2816] -> YOLOv26 [256, 512, 1024]
        self.lateral_p3 = nn.Conv2d(704, 256, 1)
        self.lateral_p4 = nn.Conv2d(1408, 512, 1)
        self.lateral_p5 = nn.Conv2d(2816, 1024, 1)

        # 4. 가중치 로드
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # 비지도 학습 가중치이므로 strict=False 유지
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ SkySense 거인족(HR) 가중치 로드 성공: {weights_path}")
        else:
            print(f"❌ 가중치 파일 없음: {weights_path}")

    def forward(self, x):
        feats = self.backbone(x)
        out_p3 = self.lateral_p3(feats[0])
        out_p4 = self.lateral_p4(feats[1])
        out_p5 = self.lateral_p5(feats[2])
        return [out_p3, out_p4, out_p5]