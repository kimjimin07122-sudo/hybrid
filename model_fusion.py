import torch
import torch.nn as nn
import sys
import os

# SkySense 소스코드 경로 등록
SKYSENSE_PATH = "/home/kim/research/hybrid/skysense/SkySense"
if SKYSENSE_PATH not in sys.path:
    sys.path.insert(0, SKYSENSE_PATH)

from models.swin_transformer_v2 import SwinTransformerV2

class SkySense_Backbone_with_Lateral(nn.Module):
    def __init__(self, weights_path="/home/kim/research/hybrid/weights/skysense_model_backbone_hr.pth"):
        super().__init__()
        
        # 1. 커스텀 거인족(HR) 딕셔너리 정의 (에러 로그 역산 결과 반영)
        custom_arch = {
            'embed_dims': 352,
            'depths': [2, 2, 18, 2],
            'num_heads': [8, 16, 32, 64],
            'extra_norm_every_n_blocks': 6
        }

        # 2. 백본 생성
        self.backbone = SwinTransformerV2(
            arch=custom_arch, 
            img_size=640,          # VRAM 한계로 640 하향 조정
            out_indices=(1, 2, 3), # P3, P4, P5 추출
            drop_path_rate=0.2
        )
        
        # 3. 1x1 컨볼루션 (Lateral Layers) - 채널 압축 어댑터
        # HR 백본의 거대한 출력 [704, 1408, 2816] -> YOLOv26 헤드 규격 [256, 512, 1024]
        self.lateral_p3 = nn.Conv2d(704, 256, 1)
        self.lateral_p4 = nn.Conv2d(1408, 512, 1)
        self.lateral_p5 = nn.Conv2d(2816, 1024, 1)

        # 4. 가중치 로드
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # 비지도 학습용 가중치이므로 껍데기가 완벽히 일치하지 않음. strict=False 필수.
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"✅ SkySense 거인족(HR) 가중치 이식 성공: {weights_path}")
        else:
            print(f"❌ 치명적 에러: 가중치 파일을 찾을 수 없습니다. 경로를 확인하십시오: {weights_path}")

    def forward(self, x):
        # x: [Batch, 3, 640, 640]
        feats = self.backbone(x)
        
        # 특징맵 채널 압축 후 리턴
        out_p3 = self.lateral_p3(feats[0])
        out_p4 = self.lateral_p4(feats[1])
        out_p5 = self.lateral_p5(feats[2])
        
        return [out_p3, out_p4, out_p5]