import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. 경로 및 실험 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'skysense', 'SkySense'))

try:
    from models.vision_transformer import VisionTransformer
except ImportError:
    print("❌ SkySense 모델을 로드할 수 없습니다. 경로 설정을 확인하십시오.")
    sys.exit()

# RTX 5070 최적화 설정
RESOLUTION = 1024  # 연구 목표 해상도
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_IMG_PATH = os.path.join(current_dir, 'datasets', 'DOTAv1', 'images', 'train', 'P0000.jpg')
SAVE_PATH = 'skysense_feature_1024.png'

# 2. 모델 준비 (논문용 표준 스펙으로 상향)
# embed_dims와 num_layers는 실제 사용할 하이브리드 모델의 스펙과 일치시켜야 함
model = VisionTransformer(
    img_size=RESOLUTION, 
    patch_size=16, 
    in_channels=3, 
    embed_dims=768, # ViT-Base 수준으로 상향
    num_layers=12
).to(DEVICE)
model.eval()

# 3. 이미지 로드 및 1024px 전처리
if not os.path.exists(SAMPLE_IMG_PATH):
    print(f"❌ 이미지를 찾을 수 없습니다: {SAMPLE_IMG_PATH}")
    sys.exit()

img = cv2.imread(SAMPLE_IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (RESOLUTION, RESOLUTION))

# [B, C, H, W] 형태로 변환 및 정규화
input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

print(f"🚀 RTX 5070 가속 가동 | 입력 해상도: {RESOLUTION}x{RESOLUTION}")

# 4. 특징 추출 (Forward Pass)
with torch.no_grad():
    # 모델 구조에 따라 반환 값이 다를 수 있으므로 확인 필요
    features = model(input_tensor) 

# 5. 특징 맵 시각화 및 분석
# Vision Transformer의 출력은 보통 (Batch, Tokens, Embed_dims) 또는 (Batch, C, H, W)
# 여기서는 시각화를 위해 첫 번째 특징 맵을 추출함
if isinstance(features, (list, tuple)):
    feature_map = features[0]
else:
    feature_map = features

print(f"✅ 특징 추출 성공 | 출력 텐서 모양: {feature_map.shape}")

# 특징 맵을 Heatmap 형태로 병합 (채널 평균값)
heatmap = torch.mean(feature_map[0], dim=0).cpu().numpy()

# 시각화 구성
plt.figure(figsize=(20, 10))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.title(f"Original Image ({RESOLUTION}px)")
plt.imshow(img_resized)
plt.axis('off')

# SkySense 특징 맵 (Heatmap)
plt.subplot(1, 2, 2)
plt.title("SkySense Feature Heatmap")
plt.imshow(heatmap, cmap='viridis')
plt.colorbar(fraction=0.046, pad=0.04)
plt.axis('off')

plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f"📸 시각화 결과 저장 완료: {SAVE_PATH}")