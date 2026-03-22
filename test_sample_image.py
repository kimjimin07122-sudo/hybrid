import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'skysense', 'SkySense'))

from models.vision_transformer import VisionTransformer

# 2. 샘플 이미지 경로 찾기 (yolov26 클론 시 기본 경로 기준)
# 폴더 구조에 따라 'datasets/DOTA/images/train/P0001.jpg' 등으로 수정될 수 있습니다.
sample_img_path = "/home/ji/yolov26/datasets/DOTAv1/images/train/P0000.jpg" # 임시: 만약 DOTA 샘플이 없다면 기본 에셋 사용
# 실제 DOTA 샘플이 있다면 아래 주석을 해제하고 경로를 맞춰주세요.
# sample_img_path = os.path.join(current_dir, 'datasets', 'DOTA', 'images', 'train', 'P0000.jpg')

if not os.path.exists(sample_img_path):
    print(f"❌ 이미지를 찾을 수 없습니다: {sample_img_path}")
    sys.exit()

# 3. 모델 준비 (어제 성공했던 설정 그대로)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(
    img_size=224, # 테스트용 표준 해상도
    patch_size=16, 
    in_channels=3, 
    embed_dims=768, 
    num_layers=12
).to(device)
model.eval()

# 4. 이미지 전처리
img = cv2.imread(sample_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))
input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

print(f"🚀 입력 이미지 로드 완료: {sample_img_path}")
print(f"📊 입력 텐서 모양: {input_tensor.shape}")

# 5. 추론 (Forward Pass)
with torch.no_grad():
    features = model(input_tensor) # 결과는 tuple 형태

# 6. 결과 확인 및 시각화
feature_map = features[0] # 첫 번째 레이어 출력 추출
print(f"✅ SkySense 특징 추출 성공! 출력 모양: {feature_map.shape}")

# 특징 맵 시각화 (첫 6개 채널만)
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(img_resized)
plt.axis('off')

for i in range(3):
    plt.subplot(1, 4, i + 2)
    # 채널별 에너지를 시각화
    f_img = feature_map[0, i, :, :].cpu().numpy()
    plt.imshow(f_img, cmap='viridis')
    plt.title(f"Feature Ch {i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('skysense_feature_test.png')
print("📸 시각화 결과가 'skysense_feature_test.png'로 저장되었습니다.")
plt.show()