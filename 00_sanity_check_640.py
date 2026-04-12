from ultralytics import YOLO
import cv2

# 1. 학습된 최상의 모델 로드
model = YOLO("runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt")

# 2. 테스트할 이미지 경로 (DOTA 데이터셋 중 하나 선택)
source = "datasets/DOTAv1/images/val/P0003.jpg" # 실제 경로에 맞춰 수정하세요

# 3. 추론 실행
results = model.predict(source, save=True, imgsz=640, conf=0.25, device=0)

print(f"✅ 결과가 저장되었습니다: {results[0].save_dir}")