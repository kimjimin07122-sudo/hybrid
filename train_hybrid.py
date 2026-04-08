import torch
from ultralytics import YOLO
import os

# 1. 경로 설정
skysense_weights = "weights/skysense_model_backbone_hr.pth"
yolo_base_model = "yolo26s-obb.pt"
data_yaml = "DOTAv1.yaml"

def train_with_skysense():
    # 2. YOLOv26 모델 로드
    print("🚀 YOLOv26-OBB 모델 로딩 중...")
    model = YOLO(yolo_base_model)
    
    # 3. SkySense HR 가중치 로드
    print(f"📡 SkySense HR 가중치 이식 중: {skysense_weights}")
    checkpoint = torch.load(skysense_weights, map_location='cuda')
    
    # 가중치 딕셔너리 추출 (파일명에 따라 'model' 혹은 'state_dict' 키에 들어있을 수 있음)
    v1_weights = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    # 4. 가중치 매칭 및 주입 (이름과 모양이 같은 것만 골라냄)
    yolo_dict = model.model.state_dict()
    matched_weights = {k: v for k, v in v1_weights.items() if k in yolo_dict and v.shape == yolo_dict[k].shape}
    
    yolo_dict.update(matched_weights)
    model.model.load_state_dict(yolo_dict)
    
    print(f"✅ 성공: {len(matched_weights)}개의 레이어에 SkySense 지식을 주입했습니다.")
    
    # 5. 본격적인 학습 시작 (3060 환경 최적화 설정)
    model.train(
        data=data_yaml,
        epochs=200,          # 학부 연구 수준에서 적당한 횟수
        imgsz=640,          # DOTA 표준 해상도
        batch=8,            # 3060(12GB)에서 가장 안정적인 배치 사이즈
        device=0,           # GPU 사용
        optimizer='AdamW',  # 위성 영상 탐지에 효과적인 옵티마이저
        lr0=0.001,          # 가중치 이식 시에는 조금 낮게 시작
        project="SkySense_Project",
        name="HR_Backbone_Experiment",
        save=True
    )

if __name__ == "__main__":
    train_with_skysense()