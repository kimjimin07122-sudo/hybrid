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
    
    # 3. SkySense HR 가중치 로드 및 이식
    print(f"📡 SkySense HR 가중치 이식 중: {skysense_weights}")
    checkpoint = torch.load(skysense_weights, map_location='cuda')
    v1_weights = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    
    yolo_dict = model.model.state_dict()
    matched_weights = {k: v for k, v in v1_weights.items() if k in yolo_dict and v.shape == yolo_dict[k].shape}
    yolo_dict.update(matched_weights)
    model.model.load_state_dict(yolo_dict)
    
    print(f"✅ 성공: {len(matched_weights)}개의 레이어에 SkySense 지식을 주입했습니다.")
    
    # 4. 본격적인 300 에포크 학습 시작
    # 텐서보드 로그가 'Jeju_Sat_Project' 폴더에 아주 예쁘게 쌓이도록 설정했습니다.
    model.train(
        data=data_yaml,
        epochs=300,          # 👈 2주의 시간을 활용한 정석 에포크
        imgsz=640,           
        batch=8,             # RTX 3060 12GB 최적화
        device=0,
        optimizer='AdamW',
        lr0=0.001,           
        cos_lr=True,         # 👈 학습률을 코사인 곡선으로 떨어뜨려 그래프를 예쁘게 만듭니다.
        project="Jeju_Sat_Project", # 👈 텐서보드 상위 폴더 이름
        name="SkySense_300e_Final",  # 👈 텐서보드 실험 이름
        save=True,
        plots=True,          # 👈 학습 결과 그래프 자동 생성
        workers=0,            # 👈 윈도우 환경 에러 방지 (필수!)
        )

# 🚀 윈도우 멀티프로세싱 보호막 (이게 없으면 저번 같은 에러가 납니다)
if __name__ == "__main__":
    train_with_skysense()