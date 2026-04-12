import torch
from ultralytics import YOLO
import os

# ==========================================
# 1. 경로 및 실험 환경 설정
# ==========================================
MODEL_WEIGHTS = "yolo26s-obb.pt"
SKYSENSE_PTH = "weights/skysense_model_backbone_hr.pth"
DATA_YAML = "DOTAv1.yaml"

# RTX 5070 환경 변수
RESOLUTION = 1024  
BATCH_SIZE = 16    # 5070 VRAM(12GB) 고려, OOM 발생 시 8~12로 조절
EPOCHS = 300       # 논문급 성능을 위해 300 에폭 권장

def train_hybrid_1024():
    print(f"🚀 [RTX 5070] SkySense-YOLOv26 하이브리드 학습 시작 ({RESOLUTION}px)")
    
    # 2. 모델 로드
    model = YOLO(MODEL_WEIGHTS)
    
    # 3. SkySense 가중치 주입 (개선된 로직)
    if os.path.exists(SKYSENSE_PTH):
        print(f"📡 SkySense 가중치 로드 중: {SKYSENSE_PTH}")
        ckpt = torch.load(SKYSENSE_PTH, map_location='cuda')
        v1_weights = ckpt.get('model', ckpt.get('state_dict', ckpt))
        
        yolo_dict = model.model.state_dict()
        matched_weights = {k: v for k, v in v1_weights.items() 
                          if k in yolo_dict and v.shape == yolo_dict[k].shape}
        
        if len(matched_weights) == 0:
            print("⚠️ 경고: 매칭된 가중치가 0개입니다. 레이어 이름을 확인하십시오.")
        else:
            yolo_dict.update(matched_weights)
            model.model.load_state_dict(yolo_dict)
            print(f"✅ 성공: {len(matched_weights)}개의 레이어 지식 이식 완료.")
    else:
        print("ℹ️ SkySense 가중치 파일이 없어 기본 가중치로 학습을 진행합니다.")

    # 4. 본격적인 학습 (5070 최적화 설정)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=RESOLUTION,      # 1024 해상도 적용
        batch=BATCH_SIZE,      # 5070 파워 활용
        device=0,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,           # 학습 안정성을 위한 코사인 스케줄러 추가
        project="SkySense_Project",
        name="1024_Hybrid_Experiment",
        save=True,
        cache=True,            # RAM 여유가 있다면 속도 향상을 위해 활성화
        exist_ok=True          # 기존 실험 폴더 덮어쓰기 허용
    )

if __name__ == "__main__":
    train_hybrid_1024()