import torch
from ultralytics import YOLO # YOLOv26 라이브러리 가정
sys.path.append("/home/kim/research/hybrid")
from model_fusion import IntegratedYOLOv26_OBB

def train_hybrid():
    # 1. 하이브리드 모델 인스턴스 생성
    model = IntegratedYOLOv26_OBB(nc=15)
    
    # 2. VRAM 부족 시 백본 동결 (선택 사항)
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    # 3. 학습 설정
    # YOLOv26이 커스텀 모델 객체를 지원하는 방식에 따라 아래 코드를 적용하십시오.
    # 일반적인 YOLOv8/9/10/26 인터페이스 예시:
    yolo_model = YOLO('yolov26s-obb.yaml') # 구조 정의
    
    # 4. 학습 시작 (VRAM 절약을 위해 batch=1, imgsz=1024 설정)
    yolo_model.train(
        data='DOTA_v1.0.yaml',
        epochs=300,
        imgsz=1024,
        batch=1,       # 12GB VRAM 고려
        amp=True,      # FP16 학습 필수
        project='runs/SkySense_YOLO26',
        name='hybrid_1024_v1'
    )

if __name__ == "__main__":
    train_hybrid()