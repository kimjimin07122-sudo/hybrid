import torch
from ultralytics import YOLO

# 1. 경로 및 장치 설정
MODEL_PATH = "yolo26s-obb.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_yolo_env():
    # 2. 모델 로드 및 GPU 할당
    try:
        model = YOLO(MODEL_PATH).to(DEVICE)
        
        # 3. 장치 상태 출력
        print(f"[STATUS] Model: {MODEL_PATH}")
        print(f"[STATUS] Assigned Device: {model.device}")
        
        # 실제 5070 점유 확인을 위한 간단한 텐서 연산 테스트
        test_tensor = torch.zeros((1, 3, 1024, 1024)).to(DEVICE)
        print(f"[STATUS] 1024px Tensor Allocation: SUCCESS")

    except Exception as e:
        print(f"[ERROR] Environment check failed: {e}")

if __name__ == "__main__":
    check_yolo_env()