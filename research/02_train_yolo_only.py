from ultralytics import YOLO
import os

# 1. 경로 설정 (절대 경로)
BASE_ROOT = "/home/kim/research/hybrid"
DATA_YAML = os.path.join(BASE_ROOT, "datasets/DOTAv1_Tiled_1024/data.yaml")
MODEL_WEIGHTS = os.path.join(BASE_ROOT, "yolo26s-obb.pt")

def train_yolo_only():
    # 2. 모델 로드 (순수 YOLOv26 OBB)
    # 가중치 파일이 최상위에 있으므로 경로를 정확히 지정하십시오.
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 가중치 파일을 찾을 수 없습니다: {MODEL_WEIGHTS}")
        return

    model = YOLO(MODEL_WEIGHTS)

    # 3. 학습 실행
    # RTX 5070 환경에 맞춰 배치와 해상도 설정
    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=1024,       # 타일 크기에 맞춤
        batch=8,          # OOM 발생 시 2로 줄이십시오
        device=0,         # GPU 0번 사용
        project=os.path.join(BASE_ROOT, "runs"),
        name="YOLOv26_Pure_1024",
        save=True,
        exist_ok=True,
        optimizer='AdamW', # 하이브리드 설정과 동일하게 AdamW 유지
        lr0=1e-4,         # 초기 학습률
        amp=True          # 자동 혼합 정밀도 (VRAM 절약)
    )

if __name__ == "__main__":
    train_yolo_only()