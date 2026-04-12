import sys
import os
import torch
from ultralytics import YOLO
from model_fusion import SkySense_Backbone_with_Lateral

def train_surgery():
    print("🚀 [수술실] YOLOv26 + SkySense HR 하이브리드 조립 시작...")

    # 1. 기성품 YOLOv26 모델 로드 (구조 파악용)
    # yolo26s-obb.pt 파일이 현재 폴더에 반드시 있어야 합니다.
    model = YOLO('/home/kim/research/hybrid/yolo26s-obb.pt') 

    # 2. 커스텀 하이브리드 백본 객체 생성
    hybrid_backbone = SkySense_Backbone_with_Lateral()

    # 3. [핵심] 백본 강제 교체 (Surgery)
    # Ultralytics 내부의 순정 백본을 뽑아내고 거인족 백본으로 치환합니다.
    model.model.backbone = hybrid_backbone
    
    print("🚀 [수술실] 순정 백본 제거 및 1x1 Conv + SkySense HR 이식 완료.")

    # 4. 학습 실행
    # RTX 5070 12GB 환경을 고려한 극한의 생존 세팅입니다.
    model.train(
        data='/home/kim/research/hybrid/datasets/DOTAv1_Tiled_1024/data.yaml',
        epochs=300,
        imgsz=640,       # VRAM 생존을 위한 640 고정
        batch=1,         # Batch 1 고정 (절대 올리지 마십시오)
        amp=True,        # FP16 혼합 정밀도 (메모리 절약 및 속도 향상)
        project='SkySense_YOLO26',
        name='Hybrid_HR_640',
        device=0,
        workers=4,
        optimizer='AdamW',
        lr0=0.0001       # 사전 학습된 무거운 백본이 망가지지 않도록 낮은 학습률 사용
    )

if __name__ == "__main__":
    try:
        train_surgery()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ [치명적 에러] VRAM 부족(OOM) 발생!")
            print("해결책: imgsz를 512로 더 낮추거나, Batch 사이즈가 1인지 확인하십시오.")
        else:
            print(f"\n❌ [실행 에러] {e}")