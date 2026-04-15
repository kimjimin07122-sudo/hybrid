import os
import torch
from ultralytics.models.yolo.obb import OBBTrainer
from model_fusion import SkySense_Backbone_with_Lateral

# 1. Ultralytics의 OBB 트레이너를 상속받아 나만의 트레이너를 만듭니다.
class HybridTrainer(OBBTrainer):
    # 트레이너가 학습 직전 모델을 생성하는 핵심 엔진을 가로챕니다.
    def get_model(self, cfg=None, weights=None, verbose=True):
        # 부모 클래스의 기능을 호출하여 정상적인 YOLO 모델을 일단 만듭니다.
        model = super().get_model(cfg, weights, verbose)
        
        # 여기서 모델이 반환되기 직전에 낚아채서 백본을 교체합니다.
        print("\n🛠️ [심층 수술] 모델 빌드 과정에 침투하여 SkySense를 이식합니다...")
        weights_path = r"C:/hybrid/weights/skysense_model_backbone_hr.pth"
        hybrid_backbone = SkySense_Backbone_with_Lateral(weights_path=weights_path)
        
        # OBBModel 내부에 강제 이식
        model.backbone = hybrid_backbone
        print("🔥 이식 완료! Optimizer가 이 거대한 파라미터를 정상적으로 인식할 것입니다.\n")
        
        return model

def train_surgery():
    print("🚀 Custom Trainer를 통한 완전한 하이브리드 학습 시작")

    # 2. 파라미터는 원래대로 '문자열' 경로로 돌려놓습니다.
    train_args = dict(
        model=r"C:/hybrid/yolo26s-obb.pt",  # 경로로 원복 (TypeError 해결)
        data=r"C:/hybrid/datasets/DOTAv1_Tiled_1024/data.yaml",
        epochs=300,
        imgsz=640,
        batch=1,
        amp=True,
        project='SkySense_YOLO26',
        name='Hybrid_Final_Evolution',
        device=0,
        workers=2,
        optimizer='AdamW',
        lr0=0.0001,
        close_mosaic=10,
        deterministic=True
    )

    # 3. 기본 트레이너가 아닌, 우리가 만든 HybridTrainer를 사용합니다.
    trainer = HybridTrainer(overrides=train_args)
    trainer.train()

if __name__ == "__main__":
    train_surgery()