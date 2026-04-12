import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.amp import autocast

# 1. 경로 및 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "ultralytics"))
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

# 00번 빌드 파일에서 정의한 OBB 클래스 임포트
from model_fusion import IntegratedYOLOv26_OBB 

# 5070 연구 사양으로 격상
WEIGHTS_PATH = os.path.join(root_dir, "weights/skysense_1024_OBB_epoch_300.pth")
VAL_IMAGES_DIR = os.path.join(root_dir, "datasets/DOTAv1_Tiled_1024/images/val")
VAL_LABELS_DIR = os.path.join(root_dir, "datasets/DOTAv1_Tiled_1024/labels/val")
RESOLUTION = 1024 

CLASS_NAMES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
               'basketball court', 'ground track field', 'harbor', 'bridge', 
               'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 
               'soccer-ball-field', 'swimming-pool']

def run_validation_1024():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [RTX 5070] 1024px 정밀 검증 엔진 가동")
    
    # 2. 모델 로드 (OBB 전용)
    model = IntegratedYOLOv26_OBB(nc=15).to(device)
    
    # 가중치 로드 및 Stride 보정
    stride_tensor = torch.tensor([8, 16, 32]).float().to(device)
    model.stride = stride_tensor
    if hasattr(model.head, 'stride'):
        model.head.stride = stride_tensor
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 가중치 파일 부재: {WEIGHTS_PATH}"); return
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # 현재 torchmetrics는 표준 HBB mAP를 계산합니다. 
    # OBB 정밀 측정을 위해서는 나중에 전용 OBB Metric 라이브러리 도입을 고려하십시오.
    metric = MeanAveragePrecision(iou_type="bbox")
    image_paths = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    print(f"📊 총 {len(image_paths)}장 이미지 성능 분석 중...")

    for img_path in tqdm(image_paths, desc="mAP Calculating"):
        filename = os.path.basename(img_path)
        label_path = os.path.join(VAL_LABELS_DIR, filename.replace(".jpg", ".txt"))
        
        target_boxes = []
        target_labels = []
        
        # 3. 라벨 파싱 (1024 정규화 대응)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # YOLO format: cls, cx, cy, w, h
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w / 2) * RESOLUTION
                        y1 = (cy - h / 2) * RESOLUTION
                        x2 = (cx + w / 2) * RESOLUTION
                        y2 = (cy + h / 2) * RESOLUTION
                        
                        target_boxes.append([x1, y1, x2, y2])
                        target_labels.append(cls_id)
        
        target = [dict(
            boxes=torch.tensor(target_boxes, dtype=torch.float32).to(device) if target_boxes else torch.empty((0, 4)).to(device),
            labels=torch.tensor(target_labels, dtype=torch.int64).to(device) if target_labels else torch.empty((0,)).to(device)
        )]

        # 4. 이미지 전처리 및 추론
        img_cv2 = cv2.imread(img_path)
        img_resized = cv2.resize(img_cv2, (RESOLUTION, RESOLUTION))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_rgb).to(device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            with autocast('cuda'): # 5070의 FP16 가속 활용
                preds_raw = model(img_tensor)
            
            # Head 출력 처리 (one2one/one2many 선택)
            out = preds_raw["one2one"] if isinstance(preds_raw, dict) and "one2one" in preds_raw else preds_raw
            if isinstance(out, (list, tuple)): out = out[0]
            if len(out.shape) == 3: out = out[0] 
            
            # HBB 기반 mAP 계산을 위한 [x1, y1, x2, y2, score, cls] 추출
            # OBB 모델이라도 mAP@bbox 계산을 위해 대표 사각형을 추출합니다.
            if out.shape[1] >= 6:
                boxes = out[:, :4]
                scores = out[:, 4]
                labels = out[:, 5]
                
                mask = scores > 0.001 # 검출 한계치
                pred_boxes, pred_scores, pred_labels = boxes[mask], scores[mask], labels[mask].to(torch.int64)
            else:
                pred_boxes = torch.empty((0, 4)).to(device)
                pred_scores = torch.empty((0,)).to(device)
                pred_labels = torch.empty((0,)).to(device)

        preds = [dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels)]
        metric.update(preds, target)

    # 5. 최종 리포트
    print("\n⏳ 정밀 지표 계산 중...")
    result = metric.compute()
    
    print("\n" + "="*50)
    print(f"🏆 [1024px 하이브리드 모델 평가 리포트]")
    print("-"*50)
    print(f"✅ mAP@.50 (HBB Proxy)  : {result['map_50'].item():.4f}")
    print(f"✅ mAP@.50:.95 (HBB)    : {result['map'].item():.4f}")
    print("="*50)
    print("💡 주의: 위 지표는 HBB 기준입니다. OBB 정밀 지표는 별도 측정이 필요합니다.")

if __name__ == "__main__":
    run_validation_1024()