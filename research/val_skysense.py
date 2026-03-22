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
from torch.cuda.amp import autocast # 💥 메모리 절약 마법사 추가

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

from model_fusion import IntegratedYOLOv26

WEIGHTS_PATH = os.path.join(root_dir, "weights/skysense_320_epoch_100.pth")
VAL_IMAGES_DIR = os.path.join(root_dir, "datasets/DOTAv1_Tiled_448/images/val")
VAL_LABELS_DIR = os.path.join(root_dir, "datasets/DOTAv1_Tiled_448/labels/val_fixed")

def run_validation():
    # 💥 다시 GPU를 당당하게 사용합니다! (0번 GPU 사용)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 검증 시작! 사용 장치: {device}")
    
    model = IntegratedYOLOv26(nc=15).to(device)
    model.model = nn.Sequential(model.backbone, model.bridge, model.head)
    
    stride_tensor = torch.tensor([8, 16, 32]).float().to(device)
    model.stride = stride_tensor
    model.head.stride = stride_tensor
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 가중치 파일이 없습니다: {WEIGHTS_PATH}")
        return
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox")
    image_paths = glob.glob(os.path.join(VAL_IMAGES_DIR, "*.jpg"))
    print(f"📊 총 {len(image_paths)}장의 이미지에 대한 평가를 시작합니다...")

    for img_path in tqdm(image_paths, desc="Evaluating"):
        filename = os.path.basename(img_path)
        label_path = os.path.join(VAL_LABELS_DIR, filename.replace(".jpg", ".txt"))
        
        target_boxes = []
        target_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (cx - w / 2) * 320.0
                        y1 = (cy - h / 2) * 320.0
                        x2 = (cx + w / 2) * 320.0
                        y2 = (cy + h / 2) * 320.0
                        
                        target_boxes.append([x1, y1, x2, y2])
                        target_labels.append(cls_id)
        
        if len(target_boxes) > 0:
            target = [dict(
                boxes=torch.tensor(target_boxes, dtype=torch.float32).to(device),
                labels=torch.tensor(target_labels, dtype=torch.int64).to(device)
            )]
        else:
            target = [dict(
                boxes=torch.empty((0, 4), dtype=torch.float32).to(device),
                labels=torch.empty((0,), dtype=torch.int64).to(device)
            )]

        img_cv2 = cv2.imread(img_path)
        img_resized = cv2.resize(img_cv2, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img_rgb).to(device).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            # 💥 핵심 해결책: 연산할 때 메모리 사용량을 절반(FP16)으로 강제 압축합니다!
            with autocast():
                preds_raw = model(img_tensor)
            
            if isinstance(preds_raw, dict):
                out = preds_raw.get("one2one", preds_raw.get("one2many"))
            else:
                out = preds_raw
                
            if isinstance(out, (list, tuple)):
                out = out[0]
            if len(out.shape) == 3:
                out = out[0] 
                
            if out.shape[1] == 6:
                boxes = out[:, :4]
                scores = out[:, 4]
                labels = out[:, 5]
                
                mask = scores > 0.001 
                boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
                
                pred_boxes = boxes
                pred_scores = scores
                pred_labels = labels.to(torch.int64)
            else:
                pred_boxes = torch.empty((0, 4), dtype=torch.float32).to(device)
                pred_scores = torch.empty((0,), dtype=torch.float32).to(device)
                pred_labels = torch.empty((0,), dtype=torch.int64).to(device)

        preds = [dict(
            boxes=pred_boxes,
            scores=pred_scores,
            labels=pred_labels
        )]

        metric.update(preds, target)

    print("\n⏳ 계산 중... (조금만 기다려주세요)")
    result = metric.compute()
    
    print("\n" + "="*50)
    print("🏆 [최종 모델 평가 결과 (Baseline)]")
    print("="*50)
    print(f"🔸 mAP@0.50      : {result['map_50'].item():.4f}")
    print(f"🔸 mAP@0.50:0.95 : {result['map'].item():.4f}")
    print("="*50)

if __name__ == "__main__":
    run_validation()