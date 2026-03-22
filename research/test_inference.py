import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import os
import sys

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

from model_fusion import IntegratedYOLOv26

# 💥 가중치 경로 (100 에포크 가중치로 확인)
WEIGHTS_PATH = os.path.join(root_dir, "weights/skysense_320_epoch_100.pth") 
TEST_IMG_PATH = os.path.join(root_dir, "datasets/DOTAv1_Tiled_448/images/val/P0262_0_0.jpg") 

CLASS_NAMES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
               'basketball court', 'ground track field', 'harbor', 'bridge', 
               'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 
               'soccer-ball-field', 'swimming-pool']

def test_single_image():
    device = torch.device("cpu")
    model = IntegratedYOLOv26(nc=15).to(device)
    
    # 족보 껍데기 씌우기
    model.model = nn.Sequential(model.backbone, model.bridge, model.head)
    
    # 💥 [핵심 해결책] 잃어버린 보폭(Stride)을 모델에게 다시 쥐어줍니다!
    stride_tensor = torch.tensor([8, 16, 32]).float().to(device)
    model.stride = stride_tensor
    model.head.stride = stride_tensor
    
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    print(f"🖼️ 이미지 분석 중: {TEST_IMG_PATH}")
    img_cv2 = cv2.imread(TEST_IMG_PATH)
    img_resized = cv2.resize(img_cv2, (320, 320))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img_rgb).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        raw_preds = model(img_tensor)
        
        # 훈련된 one2many 대가리 선택
        if isinstance(raw_preds, dict):
            preds = raw_preds["one2many"]
        else:
            preds = raw_preds
            
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if len(preds.shape) == 3:
            preds = preds[0]
            
        if preds.shape[0] == 19:
            preds = preds.transpose(1, 0)
            
        boxes_cxcywh = preds[:, :4]
        scores, labels = torch.max(preds[:, 4:], dim=1)
        
        # 1차 필터링: 신뢰도 30% 이상만 (너무 얕은 확신은 버림)
        mask = scores > 0.30
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # 좌표 변환 (이제 stride가 곱해져서 나오므로 절대 픽셀 좌표가 됩니다!)
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        
        if len(x1) > 0:
            boxes = torch.stack((x1, y1, x2, y2), dim=1)
            # 2차 필터링: 겹치는 박스 하나로 합치기
            keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        else:
            boxes = []

    # 결과 그리기
    if len(boxes) > 0:
        print(f"✅ 압축 해제 완료! 총 {len(boxes)}개의 박스가 제 위치를 찾았습니다!")
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            text = f"{CLASS_NAMES[int(label.item())]} {score.item():.2f}"
            
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, text, (x1, max(y1 - 5, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        print("🤔 물체가 없습니다.")

    save_path = os.path.join(current_dir, "inference_result.jpg")
    cv2.imwrite(save_path, img_resized)
    print(f"🎉 정상적인 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    test_single_image()