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
sys.path.insert(0, os.path.join(root_dir, "ultralytics"))
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

# 00번 빌드 파일에서 정의한 클래스 임포트
from model_fusion import IntegratedYOLOv26

# 실험 파라미터 (320px 가중치 전용)
WEIGHTS_PATH = os.path.join(root_dir, "weights/skysense_320_epoch_100.pth") 
TEST_IMG_PATH = os.path.join(root_dir, "datasets/DOTAv1_Tiled_448/images/val/P0262_0_0.jpg") 
IMG_SIZE = 320 # 가중치 규격에 맞춤

CLASS_NAMES = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 
               'basketball court', 'ground track field', 'harbor', 'bridge', 
               'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 
               'soccer-ball-field', 'swimming-pool']

def test_single_image():
    # 5070 가속 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 장치 할당: {device} | 가중치: {os.path.basename(WEIGHTS_PATH)}")

    # 2. 모델 생성 및 구조 복원
    model = IntegratedYOLOv26(nc=15).to(device)
    
    # 가중치 로드 시 Stride 정보 손실 방지 (임시 주입)
    stride_tensor = torch.tensor([8, 16, 32]).float().to(device)
    model.stride = stride_tensor
    if hasattr(model.head, 'stride'):
        model.head.stride = stride_tensor
    
    # 가중치 로드
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("✅ 가중치 로드 성공")
    except Exception as e:
        print(f"❌ 가중치 로드 실패: {e}")
        return

    model.eval()

    # 3. 전처리
    img_cv2 = cv2.imread(TEST_IMG_PATH)
    if img_cv2 is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {TEST_IMG_PATH}"); return

    h_orig, w_orig = img_cv2.shape[:2]
    img_resized = cv2.resize(img_cv2, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img_rgb).to(device).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # 4. 추론
    with torch.no_grad():
        raw_preds = model(img_tensor)
        
        # YOLOv26/v10 Head 특유의 출력 구조 처리
        if isinstance(raw_preds, dict):
            preds = raw_preds["one2many"]
        else:
            preds = raw_preds
            
        # 텐서 차원 정렬 [Batch, Anchors, Box+Cls]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if len(preds.shape) == 3:
            preds = preds[0]
            
        # [19, Anchors] -> [Anchors, 19] 변환이 필요한 경우
        if preds.shape[0] < preds.shape[1]:
            preds = preds.transpose(1, 0)
            
        # 좌표(4) + 클래스(15) 분리
        boxes_cxcywh = preds[:, :4]
        scores, labels = torch.max(preds[:, 4:], dim=1)
        
        # 필터링 (신뢰도 0.3)
        mask = scores > 0.30
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # CXCYWH -> X1Y1X2Y2 변환 (Stride 보정 포함)
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        
        if len(x1) > 0:
            boxes = torch.stack((x1, y1, x2, y2), dim=1)
            # NMS 수행
            keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        else:
            boxes = []

    # 5. 결과 시각화
    if len(boxes) > 0:
        print(f"✨ 검출 성공: {len(boxes)}개 객체")
        for box, score, label in zip(boxes, scores, labels):
            # 320px 좌표를 원본 이미지 크기 혹은 320px 캔버스에 투영
            x1, y1, x2, y2 = map(int, box.tolist())
            cls_name = CLASS_NAMES[int(label.item())]
            text = f"{cls_name} {score.item():.2f}"
            
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, text, (x1, max(y1 - 5, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        print("🤔 검출된 물체가 없습니다. Confidence 임계값을 낮추거나 가중치를 확인하십시오.")

    save_path = os.path.join(current_dir, "inference_debug.jpg")
    cv2.imwrite(save_path, img_resized)
    print(f"📸 디버깅 결과 저장 완료: {save_path}")

if __name__ == "__main__":
    test_single_image()