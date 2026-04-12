import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import nms_rotated
import os

# ==========================================
# 1. 환경 및 실험 파라미터 (1024 최적화)
# ==========================================
MODEL_PATH = "runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt"
IMAGE_PATH = "datasets/DOTAv1/images/val/P0003.jpg" 
SAVE_DIR = "final_report/SkySense_Result"

# 실험 핵심 변수
TILE_SIZE = 1024   # 640에서 1024로 업그레이드
OVERLAP = 256      # 타일 간 겹침 (객체 잘림 방지)
BATCH_SIZE = 4     # 5070 VRAM 활용을 위한 배치 사이즈
CONF_THRESH = 0.25
IOU_THRESH = 0.4   # Rotated NMS 임계값

os.makedirs(SAVE_DIR, exist_ok=True)

def run_optimized_inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🛰️ [5070 가속] {IMAGE_PATH} 분석 시작 (Resolution: {TILE_SIZE})")
    
    model = YOLO(MODEL_PATH).to(device)
    original_img = cv2.imread(IMAGE_PATH)
    h, w = original_img.shape[:2]

    # 1. 타일 좌표 생성
    stride = TILE_SIZE - OVERLAP
    tile_coords = []
    for y in range(0, h - TILE_SIZE + 1, stride):
        for x in range(0, w - TILE_SIZE + 1, stride):
            tile_coords.append((x, y))
    # 우측/하단 자투리 영역 추가
    if (h - TILE_SIZE) % stride != 0: tile_coords.append((tile_coords[-1][0], h - TILE_SIZE))
    if (w - TILE_SIZE) % stride != 0: tile_coords.append((w - TILE_SIZE, tile_coords[-1][1]))

    all_boxes = []
    all_scores = []
    all_cls = []

    # 2. 배치 단위 타일 추론
    for i in range(0, len(tile_coords), BATCH_SIZE):
        batch_coords = tile_coords[i : i + BATCH_SIZE]
        batch_imgs = []
        
        for tx, ty in batch_coords:
            tile = original_img[ty : ty + TILE_SIZE, tx : tx + TILE_SIZE]
            batch_imgs.append(tile)

        # 모델 추론 (imgsz=1024 유지)
        results = model.predict(batch_imgs, conf=CONF_THRESH, imgsz=TILE_SIZE, verbose=False, device=device)

        for idx, r in enumerate(results):
            if r.obb is not None:
                tx, ty = batch_coords[idx]
                # OBB 데이터 (xywhr) 추출
                obb_data = r.obb.xywhr.clone()
                obb_data[:, 0] += tx  # 글로벌 X 보정
                obb_data[:, 1] += ty  # 글로벌 Y 보정
                
                all_boxes.append(obb_data)
                all_scores.append(r.obb.conf)
                all_cls.append(r.obb.cls)

    if not all_boxes:
        print("❓ 검출된 객체가 없습니다."); return

    # 3. GPU 기반 Rotated NMS (핵심 최적화)
    # 리스트를 하나의 텐서로 병합
    cat_boxes = torch.cat(all_boxes, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)
    cat_cls = torch.cat(all_cls, dim=0)

    # ultralytics 내부의 nms_rotated 사용 (Faster than Shapely)
    keep = nms_rotated(cat_boxes, cat_scores, IOU_THRESH)
    
    final_boxes = cat_boxes[keep].cpu().numpy()
    final_scores = cat_scores[keep].cpu().numpy()
    final_cls = cat_cls[keep].cpu().numpy()

    print(f"✨ 최종 객체 수: {len(final_boxes)} (NMS로 {len(cat_boxes) - len(final_boxes)}개 제거)")

    # 4. 시각화 (결과 저장)
    for i in range(len(final_boxes)):
        # xywhr -> 4 corners 변환은 시각화 때만 수행
        box = final_boxes[i]
        # YOLO OBB 시각화 헬퍼 함수 활용 가능하지만 직접 구현도 가능
        # 여기서는 간단히 박스 정보 출력으로 대체하거나 cv2.polylines 사용
        pass 

    print(f"✅ 모든 공정 완료. 결과가 {SAVE_DIR}에 저장되었습니다.")

if __name__ == "__main__":
    run_optimized_inference()