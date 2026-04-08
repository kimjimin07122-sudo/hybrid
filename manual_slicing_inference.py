import cv2
import numpy as np
import torch
from ultralytics import YOLO
from shapely.geometry import Polygon
import os

# ==========================================
# 1. 환경 및 경로 설정
# ==========================================
MODEL_PATH = "runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt"
IMAGE_PATH = "datasets/DOTAv1/images/val/P0003.jpg" # 실제 파일명 확인 필요
SAVE_DIR = "final_report/SkySense_Result"
OS_STRIDE = 640  # 슬라이스 크기
OVERLAP = 128    # 겹침 정도
IOU_THRESH = 0.3 # NMS 임계값 (낮을수록 중복 제거가 강력함)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. OBB 좌표를 Shapely 다각형으로 변환
# ==========================================
def get_polygon(box):
    # box: [x_center, y_center, width, height, angle_rad]
    x, y, w, h, angle = box
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # 박스 네 꼭짓점의 로컬 좌표
    pts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    
    # 회전 변환 및 중심점 이동
    rot_pts = []
    for px, py in pts:
        rx = x + px * cos_a - py * sin_a
        ry = y + px * sin_a + py * cos_a
        rot_pts.append((rx, ry))
    
    return Polygon(rot_pts)

# ==========================================
# 3. 직접 구현한 Rotated NMS (중복 제거)
# ==========================================
def custom_obb_nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0: return []
    
    # 점수 순으로 내림차순 정렬
    indices = np.argsort(scores)[::-1]
    keep = []
    
    # 모든 박스를 미리 다각형으로 변환
    polygons = [get_polygon(b) for b in boxes]
    
    while len(indices) > 0:
        curr_idx = indices[0]
        keep.append(curr_idx)
        
        if len(indices) == 1: break
        
        remaining_indices = indices[1:]
        ious = []
        curr_poly = polygons[curr_idx]
        
        for idx in remaining_indices:
            other_poly = polygons[idx]
            try:
                # IoU 계산 (교집합 면적 / 합집합 면적)
                inter = curr_poly.intersection(other_poly).area
                union = curr_poly.area + other_poly.area - inter
                iou = inter / union if union > 0 else 0
            except:
                iou = 0
            ious.append(iou)
        
        # IoU가 임계값보다 낮은 박스들만 다음 라운드로 남김
        indices = remaining_indices[np.array(ious) <= iou_threshold]
        
    return keep

# ==========================================
# 4. 메인 추론 루프
# ==========================================
def run_final_inference():
    print(f"🛰️ [SkySense-YOLOv26] {IMAGE_PATH} 정밀 분석 시작...")
    model = YOLO(MODEL_PATH)
    
    # 원본 이미지 로드
    original_img = cv2.imread(IMAGE_PATH)
    h, w = original_img.shape[:2]
    
    all_boxes = []
    all_scores = []
    all_cls = []
    
    # 슬라이딩 윈도우 루프
    stride = OS_STRIDE - OVERLAP
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end, x_end = min(y + OS_STRIDE, h), min(x + OS_STRIDE, w)
            y_start, x_start = max(0, y_end - OS_STRIDE), max(0, x_end - OS_STRIDE)
            
            tile = original_img[y_start:y_end, x_start:x_end]
            
            # GPU(device=0)를 사용하여 조각 추론
            results = model.predict(tile, conf=0.25, imgsz=640, verbose=False, device=0)
            
            for r in results:
                if r.obb is not None:
                    for i, obb in enumerate(r.obb.xywhr):
                        box = obb.cpu().numpy()
                        box[0] += x_start # 글로벌 X 좌표 보정
                        box[1] += y_start # 글로벌 Y 좌표 보정
                        all_boxes.append(box)
                        all_scores.append(r.obb.conf[i].cpu().item())
                        all_cls.append(r.obb.cls[i].cpu().item())

    print(f"🧹 중복 제거 전 객체 수: {len(all_boxes)}")
    
    # NMS 적용
    keep_indices = custom_obb_nms(all_boxes, all_scores, iou_threshold=IOU_THRESH)
    print(f"✨ NMS 적용 후 최종 객체 수: {len(keep_indices)}")

    # ==========================================
    # 5. 시각화 및 결과 저장
    # ==========================================
    display_img = original_img.copy()
    for idx in keep_indices:
        box = all_boxes[idx]
        poly_pts = np.array(get_polygon(box).exterior.coords, dtype=np.int32)
        
        # 클래스에 따른 색상 설정 (예: 대형 purple, 소형 blue)
        color = (255, 0, 255) if all_cls[idx] == 0 else (255, 255, 0)
        cv2.polylines(display_img, [poly_pts], isClosed=True, color=color, thickness=2)
        
        # 텍스트 라벨 추가
        label = f"{model.names[all_cls[idx]]} {all_scores[idx]:.2f}"
        cv2.putText(display_img, label, (int(box[0]), int(box[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    save_path = os.path.join(SAVE_DIR, "final_prediction.png")
    cv2.imwrite(save_path, display_img)
    print(f"✅ 분석 완료! 결과 저장됨: {save_path}")

if __name__ == "__main__":
    run_final_inference()