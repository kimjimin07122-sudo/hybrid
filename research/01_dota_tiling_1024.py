import cv2
import os
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 경로 및 파라미터 설정
# ==========================================
SOURCE_PATH = "/home/kim/research/hybrid/datasets/DOTAv1"
OUTPUT_PATH = "/home/kim/research/hybrid/datasets/DOTAv1_Tiled_1024"

TILE_SIZE = 1024      # 타일 크기
OVERLAP = 256         # 타일 간 겹침 영역
STRIDE = TILE_SIZE - OVERLAP
VISIBILITY_THR = 0.4  # 객체 포함 최소 면적 비율

def get_dota_obb_labels(label_path, w_orig, h_orig):
    """
    사용자의 정규화된 YOLO-OBB 라벨을 읽어 픽셀 단위로 복원합니다.
    Format: cls x1 y1 x2 y2 x3 y3 x4 y4 (all normalized 0~1)
    """
    if not os.path.exists(label_path): return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            try:
                cls_id = int(parts[0])
                # 정규화된 좌표 추출 및 픽셀 복원
                coords = np.array(list(map(float, parts[1:9])))
                coords[0::2] *= w_orig  # x coordinates
                coords[1::2] *= h_orig  # y coordinates
                
                labels.append({'cls': cls_id, 'pts': coords.reshape(-1, 2)})
            except ValueError:
                continue
    return labels

def process_split(split):
    img_src = os.path.join(SOURCE_PATH, "images", split)
    lbl_src = os.path.join(SOURCE_PATH, "labels", split)
    img_dst = os.path.join(OUTPUT_PATH, "images", split)
    lbl_dst = os.path.join(OUTPUT_PATH, "labels", split)

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    fnames = [f for f in os.listdir(img_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not fnames:
        print(f"⚠️ {split} 폴더에서 이미지를 찾을 수 없습니다."); return

    tiled_count = 0
    for fname in tqdm(fnames, desc=f"🎬 {split} 타일링 중"):
        img = cv2.imread(os.path.join(img_src, fname))
        if img is None: continue
        h_orig, w_orig = img.shape[:2]
        
        # 라벨 로드 (픽셀 복원 포함)
        labels = get_dota_obb_labels(os.path.join(lbl_src, os.path.splitext(fname)[0] + ".txt"), w_orig, h_orig)

        for ty in range(0, max(1, h_orig - OVERLAP), STRIDE):
            for tx in range(0, max(1, w_orig - OVERLAP), STRIDE):
                th = min(TILE_SIZE, h_orig - ty)
                tw = min(TILE_SIZE, w_orig - tx)
                
                new_labels = []
                for label in labels:
                    pts = label['pts'].copy()
                    
                    # 1. 타일 기준 좌표 이동
                    pts[:, 0] -= tx
                    pts[:, 1] -= ty
                    
                    # 2. 클리핑 및 면적 계산 (Shoelace Formula)
                    clipped_pts = np.clip(pts, [0, 0], [tw, th])
                    
                    orig_area = 0.5 * np.abs(np.dot(label['pts'][:,0], np.roll(label['pts'][:,1], 1)) - 
                                           np.dot(label['pts'][:,1], np.roll(label['pts'][:,0], 1)))
                    clipped_area = 0.5 * np.abs(np.dot(clipped_pts[:,0], np.roll(clipped_pts[:,1], 1)) - 
                                               np.dot(clipped_pts[:,1], np.roll(clipped_pts[:,0], 1)))
                    
                    if orig_area > 0 and (clipped_area / orig_area) > VISIBILITY_THR:
                        # 3. 다시 타일 크기(1024) 기준으로 정규화하여 저장
                        norm_pts = pts.flatten() / TILE_SIZE
                        line = f"{label['cls']} " + " ".join([f"{p:.6f}" for p in norm_pts])
                        new_labels.append(line)

                if new_labels:
                    tile_img = img[ty:ty+th, tx:tx+tw]
                    
                    # 패딩 처리
                    if th < TILE_SIZE or tw < TILE_SIZE:
                        tile_img = cv2.copyMakeBorder(tile_img, 0, TILE_SIZE-th, 0, TILE_SIZE-tw, 
                                                     cv2.BORDER_CONSTANT, value=(0,0,0))
                    
                    save_name = f"{os.path.splitext(fname)[0]}_{tx}_{ty}.jpg"
                    cv2.imwrite(os.path.join(img_dst, save_name), tile_img)
                    with open(os.path.join(lbl_dst, os.path.splitext(save_name)[0] + ".txt"), 'w') as f:
                        f.write("\n".join(new_labels))
                    tiled_count += 1

    print(f"✅ {split} 완료: {tiled_count}개 타일 생성.")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    print(f"🎉 모든 전처리 완료! 결과: {OUTPUT_PATH}")