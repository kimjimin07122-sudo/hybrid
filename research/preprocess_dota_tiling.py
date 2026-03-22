import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

# --- [강력 권고] 모든 경로를 절대 경로로 고정합니다 ---
SOURCE_PATH = "/home/ji/yolov26/datasets/DOTAv1"
OUTPUT_PATH = "/home/ji/yolov26/datasets/DOTAv1_Tiled_448"

TILE_SIZE = 448
OVERLAP = 100
STRIDE = TILE_SIZE - OVERLAP
VISIBILITY_THR = 0.3

CLASS_MAP = {
    'plane': 0, 'ship': 1, 'storage tank': 2, 'baseball diamond': 3,
    'tennis court': 4, 'basketball court': 5, 'ground track field': 6,
    'harbor': 7, 'bridge': 8, 'large-vehicle': 9, 'small-vehicle': 10,
    'helicopter': 11, 'roundabout': 12, 'soccer-ball-field': 13, 'swimming-pool': 14
}

def get_yolo_labels(label_path, img_w, img_h):
    if not os.path.exists(label_path): return np.array([])
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            try:
                coords = list(map(float, parts[:8]))
                x_c, y_c = coords[0::2], coords[1::2]
                l, t, r, b = min(x_c), min(y_c), max(x_c), max(y_c)
                cat = parts[8]
                cls = CLASS_MAP.get(cat, 0)
                labels.append([cls, l, t, r, b])
            except: continue
    return np.array(labels)

def process_split(split):
    img_src = os.path.join(SOURCE_PATH, "images", split)
    lbl_src = os.path.join(SOURCE_PATH, "labels", split)
    img_dst = os.path.join(OUTPUT_PATH, "images", split)
    lbl_dst = os.path.join(OUTPUT_PATH, "labels", split)

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    fnames = [f for f in os.listdir(img_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    tiled_count = 0

    for fname in tqdm(fnames, desc=f"🎬 {split} 조각내는 중"):
        img = cv2.imread(os.path.join(img_src, fname))
        if img is None: continue
        h_orig, w_orig = img.shape[:2]
        labels = get_yolo_labels(os.path.join(lbl_src, os.path.splitext(fname)[0] + ".txt"), w_orig, h_orig)

        for ty in range(0, max(1, h_orig - OVERLAP), STRIDE):
            for tx in range(0, max(1, w_orig - OVERLAP), STRIDE):
                # 타일 크기 결정
                th = min(TILE_SIZE, h_orig - ty)
                tw = min(TILE_SIZE, w_orig - tx)
                
                new_labels = []
                for cls, l, t, r, b in labels:
                    # 타일과 겹치는 영역 계산
                    il, it = max(l, tx), max(t, ty)
                    ir, ib = min(r, tx + tw), min(b, ty + th)
                    iw, ih = max(0, ir - il), max(0, ib - it)
                    
                    if (iw * ih) / ((r - l) * (b - t) + 1e-6) > VISIBILITY_THR:
                        # 타일 기준 상대 좌표 및 정규화
                        nxt, nyt = (il + ir - 2 * tx) / (2 * TILE_SIZE), (it + ib - 2 * ty) / (2 * TILE_SIZE)
                        nw, nh = iw / TILE_SIZE, ih / TILE_SIZE
                        new_labels.append(f"{int(cls)} {nxt:.6f} {nyt:.6f} {nw:.6f} {nh:.6f}")

                if new_labels:
                    tile_img = img[ty:ty+th, tx:tx+tw]
                    # 448x448로 패딩 (사이즈 불일치 방지)
                    if th < TILE_SIZE or tw < TILE_SIZE:
                        tile_img = cv2.copyMakeBorder(tile_img, 0, TILE_SIZE-th, 0, TILE_SIZE-tw, cv2.BORDER_CONSTANT, value=(0,0,0))
                    
                    save_name = f"{os.path.splitext(fname)[0]}_{tx}_{ty}.jpg"
                    save_path = os.path.join(img_dst, save_name)
                    
                    if cv2.imwrite(save_path, tile_img):
                        with open(os.path.join(lbl_dst, os.path.splitext(save_name)[0] + ".txt"), 'w') as f:
                            f.write("\n".join(new_labels))
                        tiled_count += 1
                    else:
                        print(f"⚠️ 저장 실패: {save_path}")

    print(f"✅ {split} 완료: {tiled_count}개 타일 생성됨.")

if __name__ == "__main__":
    process_split("train")
    process_split("val")
    print(f"🎉 모든 작업 완료! 경로 확인: ls {OUTPUT_PATH}/images/train")