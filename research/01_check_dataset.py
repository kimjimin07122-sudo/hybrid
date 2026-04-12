import cv2
import os
import numpy as np
import random

# 검증할 데이터셋 경로
BASE_PATH = "/home/kim/research/hybrid/datasets/DOTAv1_Tiled_1024"
IMG_DIR = os.path.join(BASE_PATH, "images/train")
LBL_DIR = os.path.join(BASE_PATH, "labels/train")
SAVE_PATH = "./dataset_check_result.jpg"

def verify_dataset():
    # 1. 무작위 파일 선택
    fnames = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    if not fnames:
        print("❌ 이미지를 찾을 수 없습니다."); return
    
    target_img = random.choice(fnames)
    target_lbl = os.path.splitext(target_img)[0] + ".txt"
    
    img_path = os.path.join(IMG_DIR, target_img)
    lbl_path = os.path.join(LBL_DIR, target_lbl)
    
    # 2. 이미지 로드
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 3. 라벨 로드 및 그리기
    if not os.path.exists(lbl_path):
        print(f"⚠️ 라벨 파일 부재: {target_lbl}"); return
        
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            
            # 1024px 규격에 맞춰 좌표 복원 (0~1 -> 0~1024)
            pts = np.array(list(map(float, parts[1:9]))).reshape(-1, 2)
            pts[:, 0] *= 1024
            pts[:, 1] *= 1024
            
            # 박스 그리기 (Cyan 색상)
            pts = pts.astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.putText(img, parts[0], (pts[0][0], pts[0][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # 4. 결과 저장
    cv2.imwrite(SAVE_PATH, img)
    print(f"✅ 검증 완료: '{target_img}' 결과가 '{SAVE_PATH}'에 저장되었습니다.")
    print(f"💡 이제 해당 이미지를 열어 박스가 객체 위에 정확히 있는지 확인하십시오.")

if __name__ == "__main__":
    verify_dataset()