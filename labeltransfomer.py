import os
import glob
from tqdm import tqdm

# 💥 수정해야 할 라벨 폴더 경로 (train, val 둘 다 따로따로 돌리세요!)
LABEL_DIR = "/home/ji/yolov26/datasets/DOTAv1_Tiled_448/labels/val"
SAVE_DIR = "/home/ji/yolov26/datasets/DOTAv1_Tiled_448/labels/val_fixed" # 새로 저장할 폴더

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))

print(f"🩹 {len(label_files)}개의 라벨 파일을 수리합니다...")

for file_path in tqdm(label_files):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        cls = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        
        # 💥 [핵심 수리] 면적으로 잘못 나눠진 y와 h에 448을 다시 곱해서 정상화합니다.
        fixed_y = y * 448.0
        fixed_h = h * 448.0
        
        # 0.0~1.0 범위를 벗어나지 않도록 방어 코드 (Clipping)
        fixed_y = min(max(fixed_y, 0.0), 1.0)
        fixed_h = min(max(fixed_h, 0.0), 1.0)
        
        fixed_lines.append(f"{cls} {x:.6f} {fixed_y:.6f} {w:.6f} {fixed_h:.6f}\n")
    
    # 수리된 파일 저장
    save_path = os.path.join(SAVE_DIR, os.path.basename(file_path))
    with open(save_path, 'w') as f:
        f.writelines(fixed_lines)

print(f"🎉 수리 완료! '{SAVE_DIR}' 폴더를 확인하세요.")