import torch
import sys
import os

# 현재 check_env.py가 있는 폴더 (yolov26/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 경로 추가 (중첩된 SkySense 폴더 구조 대응)
sys.path.append(os.path.join(current_dir, 'ultralytics'))
# 파이썬이 skysense/SkySense 안의 'models' 폴더를 바로 찾을 수 있게 설정
sys.path.append(os.path.join(current_dir, 'skysense', 'SkySense'))

print(f"🔍 현재 경로 설정 완료")

try:
    # --- 1. YOLOv26 로드 테스트 ---
    # Ultralytics 기반 모델 로드 확인
    from ultralytics import YOLO
    yolo_model = YOLO('yolo26s-obb.pt') # 현재 폴더에 있는 파일명
    print("✅ 1. YOLOv26 (Ultralytics) 로드 성공")
    
    # --- 2. SkySense (VisionTransformer) 로드 테스트 ---
    # 파일명: vision_transformer.py / 클래스명: VisionTransformer
    from models.vision_transformer import VisionTransformer
    
    # 모델 객체 생성 (수정된 코드의 기본 인자값 사용)
    # 위성 영상 데이터 처리를 위해 다채널(in_channels=10) 설정이 기본일 수 있습니다.
    skysense_backbone = VisionTransformer(
        img_size=64, 
        patch_size=4, 
        in_channels=3, # 테스트용으로 일반 RGB(3채널) 설정
        embed_dims=128, # 가볍게 테스트하기 위해 크기 축소
        num_layers=2,   # 가볍게 테스트하기 위해 층 축소
        num_heads=4
    )
    
    print("✅ 2. SkySense (VisionTransformer) 로드 및 객체 생성 성공")
    
    # --- 3. 간단한 연산 테스트 (가짜 데이터) ---
    dummy_input = torch.randn(1, 3, 64, 64)
    output = skysense_backbone(dummy_input)
    print(f"✅ 3. 추론 테스트 완료 (출력 텐서 개수: {len(output)})")

except ImportError as e:
    print(f"❌ 임포트 에러: {e}")
    print("💡 팁: 'pip install mmengine mmcv-lite mmsegmentation'이 필요합니다.")
except Exception as e:
    print(f"❌ 에러 발생: {e}")