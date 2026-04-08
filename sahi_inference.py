import os
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. 경로 설정 (사용자님 환경 정조준)
MODEL_PATH = "runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt"
IMAGE_PATH = "datasets/DOTA/val/images/P0001.png"

def run_sahi_inference():
    print(f"📡 SAHI 위치 확인: {sahi.__file__}")
    print("🚀 [SkySense-YOLOv26] SAHI 정밀 추론 엔진 가동...")
    
    # 2. 모델 로드
    # pip install "sahi[ultralytics]" 이후에는 이 방식이 가장 확실합니다.
    try:
        detection_model = AutoDetectionModel.from_model_type(
            model_type='yolov8', # YOLOv26은 v8 엔진을 공유합니다
            model_path=MODEL_PATH,
            confidence_threshold=0.3,
            device="cuda:0"
        )
    except AttributeError:
        # 만약 또 AttributeError가 나면, 클래스를 직접 호출하는 최후의 수단 사용
        print("⚠️ AutoDetectionModel 에러 발생, 수동 로드 시도...")
        from sahi.models.yolov8 import Yolov8DetectionModel
        detection_model = Yolov8DetectionModel(
            model_path=MODEL_PATH,
            confidence_threshold=0.3,
            device="cuda:0"
        )

    print("📸 거대 위성 영상 슬라이싱(Slicing) 추론 시작...")
    # 3. 조각 추론 실행
    result = get_sliced_prediction(
        IMAGE_PATH,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 4. 결과 저장
    output_dir = "sahi_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    result.export_visuals(export_dir=output_dir, file_name="sahi_final_success")
    print(f"✅ 드디어 성공! 결과 확인: {output_dir}/sahi_final_success.png")

if __name__ == "__main__":
    run_sahi_inference()