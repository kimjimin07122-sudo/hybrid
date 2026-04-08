import os
import sahi
from sahi.utils.file import import_model_class # 모델 클래스 직접 가져오기용
from sahi.predict import get_sliced_prediction

# 1. 경로 설정
MODEL_PATH = "runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt"
IMAGE_PATH = "datasets/DOTA/val/images/P0001.png"

def run_sahi_final():
    print(f"✅ SAHI 버전: {sahi.__version__}")
    print("🚀 [SkySense-YOLOv26] 강제 클래스 로드 모드 가동...")

    try:
        # 2. 'AutoDetectionModel'이 고장 났으므로, YOLOv8용 클래스를 직접 지목합니다.
        # YOLOv26은 내부적으로 v8 엔진을 쓰기 때문에 이 클래스가 정답입니다.
        from sahi.models.yolov8 import Yolov8DetectionModel
        
        detection_model = Yolov8DetectionModel(
            model_path=MODEL_PATH,
            confidence_threshold=0.3,
            device="cuda:0" # RTX 3060 사용
        )
        print("✨ 모델 직접 로드 성공!")

    except (ImportError, ModuleNotFoundError):
        # 만약 위 경로도 에러가 나면, 최후의 수단으로 'ultralytics' 범용 클래스 시도
        print("⚠️ 직접 로드 실패, 범용 Ultralytics 클래스 시도...")
        from sahi.models.ultralytics import UltralyticsDetectionModel
        detection_model = UltralyticsDetectionModel(
            model_path=MODEL_PATH,
            model_type="yolov8",
            confidence_threshold=0.3,
            device="cuda:0"
        )

    # 3. 슬라이싱 추론 실행
    print("📸 대면적 위성 영상 슬라이싱 시작...")
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
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    result.export_visuals(export_dir=output_dir, file_name="sahi_final_result")
    print(f"✅ 드디어 성공! 결과 확인: {output_dir}/sahi_final_result.png")

if __name__ == "__main__":
    run_sahi_final()