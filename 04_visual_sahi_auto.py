import os
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==========================================
# 1. 경로 및 실험 환경 설정
# ==========================================
# 5070 서버 내 실제 경로를 정확히 기입하십시오.
MODEL_PATH = "runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt"
IMAGE_PATH = "datasets/DOTA/val/images/P0001.png"
OUTPUT_DIR = "final_report/SAHI_1024_Results"

# 핵심 파라미터: 640을 버리고 1024로 고정합니다.
RESOLUTION = 1024  
OVERLAP_RATIO = 0.25 # 1024px에서는 객체 연속성을 위해 25% 권장
CONF_THRESH = 0.3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_sahi_inference_1024():
    print(f"📡 SAHI 위치 확인: {sahi.__file__}")
    print(f"🚀 [RTX 5070] 1024px 정밀 슬라이싱 모드 가동...")
    
    # 2. 모델 로드 (에러 방지 구조)
    try:
        # YOLOv26은 내부적으로 v8 엔진을 공유하므로 yolov8 타입을 사용합니다.
        detection_model = AutoDetectionModel.from_model_type(
            model_type='yolov8',
            model_path=MODEL_PATH,
            confidence_threshold=CONF_THRESH,
            device="cuda:0" # 5070 GPU 강제 지정
        )
        print("✨ 모델 로드 성공 (AutoDetectionModel)")
    except Exception as e:
        print(f"⚠️ 자동 로드 실패({e}), 수동 클래스 로드 시도...")
        from sahi.models.yolov8 import Yolov8DetectionModel
        detection_model = Yolov8DetectionModel(
            model_path=MODEL_PATH,
            confidence_threshold=CONF_THRESH,
            device="cuda:0"
        )

    # 3. 1024px 슬라이싱 추론 실행
    # 640으로 조각낼 때보다 타일 수가 줄어들어 5070에서 더 빠른 추론이 가능합니다.
    print(f"📸 {RESOLUTION}px 단위 대면적 영상 분석 시작...")
    result = get_sliced_prediction(
        IMAGE_PATH,
        detection_model,
        slice_height=RESOLUTION,
        slice_width=RESOLUTION,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        perform_standard_prediction=False # 메모리 보호를 위해 원본 전체 추론은 끕니다.
    )

    # 4. 결과 시각화 및 저장
    save_name = os.path.basename(IMAGE_PATH).split('.')[0] + "_sahi_1024"
    result.export_visuals(export_dir=OUTPUT_DIR, file_name=save_name)
    
    print(f"✅ 분석 완료! 결과 확인: {OUTPUT_DIR}/{save_name}.png")
    print(f"📊 검출 객체 수: {len(result.object_prediction_list)}")

if __name__ == "__main__":
    run_sahi_inference_1024()