import os
import sahi
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# ==========================================
# 1. 환경 및 경로 설정 (1024 연구 전용)
# ==========================================
# 5070 서버의 실제 가중치 경로와 이미지 경로로 수정하십시오.
MODEL_PATH = "runs/obb/SkySense_Project/HR_Backbone_Experiment/weights/best.pt"
IMAGE_PATH = "datasets/DOTA/val/images/P0001.png"
OUTPUT_DIR = "final_report/SAHI_Visuals"

# 실험 핵심 파라미터
RESOLUTION = 1024  # 640에서 1024로 상향
OVERLAP_RATIO = 0.25 # 1024 해상도에서 객체 잘림 방지를 위해 25% 설정
CONF_THRESH = 0.3

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_sahi_optimized():
    print(f"✅ SAHI 버전: {sahi.__version__}")
    print(f"🚀 [RTX 5070] 1024px 정밀 슬라이싱 모드 가동...")

    try:
        # 2. 모델 로드 (YOLOv26/v8 OBB 호환을 위해 UltralyticsDetectionModel 사용)
        detection_model = UltralyticsDetectionModel(
            model_path=MODEL_PATH,
            model_type="yolov8",      # YOLOv26은 v8 엔진 기반이므로 유지
            confidence_threshold=CONF_THRESH,
            device="cuda:0"           # 5070 GPU 강제 지정
        )
        print("✨ 5070 GPU에 모델 로드 성공!")

        # 3. 슬라이싱 추론 실행
        # 1024 해상도로 조각을 내어 5070의 VRAM을 효율적으로 사용합니다.
        print(f"📸 {RESOLUTION}px 단위로 대면적 영상 분석 중...")
        result = get_sliced_prediction(
            IMAGE_PATH,
            detection_model,
            slice_height=RESOLUTION,
            slice_width=RESOLUTION,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            perform_standard_prediction=False, # 전체 이미지 추론은 VRAM 낭비이므로 비활성
            verbose=1
        )

        # 4. 결과 시각화 및 저장
        file_name = os.path.basename(IMAGE_PATH).split('.')[0]
        result.export_visuals(
            export_dir=OUTPUT_DIR, 
            file_name=f"{file_name}_sahi_1024"
        )
        
        print(f"✅ 분석 완료! 결과 확인: {OUTPUT_DIR}/{file_name}_sahi_1024.png")
        print(f"📊 검출된 총 객체 수: {len(result.object_prediction_list)}")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("💡 팁: 만약 OBB 박스가 그려지지 않는다면, 'best.pt'가 실제 OBB 모델인지 재확인하십시오.")

if __name__ == "__main__":
    run_sahi_optimized()