import torch
from ultralytics import YOLO

# 1. 모델 로드 (폴더에 있는 yolo26s-obb.pt 사용)
model = YOLO("yolo26s-obb.pt")

# 2. GPU 연결 상태 최종 확인
print(f"사용 중인 장치: {model.device}")

# 3. 간단한 추론 테스트 (이미지 파일이 있다면 경로를 넣으세요)
# 만약 테스트용 이미지가 없다면 아래 줄은 주석 처리해도 됩니다.
# results = model.predict(source="skysense_feature_test.png", save=True, device=0)

print("\n🔥 모든 준비가 완료되었습니다. 이제 연구를 시작하셔도 좋습니다!")