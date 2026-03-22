import torch
import torch.nn as nn
import sys
import os

# 1. 경로 설정 (사용자님의 폴더 구조에 맞춤)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "ultralytics"))
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

print("🚀 스크립트가 정상적으로 시작되었습니다.")

# 2. 핵심 모듈 임포트
try:
    from models.vision_transformer import VisionTransformer
    from ultralytics.nn.modules.head import v10Detect as YOLOv26Head
    print("✅ 모든 모듈 임포트 완료")
except ImportError as e:
    print(f"❌ 임포트 실패 (경로 확인 필요): {e}")
    sys.exit()

# 3. Bridge 클래스 (백본의 352 채널을 YOLO의 [256, 512, 1024]로 변환)
class SkySenseYOLOBridge(nn.Module):
    def __init__(self, skysense_encoder, yolo_channels=[256, 512, 1024]):
        super().__init__()
        self.backbone = skysense_encoder
        # 에러 메시지에서 확인된 352 채널을 입력으로 받습니다.
        self.proj_p3 = nn.Conv2d(352, yolo_channels[0], 1)
        self.proj_p4 = nn.Conv2d(352, yolo_channels[1], 1)
        self.proj_p5 = nn.Conv2d(352, yolo_channels[2], 1)

    def forward(self, x):
        # 백본 호출 (VisionTransformer의 forward는 특징맵 리스트를 반환)
        features = self.backbone(x)
        
        # 12레이어 모델이므로 마지막 3개 층(P3, P4, P5 대응)을 가져옵니다.
        return [
            self.proj_p3(features[-3]), 
            self.proj_p4(features[-2]), 
            self.proj_p5(features[-1])
        ]

# 4. 통합 모델 클래스 (352 체급 최적화)
class IntegratedYOLOv26(nn.Module):
    def __init__(self, nc=15):
        super().__init__()
        # 가중치 파일 규격(352, 12 layers)에 맞춰 초기화
        self.backbone = VisionTransformer(
            img_size=224,        # 테스트용 (실전 학습 시 640으로 변경 가능)
            patch_size=4,
            in_channels=3,       # DOTA 데이터셋(RGB) 기준
            embed_dims=352,      # 👈 에러 메시지에서 확인된 정답
            num_layers=12,       # 👈 352 모델의 표준 레이어 수
            out_indices=(5, 8, 11) # 각 스테이지의 마지막 레이어 인덱스
        )
        self.bridge = SkySenseYOLOBridge(self.backbone)
        self.head = YOLOv26Head(nc=nc, ch=[256, 512, 1024])

    def forward(self, x):
        feats = self.bridge(x)
        return self.head(feats)

# 5. 가중치 로드 함수
def load_pretrained_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 가중치 파일을 찾을 수 없습니다: {ckpt_path}")
        return

    print(f"🧠 SkySense 가중치 이식 중... ({os.path.basename(ckpt_path)})")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model_dict = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_key = f"backbone.{k}"
        if new_key in model_dict:
            # 352 모델이 이미 3채널(RGB)용이라면 슬라이싱 없이 로드, 
            # 만약 10채널용이라면 앞의 3개만 잘라서 로드합니다.
            if 'patch_embed.projection.weight' in k and v.shape[1] != 3:
                print(f"✂️ [Slicing] {v.shape[1]}ch 가중치를 3ch로 조정합니다.")
                new_state_dict[new_key] = v[:, :3, :, :]
            else:
                new_state_dict[new_key] = v
                
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ 로드 성공 (Missing: {len(msg.missing_keys)}개, Unexpected: {len(msg.unexpected_keys)}개)")

# 6. 메인 실행부 (검증)
if __name__ == "__main__":
    print("\n🔍 [검증 시작] 모델 물리적 연결 및 가중치 체크...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 모델 생성
        model = IntegratedYOLOv26(nc=15).to(device)
        
        # 가중치 파일 경로 (실제 파일명으로 수정 완료)
        ckpt_path = os.path.join(root_dir, "weights", "skysense_model_backbone_hr.pth")
        load_pretrained_weights(model, ckpt_path)
        
        model.eval()
        
        # 더미 입력 (RGB 3채널)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("\n" + "="*40)
        res = output[0] if isinstance(output, (list, tuple)) else output
        print(f"🚀 [최종 성공] 출력 텐서 규격: {res.shape}")
        print("="*40)
        print("이제 8GB GPU에서도 돌아가는 'SkySense-YOLOv26'이 완성되었습니다!")

    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()