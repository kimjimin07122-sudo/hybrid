import torch
import torch.nn as nn
import sys
import os

# 1. 경로 설정 (사용자님의 폴더 구조 최적화)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "ultralytics"))
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

print("🚀 [Build] 하이브리드 모델 설계 엔진 가동...")

# 2. 핵심 모듈 임포트
try:
    from models.vision_transformer import VisionTransformer
    # DOTA는 OBB(회전박스) 작업이므로 OBB 전용 Head를 호출해야 합니다.
    from ultralytics.nn.modules.head import OBBDetect 
    print("✅ SkySense 백본 및 YOLO-OBB Head 임포트 완료")
except ImportError as e:
    print(f"❌ 임포트 실패: {e}\n'ultralytics' 폴더 내에 OBBDetect가 있는지 확인하십시오.")
    sys.exit()

# 3. Bridge 클래스 (352ch -> YOLO 표준 채널 변환)
class SkySenseYOLOBridge(nn.Module):
    def __init__(self, skysense_encoder, yolo_channels=[256, 512, 1024]):
        super().__init__()
        self.backbone = skysense_encoder
        # 백본의 352 채널을 YOLO Neck/Head가 인식하는 표준 채널로 사영(Projection)
        self.proj_p3 = nn.Conv2d(352, yolo_channels[0], 1)
        self.proj_p4 = nn.Conv2d(352, yolo_channels[1], 1)
        self.proj_p5 = nn.Conv2d(352, yolo_channels[2], 1)

    def forward(self, x):
        features = self.backbone(x) # 반환값: [layer1, layer2, ... layer12] 특징맵 리스트
        
        # P3, P4, P5 단계에 대응하는 특징맵 추출 (마지막 3개 층)
        return [
            self.proj_p3(features[-3]), 
            self.proj_p4(features[-2]), 
            self.proj_p5(features[-1])
        ]

# 4. 통합 하이브리드 모델 (1024px 연구용 최적화)
class IntegratedYOLOv26_OBB(nn.Module):
    def __init__(self, nc=15):
        super().__init__()
        # 5070 자원을 고려하여 1024 해상도 대응 설정
        self.backbone = VisionTransformer(
            img_size=1024,       # 👈 테스트용 224에서 1024로 상향
            patch_size=4,
            in_channels=3, 
            embed_dims=352,      # 352 체급 고정
            num_layers=12,
            out_indices=(5, 8, 11) 
        )
        self.bridge = SkySenseYOLOBridge(self.backbone)
        # OBBDetect를 사용하여 DOTA 데이터셋의 회전 박스를 지원합니다.
        self.head = OBBDetect(nc=nc, ch=[256, 512, 1024])

    def forward(self, x):
        feats = self.bridge(x)
        return self.head(feats)

# 5. 지능형 가중치 로드 함수
def load_skysense_weights(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 가중치 파일 부재: {ckpt_path}")
        return

    print(f"🧠 가중치 이식 시작: {os.path.basename(ckpt_path)}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model_dict = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_key = f"backbone.{k}" # Integrated 모델 내 backbone 경로로 매핑
        if new_key in model_dict:
            # 입력 채널 불일치 해결 (10ch 위성 -> 3ch RGB)
            if 'patch_embed.projection.weight' in k and v.shape[1] != 3:
                print(f"✂️ [Channel Slicing] {v.shape[1]}ch -> 3ch 변환")
                new_state_dict[new_key] = v[:, :3, :, :]
            else:
                new_state_dict[new_key] = v
                
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ 이식 완료 (성공: {len(new_state_dict)}개, 누락: {len(msg.missing_keys)}개)")

# 6. 최종 빌드 검증
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔍 [검증] RTX 5070 장치 할당: {device}")
    
    try:
        # 모델 생성 (DOTA 클래스 15개 기준)
        model = IntegratedYOLOv26_OBB(nc=15).to(device)
        
        # 가중치 이식
        ckpt_path = os.path.join(root_dir, "weights", "skysense_model_backbone_hr.pth")
        load_skysense_weights(model, ckpt_path)
        
        model.eval()
        
        # 1024px 더미 데이터로 VRAM 부하 테스트
        dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print("\n" + "="*50)
        # OBB 출력은 리스트 형태일 수 있으므로 규격 확인
        res = output[0] if isinstance(output, (list, tuple)) else output
        print(f"🚀 [빌드 성공] 최종 출력 텐서 모양: {res.shape}")
        print(f"📊 메모리 점유: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print("="*50)

    except Exception as e:
        print(f"❌ 빌드 실패: {e}")
        import traceback
        traceback.print_exc()