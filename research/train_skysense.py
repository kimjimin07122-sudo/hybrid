import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import yaml
from tqdm import tqdm
from types import SimpleNamespace

# 0. 경로 정의
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 1. 모듈 경로 등록 및 임포트
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "skysense", "SkySense"))

try:
    from model_fusion import IntegratedYOLOv26, load_pretrained_weights
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG_DICT
    print("✅ 모든 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ 임포트 실패: {e}")
    sys.exit()

# 2. RTX 2080 (8GB) 최적화 설정
CONFIG = {
    "data_yaml": os.path.join(root_dir, "ultralytics/ultralytics/cfg/datasets/DOTAv1_Tiled_448.yaml"),
    "ckpt_path": os.path.join(root_dir, "weights/skysense_model_backbone_hr.pth"),
    "imgsz": 320,            # 8GB VRAM 생존 해상도
    "batch_size": 1,         
    "accumulate": 32,        # 실제 배치 32 효과 (1x32)
    "epochs": 100,
    "lr": 1e-4,              
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def train():
    device = CONFIG["device"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    # 3. 모델 빌드 및 텐서보드 설정
    print("🏗️ 모델 빌드 및 가중치 이식 중...")
    model = IntegratedYOLOv26(nc=15).to(device)
    load_pretrained_weights(model, CONFIG["ckpt_path"])

    # 텐서보드 기록기 (로그 저장 경로)
    writer = SummaryWriter(log_dir=os.path.join(root_dir, "runs/skysense_320_experiment"))

    # 백본 동결 (8GB VRAM에서는 동결이 안전합니다)
    print("❄️ SkySense 백본 동결 (Memory Optimization)")
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Loss 계산을 위한 가짜 속성 주입
    model.model = nn.Sequential(model.backbone, model.bridge, model.head)
    model.stride = torch.tensor([8, 16, 32])
    model.nc = 15
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, tal_topk=10, degrees=0.0)
    
    criterion = v8DetectionLoss(model) 

    # 4. 데이터셋 로드
    print("📡 타일링된 데이터셋 로드 중...")
    with open(CONFIG["data_yaml"], 'r') as f:
        data_info = yaml.safe_load(f)
    
    img_path_train = os.path.join(data_info['path'], data_info['train'])
    cfg = get_cfg(DEFAULT_CFG_DICT)
    cfg.imgsz, cfg.task, cfg.data, cfg.batch = CONFIG["imgsz"], 'detect', CONFIG["data_yaml"], CONFIG["batch_size"]

    dataset = build_yolo_dataset(cfg=cfg, img_path=img_path_train, batch=CONFIG["batch_size"], data=data_info, mode='train', stride=32)
    loader = build_dataloader(dataset, batch=CONFIG["batch_size"], workers=2, shuffle=True)

    # 5. 옵티마이저 및 AMP 설정
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scaler = GradScaler('cuda') 

    print(f"🚀 학습 시작! (VRAM Usage: ~4.6GB)")

    global_step = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")

        epoch_loss = 0
        for i, batch in pbar:
            imgs = batch['img'].to(device).float() / 255.0
            
            with autocast('cuda'):
                preds = model(imgs)
                train_preds = preds["one2many"] if isinstance(preds, dict) else preds
                raw_loss, _ = criterion(train_preds, batch)
                loss = raw_loss.sum() / CONFIG["accumulate"]

            scaler.scale(loss).backward()

            if (i + 1) % CONFIG["accumulate"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 실시간 Loss 기록 (텐서보드용)
                current_loss_val = loss.item() * CONFIG["accumulate"]
                writer.add_scalar("Loss/train_step", current_loss_val, global_step)
                pbar.set_postfix({"loss": f"{current_loss_val:.4f}"})
                global_step += 1
                epoch_loss += current_loss_val

        # 에포크당 평균 Loss 기록
        avg_epoch_loss = epoch_loss / (len(loader) / CONFIG["accumulate"])
        writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)

        # 5에포크마다 모델 저장
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(root_dir, f"weights/skysense_320_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 체크포인트 저장: {save_path}")

    writer.close()
    print("🎉 모든 학습 완료!")

if __name__ == "__main__":
    train()