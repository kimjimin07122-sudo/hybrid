import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Current Device: {torch.cuda.get_device_name(0)}")
# 아래 결과가 sm_120을 포함해야 합니다.
print(f"Supported Archs: {torch.cuda.get_arch_list()}")