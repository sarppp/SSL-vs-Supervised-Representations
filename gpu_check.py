import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Can't initialize NVML")
def gpu_status():
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    statuses = []
    for idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(idx)
        name = torch.cuda.get_device_name(idx)
        free, total = torch.cuda.mem_get_info()
        statuses.append(f"GPU{idx} {name}: {free/1e9:.2f}GB free / {total/1e9:.2f}GB")
    return " | ".join(statuses)

print(gpu_status())