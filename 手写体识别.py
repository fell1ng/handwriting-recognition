import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("显存:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")