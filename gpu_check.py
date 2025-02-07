# gpu_check.py
import torch

def check_gpu():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current Device:", torch.cuda.current_device())
        print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__ == "__main__":
    check_gpu()