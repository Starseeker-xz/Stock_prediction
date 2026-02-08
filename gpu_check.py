#!/usr/bin/env python3
"""
PyTorch GPU 检测脚本
"""

import torch


def check_gpu():
    """检测 GPU 信息"""
    print("=" * 50)
    print("PyTorch GPU 检测")
    print("=" * 50)
    
    # 基本信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN 可用: {torch.backends.cudnn.enabled}")
        
        print("\nGPU 详细信息:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU")
    
    print("=" * 50)


if __name__ == "__main__":
    check_gpu()
