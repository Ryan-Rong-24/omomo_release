#!/usr/bin/env python3
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'Current device: {torch.cuda.current_device()}')
else:
    print('CUDA not detected by PyTorch')
    print('This could be due to:')
    print('1. PyTorch CPU-only version installed')
    print('2. CUDA version mismatch')
    print('3. Environment configuration issue')
