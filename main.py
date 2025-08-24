#!/usr/bin/env python3
"""Main entry point for hand-to-object diffusion training"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training import train_overfit
from src.training.trainer_hand_to_object_diffusion_overfit import parse_opt

def main():
    """Main entry point"""
    print("🚀 Hand-to-Object Diffusion Training")
    print("=" * 40)
    
    # Parse command line arguments
    opt = parse_opt()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run training
    train_overfit(opt, device)

if __name__ == "__main__":
    main()
