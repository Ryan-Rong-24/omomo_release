#!/usr/bin/env python3
"""Demo script for training with integrated Rerun visualization"""

import subprocess
import sys
import os

def main():
    """Demo the integrated training system"""
    print("🚀 Demo: Training with Integrated Rerun Visualization")
    print("=" * 60)
    
    # Check if we have the required files
    required_files = [
        "trainer_hand_to_object_diffusion_overfit.py",
        "rerun_visualizer.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure all files are in the current directory")
        return
    
    print("✅ All required files found")
    print()
    
    # Show the available options
    print("📋 Available Training Options:")
    print("  --use_rerun          : Enable Rerun visualization with MANO models")
    print("  --use_wandb          : Enable Weights & Biases logging")
    print("  --window 128         : Set training window size")
    print("  --num_steps 10000    : Set number of training steps")
    print("  --use_velocity       : Use 12D data with velocity")
    print()
    
    # Show example commands
    print("💡 Example Commands:")
    print()
    
    # Basic training without visualization
    print("1. Basic Training (No Visualization):")
    print("   pixi run python trainer_hand_to_object_diffusion_overfit.py \\")
    print("     --data_path data/processed_data_with_velocity.pkl \\")
    print("     --save_dir runs/basic_training \\")
    print("     --num_steps 5000")
    print()
    
    # Training with Rerun visualization
    print("2. Training with Rerun Visualization:")
    print("   pixi run python trainer_hand_to_object_diffusion_overfit.py \\")
    print("     --data_path data/processed_data_with_velocity.pkl \\")
    print("     --save_dir runs/visualized_training \\")
    print("     --use_rerun \\")
    print("     --num_steps 10000 \\")
    print("     --exp_name demo_with_visualization")
    print()
    
    # Training with both Rerun and WandB
    print("3. Training with Rerun + WandB:")
    print("   pixi run python trainer_hand_to_object_diffusion_overfit.py \\")
    print("     --data_path data/processed_data_with_velocity.pkl \\")
    print("     --save_dir runs/full_monitoring \\")
    print("     --use_rerun \\")
    print("     --use_wandb \\")
    print("     --wandb_pj_name egorecon \\")
    print("     --entity egorecon \\")
    print("     --exp_name full_monitoring_demo")
    print()
    
    # Show what the visualization provides
    print("🎯 What You'll Get with Rerun Visualization:")
    print("  ✓ Real-time training progress in 3D")
    print("  ✓ Hand trajectory tracking (left=green, right=blue)")
    print("  ✓ Object motion comparison (GT=red, predicted=orange)")
    print("  ✓ Motion classification (moving/stationary)")
    print("  ✓ Best model prediction visualization")
    print("  ✓ Full trajectory comparison")
    print("  ✓ Enhanced hand articulations with MANO models")
    print("  ✓ 3D scene visualization with objects")
    print()
    
    # Show output structure
    print("📁 Output Structure:")
    print("  runs/your_experiment/")
    print("  ├── weights/                    # Model checkpoints")
    print("  │   ├── model-1000.pt")
    print("  │   ├── model-2000.pt")
    print("  │   └── best_model.pt")
    print("  ├── rerun_visualizations/       # Rerun files")
    print("  │   ├── training_step_1000.rrd")
    print("  │   ├── training_step_2000.rrd")
    print("  │   ├── full_trajectory_final.rrd")
    print("  │   └── demo_id_enhanced_hands.rrd")
    print("  ├── sampled_motion.npy          # Generated trajectories")
    print("  ├── input_hand_poses.npy        # Input hand data")
    print("  ├── ground_truth_object.npy     # Ground truth")
    print("  ├── training_summary.txt        # Training summary")
    print("  └── opt.yaml                    # Configuration")
    print()
    
    # Show how to view results
    print("👀 How to View Results:")
    print("  1. Download Rerun files from remote server:")
    print("     scp egorecon:runs/your_experiment/rerun_visualizations/*.rrd ./")
    print()
    print("  2. View training progress:")
    print("     rerun training_step_1000.rrd --web-viewer --port 9877")
    print()
    print("  3. View final trajectory:")
    print("     rerun full_trajectory_final.rrd --web-viewer --port 9878")
    print()
    print("  4. View enhanced scene:")
    print("     rerun demo_id_enhanced_hands.rrd --web-viewer --port 9879")
    print()
    
    # Check if user wants to run a demo
    print("🤔 Would you like to run a quick demo? (y/n): ", end="")
    try:
        user_input = input().strip().lower()
        if user_input in ['y', 'yes']:
            print("\n🚀 Starting demo training...")
            print("   This will run a short training session with visualization")
            print("   Press Ctrl+C to stop early")
            print()
            
            # Run a quick demo
            demo_cmd = [
                "pixi", "run", "python", "trainer_hand_to_object_diffusion_overfit.py",
                "--data_path", "data/processed_data_with_velocity.pkl",
                "--save_dir", "runs/demo_training",
                "--use_rerun",
                "--num_steps", "1000",
                "--exp_name", "quick_demo"
            ]
            
            print(f"Running: {' '.join(demo_cmd)}")
            print()
            
            try:
                subprocess.run(demo_cmd, check=True)
            except KeyboardInterrupt:
                print("\n⏹️  Demo stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Demo failed with error: {e}")
            except FileNotFoundError:
                print("\n❌ 'pixi' command not found. Please ensure pixi is installed and in PATH")
        else:
            print("\n👋 Demo skipped. You can run the training commands manually when ready!")
    
    except KeyboardInterrupt:
        print("\n\n👋 Demo setup completed. You can run the training commands manually!")
    
    print("\n✨ Happy training with Rerun visualization!")

if __name__ == "__main__":
    main()
