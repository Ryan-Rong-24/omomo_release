#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import rerun as rr
from tqdm import tqdm
from pathlib import Path

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from quaternion import quat_from_6v

def load_numpy_data(file_path):
    """Load numpy data from file."""
    return np.load(file_path)

def pose_9d_to_se3(pose_9d):
    """
    Convert 9D pose (3D translation + 6D rotation) to SE3 transform.
    
    Args:
        pose_9d: numpy array of shape [9] -> [tx, ty, tz, r1, r2, r3, r4, r5, r6]
    
    Returns:
        SE3 transform
    """
    translation = pose_9d[:3]  # [3]
    rotation_6d = pose_9d[3:]  # [6]
    
    # Convert 6D rotation back to quaternion
    rotation_6d_tensor = torch.tensor(rotation_6d, dtype=torch.float32).unsqueeze(0)  # [1, 6]
    quaternion_tensor = quat_from_6v(rotation_6d_tensor)  # [1, 4]
    quaternion = quaternion_tensor.squeeze(0).numpy()  # [4]
    
    # Convert quaternion to rotation matrix
    # quaternion is [x, y, z, w], we need to convert to rotation matrix
    x, y, z, w = quaternion
    
    # Quaternion to rotation matrix formula
    rotation_matrix = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    # Create SE3 transform using from_matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    se3_transform = SE3.from_matrix(transformation_matrix)
    
    return se3_transform

def log_pose(label: str, pose: SE3, static=False):
    """Log a pose to rerun using the same pattern as Hot3DVisualizer."""
    rr.log(label, ToTransform3D(pose, False), static=static)

def visualize_training_results(results_dir, max_frames=None):
    """
    Visualize training results by comparing ground truth vs sampled trajectories.
    
    Args:
        results_dir: Directory containing the training results
        max_frames: Maximum number of frames to visualize
    """
    results_path = Path(results_dir)
    
    # Load the saved results
    sampled_motion_path = results_path / "sampled_motion.npy"
    ground_truth_path = results_path / "ground_truth_object.npy"
    hand_poses_path = results_path / "input_hand_poses.npy"
    
    if not all(p.exists() for p in [sampled_motion_path, ground_truth_path, hand_poses_path]):
        missing = [p.name for p in [sampled_motion_path, ground_truth_path, hand_poses_path] if not p.exists()]
        print(f"Missing files: {missing}")
        print(f"Make sure the training has completed and saved results to {results_dir}")
        return
    
    print("Loading training results...")
    sampled_motion = load_numpy_data(sampled_motion_path)  # [1, T, 9]
    ground_truth_object = load_numpy_data(ground_truth_path)  # [1, T, 9]  
    input_hand_poses = load_numpy_data(hand_poses_path)  # [1, T, 18]
    
    # Remove batch dimension
    sampled_motion = sampled_motion.squeeze(0)  # [T, 9]
    ground_truth_object = ground_truth_object.squeeze(0)  # [T, 9]
    input_hand_poses = input_hand_poses.squeeze(0)  # [T, 18]
    
    # Split hand poses
    left_hand_poses = input_hand_poses[:, :9]  # [T, 9]
    right_hand_poses = input_hand_poses[:, 9:]  # [T, 9]
    
    seq_length = sampled_motion.shape[0]
    print(f"Sequence length: {seq_length}")
    
    # Limit frames if specified
    if max_frames is not None and seq_length > max_frames:
        seq_length = max_frames
        sampled_motion = sampled_motion[:max_frames]
        ground_truth_object = ground_truth_object[:max_frames]
        left_hand_poses = left_hand_poses[:max_frames]
        right_hand_poses = right_hand_poses[:max_frames]
        print(f"Limiting to {max_frames} frames")
    
    # Initialize rerun using the exact pattern from Hot3D Tutorial
    rr.init("Training Results: Ground Truth vs Sampled")
    rec = rr.memory_recording()
    
    rr.set_time_seconds("session_time", 0)  # Set initial time
    
    # Configure world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Collect trajectory points for line strips
    sampled_translations = []
    ground_truth_translations = []
    left_hand_translations = []
    right_hand_translations = []
    
    print("Processing trajectory frames...")
    
    # Subsample for performance
    subsample_rate = max(1, seq_length // 500)  # Aim for ~500 poses max
    
    for i in tqdm(range(seq_length)):
        # Set rerun timeline
        rr.set_time_seconds("session_time", i * 0.033)  # ~30 FPS
        rr.set_time_sequence("frame", i)
        
        # Convert poses to SE3
        sampled_se3 = pose_9d_to_se3(sampled_motion[i])
        ground_truth_se3 = pose_9d_to_se3(ground_truth_object[i])
        left_hand_se3 = pose_9d_to_se3(left_hand_poses[i])
        right_hand_se3 = pose_9d_to_se3(right_hand_poses[i])
        
        # Collect translations for trajectory lines
        sampled_translations.append(sampled_se3.translation())
        ground_truth_translations.append(ground_truth_se3.translation())
        left_hand_translations.append(left_hand_se3.translation())
        right_hand_translations.append(right_hand_se3.translation())
        
        # Log poses (subsampled for performance)
        if i % subsample_rate == 0:
            log_pose("world/objects/sampled_object", sampled_se3)
            log_pose("world/objects/ground_truth_object", ground_truth_se3)
            log_pose("world/hands/left_hand", left_hand_se3)
            log_pose("world/hands/right_hand", right_hand_se3)
    
    # Log trajectory lines (static)
    print("Logging trajectory lines...")

    # Convert translation lists to proper numpy arrays and reshape
    sampled_translations = np.array(sampled_translations).squeeze()  # Remove extra dimensions
    ground_truth_translations = np.array(ground_truth_translations).squeeze()
    left_hand_translations = np.array(left_hand_translations).squeeze()
    right_hand_translations = np.array(right_hand_translations).squeeze()

    rr.log("world/trajectories/sampled_object_trajectory", 
           rr.LineStrips3D([sampled_translations], colors=[255, 100, 100]), static=True)  # Light red
    rr.log("world/trajectories/ground_truth_object_trajectory", 
           rr.LineStrips3D([ground_truth_translations], colors=[0, 255, 0]), static=True)  # Green
    rr.log("world/trajectories/left_hand_trajectory", 
           rr.LineStrips3D([left_hand_translations], colors=[255, 0, 0]), static=True)  # Red
    rr.log("world/trajectories/right_hand_trajectory", 
           rr.LineStrips3D([right_hand_translations], colors=[0, 0, 255]), static=True)  # Blue
    
    # Add origin for reference using from_matrix
    origin_transformation_matrix = np.eye(4)
    origin_se3 = SE3.from_matrix(origin_transformation_matrix)
    log_pose("world/origin", origin_se3, static=True)
    
    # Calculate and log some metrics
    position_errors = []
    for i in range(seq_length):
        gt_pos = ground_truth_translations[i]
        sampled_pos = sampled_translations[i] 
        error = np.linalg.norm(gt_pos - sampled_pos)
        position_errors.append(error)
        
        # Log error as scalar
        rr.set_time_sequence("frame", i)
        rr.log("metrics/position_error", rr.Scalar(error))
    
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    
    print("Trajectory lines logged!")
    
    # Reset timeline to beginning for viewing
    rr.set_time_seconds("session_time", 0)
    rr.set_time_sequence("frame", 0)
    
    # Start web server - the recording is already connected to the current session
    print("Starting rerun web server...")
    rr.serve(open_browser=False, web_port=9090)
    
    print("\nVisualization ready!")
    print("Rerun server started at: http://localhost:9090")
    print("⚠️  If you see WebGPU errors, use WebGL renderer:")
    print("   http://localhost:9090/?renderer=webgl")
    print("")
    print("For remote SSH access:")
    print("  1. On your local machine: ssh -L 9090:localhost:9090 your_username@your_server")
    print("  2. Then open: http://localhost:9090/?renderer=webgl")
    print("")
    print(f"Sequence length: {seq_length}")
    print(f"Mean position error: {mean_error:.4f} meters")
    print(f"Max position error: {max_error:.4f} meters")
    print("\nTrajectory Legend:")
    print("  🔴 Red line: Left hand trajectory")  
    print("  🔵 Blue line: Right hand trajectory")
    print("  🟢 Green line: Ground truth object trajectory")
    print("  🌸 Light red line: Sampled object trajectory")
    print("\n💡 Tip: Use the timeline scrubber to navigate through the sequence")
    print("💡 Tip: Click and drag to rotate the 3D view, scroll to zoom")
    
    # Keep the server running
    input("Press Enter to exit...")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training results by comparing ground truth vs sampled trajectories."
    )
    parser.add_argument(
        "--results_dir", "-r",
        type=str,
        default="runs/overfit",
        help="Directory containing training results"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to visualize (if None, use all)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_dir).exists():
        print(f"Error: Results directory {args.results_dir} not found!")
        print("Make sure you have run the training script first.")
        return
    
    visualize_training_results(args.results_dir, max_frames=args.max_frames)

if __name__ == "__main__":
    main() 