#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import torch
import rerun as rr
from tqdm import tqdm
from pathlib import Path

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from quaternion import quat_from_6v

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

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

def visualize_trajectories(data_path, demo_id=None, target_object_id=None, max_frames=None):
    """
    Visualize hand and object trajectories using rerun.
    
    Args:
        data_path: Path to processed data pickle file
        demo_id: Specific demo to visualize (if None, use first)
        target_object_id: Specific object to visualize (if None, use first)
        max_frames: Maximum number of frames to visualize (if None, use all)
    """
    print(f"Loading data from {data_path}...")
    processed_data = load_pickle(data_path)
    
    # Select demonstration
    if demo_id is None:
        demo_id = list(processed_data.keys())[0]
    
    if demo_id not in processed_data:
        print(f"Demo {demo_id} not found! Available demos: {list(processed_data.keys())[:5]}...")
        return
    
    demo_data = processed_data[demo_id]
    print(f"Visualizing demonstration: {demo_id}")
    
    # Get trajectory data
    left_hand_data = demo_data['left_hand']['poses_9d']  # [T, 9]
    right_hand_data = demo_data['right_hand']['poses_9d']  # [T, 9]
    
    # Select target object
    objects_data = demo_data['objects']
    if target_object_id is None:
        target_object_id = list(objects_data.keys())[0]
    
    if target_object_id not in objects_data:
        print(f"Object {target_object_id} not found! Available objects: {list(objects_data.keys())}")
        return
    
    object_data = objects_data[target_object_id]['poses_9d']  # [T, 9]
    print(f"Visualizing object: {target_object_id}")
    
    # Find common length
    min_length = min(len(left_hand_data), len(right_hand_data), len(object_data))
    print(f"Trajectory lengths - Left: {len(left_hand_data)}, Right: {len(right_hand_data)}, Object: {len(object_data)}")
    print(f"Using common length: {min_length}")
    
    # Limit frames if specified
    if max_frames is not None and min_length > max_frames:
        min_length = max_frames
        print(f"Limiting to {max_frames} frames")
    
    # Truncate to common length
    left_hand_data = left_hand_data[:min_length]
    right_hand_data = right_hand_data[:min_length]
    object_data = object_data[:min_length]
    
    # Initialize rerun using the same pattern as Hot3D Tutorial
    rr.init(f"Hand-Object Trajectories: {demo_id}")
    
    # Configure rerun to use WebGL renderer and bind to all interfaces for remote access
    rr.serve(
        open_browser=False, 
        web_port=9090,
        ws_port=9091,
        server_memory_limit="2GB"
    )
    
    # Set rerun to use WebGL renderer by default (avoids WebGPU issues)
    # Users can access with: http://localhost:9090/?renderer=webgl
    
    rr.set_time_seconds("session_time", 0)  # Set initial time
    
    # Configure world coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Collect trajectory points for line strips
    left_hand_translations = []
    right_hand_translations = []
    object_translations = []
    
    print("Processing trajectory frames...")
    
    # Subsample for performance
    subsample_rate = max(1, min_length // 1000)  # Aim for ~1000 poses max
    
    for i in tqdm(range(min_length)):
        # Set rerun timeline
        rr.set_time_seconds("session_time", i * 0.033)  # ~30 FPS
        rr.set_time_sequence("frame", i)
        
        # Convert poses to SE3
        left_hand_se3 = pose_9d_to_se3(left_hand_data[i])
        right_hand_se3 = pose_9d_to_se3(right_hand_data[i])
        object_se3 = pose_9d_to_se3(object_data[i])
        
        # Collect translations for trajectory lines
        left_hand_translations.append(left_hand_se3.translation())
        right_hand_translations.append(right_hand_se3.translation())
        object_translations.append(object_se3.translation())
        
        # Log poses (subsampled for performance)
        if i % subsample_rate == 0:
            log_pose("world/hands/left_hand", left_hand_se3)
            log_pose("world/hands/right_hand", right_hand_se3)
            log_pose(f"world/objects/{target_object_id}", object_se3)
    
    # Log trajectory lines (static)
    print("Logging trajectory lines...")
    
    # Convert translation lists to proper numpy arrays and reshape
    left_hand_translations = np.array(left_hand_translations).squeeze()  # Remove extra dimensions
    right_hand_translations = np.array(right_hand_translations).squeeze()
    object_translations = np.array(object_translations).squeeze()
    
    rr.log("world/trajectories/left_hand_trajectory", 
           rr.LineStrips3D([left_hand_translations], colors=[255, 0, 0]), static=True)
    rr.log("world/trajectories/right_hand_trajectory", 
           rr.LineStrips3D([right_hand_translations], colors=[0, 0, 255]), static=True)
    rr.log("world/trajectories/object_trajectory", 
           rr.LineStrips3D([object_translations], colors=[0, 255, 0]), static=True)
    
    # Add origin for reference using from_matrix
    origin_transformation_matrix = np.eye(4)
    origin_se3 = SE3.from_matrix(origin_transformation_matrix)
    log_pose("world/origin", origin_se3, static=True)
    
    print("\nVisualization ready!")
    print("Rerun server started at: http://localhost:9090")
    print("⚠️  If you see WebGPU errors, use WebGL renderer:")
    print("   http://localhost:9090/?renderer=webgl")
    print("")
    print("For remote SSH access:")
    print("  1. On your local machine: ssh -L 9090:localhost:9090 your_username@your_server")
    print("  2. Then open: http://localhost:9090/?renderer=webgl")
    print("")
    print(f"Demo: {demo_id}")
    print(f"Object: {target_object_id}")
    print(f"Frames: {min_length}")
    print(f"Left hand trajectory: {len(left_hand_translations)} points")
    print(f"Right hand trajectory: {len(right_hand_translations)} points")
    print(f"Object trajectory: {len(object_translations)} points")
    print("\nTrajectory Legend:")
    print("  🔴 Red line: Left hand trajectory")
    print("  🔵 Blue line: Right hand trajectory")
    print("  🟢 Green line: Object trajectory")
    print("\n💡 Tip: Use the timeline scrubber to navigate through the sequence")
    print("💡 Tip: Click and drag to rotate the 3D view, scroll to zoom")
    
    # Keep the server running
    input("Press Enter to exit...")

def list_available_data(data_path):
    """List available demonstrations and objects in the data."""
    processed_data = load_pickle(data_path)
    
    print(f"Available demonstrations ({len(processed_data)} total):")
    for i, (demo_id, demo_data) in enumerate(processed_data.items()):
        if i >= 5:  # Show first 5 demos
            print(f"... and {len(processed_data) - 5} more")
            break
        
        print(f"\n  {demo_id}:")
        if 'left_hand' in demo_data:
            print(f"    Left hand: {demo_data['left_hand']['poses_9d'].shape}")
        if 'right_hand' in demo_data:
            print(f"    Right hand: {demo_data['right_hand']['poses_9d'].shape}")
        if 'objects' in demo_data:
            print(f"    Objects ({len(demo_data['objects'])} total):")
            for j, (obj_id, obj_data) in enumerate(demo_data['objects'].items()):
                if j >= 3:  # Show first 3 objects per demo
                    print(f"      ... and {len(demo_data['objects']) - 3} more")
                    break
                print(f"      {obj_id}: {obj_data['poses_9d'].shape}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize hand and object trajectories using rerun."
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        default="data/processed_data.pkl",
        help="Path to processed data pickle file"
    )
    parser.add_argument(
        "--demo_id",
        type=str,
        default=None,
        help="Specific demo ID to visualize (if None, use first available)"
    )
    parser.add_argument(
        "--target_object_id",
        type=str,
        default=None,
        help="Specific object ID to visualize (if None, use first available)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to visualize (if None, use all)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available demonstrations and objects"
    )
    
    args = parser.parse_args()
    
    if not Path(args.data_path).exists():
        print(f"Error: Data file {args.data_path} not found!")
        return
    
    if args.list:
        list_available_data(args.data_path)
        return
    
    visualize_trajectories(
        args.data_path,
        demo_id=args.demo_id,
        target_object_id=args.target_object_id,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main() 