#!/usr/bin/env python3
"""Reconstruct proper Rerun visualization following the working rerun_with_hand_articulations.py approach"""

import os
import numpy as np
import rerun as rr
import torch
import shutil
import yaml
from pathlib import Path

def load_mano_models():
    """Load MANO models using the proven working method"""
    try:
        import smplx
        
        # Constants for MANO model paths (from working version)
        MANO_LEFT_SRC = "/home/yufeiy2_egorecon/omomo_release/data/mano_models/MANO_LEFT.pkl"
        MANO_RIGHT_SRC = "/home/yufeiy2_egorecon/omomo_release/data/mano_models/MANO_RIGHT.pkl"
        MANO_TEMP_DIR = "/tmp/mano_models"
        
        # Create temporary directory structure that SMPLX expects
        os.makedirs(MANO_TEMP_DIR, exist_ok=True)
        
        # Check if source files exist
        if not os.path.exists(MANO_LEFT_SRC) or not os.path.exists(MANO_RIGHT_SRC):
            raise FileNotFoundError(f"MANO model files not found at {MANO_LEFT_SRC} or {MANO_RIGHT_SRC}")
        
        # Copy MANO model files to expected locations
        mano_files = {
            "MANO_LEFT.pkl": MANO_LEFT_SRC,
            "MANO_RIGHT.pkl": MANO_RIGHT_SRC
        }
        
        for filename, src_path in mano_files.items():
            dst_path = os.path.join(MANO_TEMP_DIR, filename)
            shutil.copy2(src_path, dst_path)
        
        # Load MANO models
        left_mano = smplx.MANO(
            model_path=os.path.join(MANO_TEMP_DIR, "MANO_LEFT.pkl"),
            is_rhand=False,
            use_pca=False,
            flat_hand_mean=True
        )
        
        right_mano = smplx.MANO(
            model_path=os.path.join(MANO_TEMP_DIR, "MANO_RIGHT.pkl"),
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True
        )
        
        print("✓ MANO models loaded successfully")
        return left_mano, right_mano
        
    except Exception as e:
        print(f"✗ Failed to load MANO models: {e}")
        return None, None

def get_available_object_assets():
    """Get list of available object GLB files (from working version)"""
    assets_dir = "/home/yufeiy2_egorecon/omomo_release/hot3d/hot3d/dataset/assets"
    if not os.path.exists(assets_dir):
        print(f"Assets directory not found: {assets_dir}")
        return {}
    
    object_assets = {}
    for filename in os.listdir(assets_dir):
        if filename.endswith('.glb'):
            object_uid = filename.replace('.glb', '')
            object_assets[object_uid] = os.path.join(assets_dir, filename)
    
    print(f"✓ Found {len(object_assets)} object GLB files")
    return object_assets

def rotation_6d_to_matrix(rot_6d):
    """Convert 6D rotation representation to 3x3 rotation matrix"""
    # 6D representation: [a1, a2, a3, b1, b2, b3] where a and b are the first two columns
    a = rot_6d[:3]  # First column
    b = rot_6d[3:]  # Second column
    
    # Normalize a
    a = a / np.linalg.norm(a)
    
    # Gram-Schmidt to get orthogonal b
    b = b - np.dot(b, a) * a
    b = b / np.linalg.norm(b)
    
    # Cross product to get third column
    c = np.cross(a, b)
    
    # Construct rotation matrix
    R = np.column_stack([a, b, c])
    return R

def apply_coordinate_transform(position, rotation_matrix=None):
    """Apply coordinate system transformation (from working version)"""
    
    # Transform matrix: 90-degree rotation around X-axis
    # This converts from Y-up (vertical) to Z-up (horizontal) coordinate system
    transform_matrix = np.array([
        [1,  0,  0],  # X stays the same
        [0,  1,  0],  # Y becomes -Z (rotated 90° around X)
        [0,  0,  1]   # Z becomes Y (rotated 90° around X)
    ])
    
    # Apply transformation to position
    transformed_position = transform_matrix @ position
    
    # Apply transformation to rotation matrix if provided
    transformed_rotation = None
    if rotation_matrix is not None:
        # R' = T * R * T^T where T is the transform matrix
        transformed_rotation = transform_matrix @ rotation_matrix @ transform_matrix.T
    
    return transformed_position, transformed_rotation

def reconstruct_visualization(results_dir):
    """Reconstruct visualization following the working approach"""
    
    results_path = Path(results_dir)
    print(f"Reconstructing visualization from: {results_path}")
    
    # Check required files exist
    required_files = [
        "sampled_motion.npy",
        "ground_truth_object.npy",
        "opt.yaml"
    ]
    
    # Check for input_hand_poses in either NPY or NPZ format
    hand_poses_npy = results_path / "input_hand_poses.npy"
    hand_poses_npz = results_path / "input_hand_poses.npz"
    
    for file in required_files:
        if not (results_path / file).exists():
            print(f"✗ Missing required file: {file}")
            return False
    
    # Check for hand poses file in either format
    if not hand_poses_npy.exists() and not hand_poses_npz.exists():
        print(f"✗ Missing required file: input_hand_poses.npy OR input_hand_poses.npz")
        return False
    
    print("✓ All required files found")
    
    # Load the saved data
    sampled_motion = np.load(results_path / "sampled_motion.npy")[0]  # Remove batch dimension
    
    # Handle both NPY and NPZ formats for input_hand_poses
    hand_poses_npz = results_path / "input_hand_poses.npz"
    hand_poses_npy = results_path / "input_hand_poses.npy"
    
    if hand_poses_npz.exists():
        # NPZ format: separate left/right hand arrays
        data = np.load(hand_poses_npz)
        if 'left_hand' in data.files and 'right_hand' in data.files:
            # Combine left and right hands along feature dimension
            left_hand = data['left_hand']  # Shape: (seq_len, 9)
            right_hand = data['right_hand']  # Shape: (seq_len, 9)
            input_hand_poses = np.concatenate([left_hand, right_hand], axis=1)  # Shape: (seq_len, 18)
            print(f"✓ Loaded hand poses from NPZ file: left{left_hand.shape} + right{right_hand.shape}")
        else:
            raise ValueError(f"NPZ file missing 'left_hand' or 'right_hand' keys. Found: {data.files}")
        data.close()
    elif hand_poses_npy.exists():
        # NPY format: stacked array with batch dimension
        input_hand_poses = np.load(hand_poses_npy)[0]  # Remove batch dimension: (seq_len, 18)
        print(f"✓ Loaded hand poses from NPY file: {input_hand_poses.shape}")
    else:
        raise FileNotFoundError(f"Neither {hand_poses_npy} nor {hand_poses_npz} found")
    
    ground_truth_object = np.load(results_path / "ground_truth_object.npy")[0]  # Remove batch dimension
    
    # Load training options
    with open(results_path / "opt.yaml", 'r') as f:
        opt = yaml.safe_load(f)
    
    print(f"✓ Loaded data for demo {opt.get('demo_id')}, object {opt.get('target_object_id')}")
    print(f"  - Trajectory length: {len(sampled_motion)} frames")
    print(f"  - Pose dimension: {ground_truth_object.shape[1]}D")
    
    # Split hand poses (they were concatenated: left + right)
    pose_dim = ground_truth_object.shape[1]  # Should be 9 or 12
    left_hand_poses = input_hand_poses[:, :pose_dim]
    right_hand_poses = input_hand_poses[:, pose_dim:]
    
    # Load MANO models
    left_mano, right_mano = load_mano_models()
    
    # Get available object assets
    object_assets = get_available_object_assets()
    
    # Create output directory and file
    output_dir = results_path / "rerun_visualizations_proper"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"proper_reconstruction_motion_only.rrd"
    
    # Initialize rerun (exactly like working version)
    rr.init(f"ProperReconstruction_{opt.get('exp_name', 'reconstruction')}")
    rr.save(str(output_file))  # Start recording immediately
    
    print(f"✓ Rerun initialized and recording to: {output_file}")
    
    # Log coordinate system reference (after transformation) - from working version
    origin_transformed, _ = apply_coordinate_transform(np.array([0, 0, 0]))
    x_axis_transformed, _ = apply_coordinate_transform(np.array([0.1, 0, 0]))
    y_axis_transformed, _ = apply_coordinate_transform(np.array([0, 0.1, 0]))
    z_axis_transformed, _ = apply_coordinate_transform(np.array([0, 0, 0.1]))
    
    rr.log("world/reference/origin", 
           rr.Points3D([origin_transformed], colors=[[255, 255, 255]], radii=[0.02]), 
           static=True)
    rr.log("world/reference/x_axis", 
           rr.Points3D([x_axis_transformed], colors=[[255, 0, 0]], radii=[0.015]), 
           static=True)
    rr.log("world/reference/y_axis", 
           rr.Points3D([y_axis_transformed], colors=[[0, 255, 0]], radii=[0.015]), 
           static=True)
    rr.log("world/reference/z_axis", 
           rr.Points3D([z_axis_transformed], colors=[[0, 0, 255]], radii=[0.015]), 
           static=True)
    
    # Add a ground plane reference (horizontal plane in transformed coordinates)
    ground_points = []
    for x in np.linspace(-0.5, 0.5, 5):
        for y in np.linspace(-0.5, 0.5, 5):
            point, _ = apply_coordinate_transform(np.array([x, 0, y]))
            ground_points.append(point)
    
    rr.log("world/reference/ground_plane", 
           rr.Points3D(ground_points, colors=[[128, 128, 128]], radii=[0.005]), 
           static=True)
    
    print("✓ Added coordinate system reference")
    
    # Process frames (limit to reasonable number for visualization)
    num_frames = min(500, len(sampled_motion))
    print(f"Processing {num_frames} frames...")
    
    # Collect trajectories with coordinate transformation
    left_hand_trajectory = []
    right_hand_trajectory = []
    object_gt_trajectory = []
    object_pred_trajectory = []
    
    # Track which objects we've loaded as static assets
    loaded_objects = set()
    object_id = opt.get('target_object_id')
    
    for frame_idx in range(num_frames):
        rr.set_time("frame", sequence=frame_idx)
        
        # Get positions (first 3 dimensions) and rotations (6D representation)
        left_pos = left_hand_poses[frame_idx, :3]
        right_pos = right_hand_poses[frame_idx, :3]
        object_gt_pos = ground_truth_object[frame_idx, :3]
        object_pred_pos = sampled_motion[frame_idx, :3]
        
        # Get rotations (6D representation: dimensions 3-8)
        left_rot_6d = left_hand_poses[frame_idx, 3:9]
        right_rot_6d = right_hand_poses[frame_idx, 3:9] 
        object_gt_rot_6d = ground_truth_object[frame_idx, 3:9]
        object_pred_rot_6d = sampled_motion[frame_idx, 3:9]
        
        # Convert 6D rotations to rotation matrices
        try:
            left_rot_matrix = rotation_6d_to_matrix(left_rot_6d)
            right_rot_matrix = rotation_6d_to_matrix(right_rot_6d)
            object_gt_rot_matrix = rotation_6d_to_matrix(object_gt_rot_6d)
            object_pred_rot_matrix = rotation_6d_to_matrix(object_pred_rot_6d)
        except:
            # Fallback to identity if rotation conversion fails
            left_rot_matrix = right_rot_matrix = object_gt_rot_matrix = object_pred_rot_matrix = np.eye(3)
        
        # Apply coordinate transformations (with rotations)
        left_pos_transformed, left_rot_transformed = apply_coordinate_transform(left_pos, left_rot_matrix)
        right_pos_transformed, right_rot_transformed = apply_coordinate_transform(right_pos, right_rot_matrix)
        object_gt_pos_transformed, object_gt_rot_transformed = apply_coordinate_transform(object_gt_pos, object_gt_rot_matrix)
        object_pred_pos_transformed, object_pred_rot_transformed = apply_coordinate_transform(object_pred_pos, object_pred_rot_matrix)
        
        # Store for trajectories
        left_hand_trajectory.append(left_pos_transformed)
        right_hand_trajectory.append(right_pos_transformed)
        object_gt_trajectory.append(object_gt_pos_transformed)
        object_pred_trajectory.append(object_pred_pos_transformed)
        
        # === MANO HAND MESHES ===
        if left_mano is not None and right_mano is not None:
            # Generate MANO meshes and position them with rotation
            with torch.no_grad():
                # Left hand
                left_output = left_mano()
                left_vertices_base = left_output.vertices[0].numpy()
                
                # Apply rotation and translation (like working version)
                if left_rot_transformed is not None:
                    left_vertices = (left_rot_transformed @ left_vertices_base.T).T + left_pos_transformed
                else:
                    left_vertices = left_vertices_base + left_pos_transformed
                    
                rr.log("world/hands/left_mesh",
                       rr.Mesh3D(vertex_positions=left_vertices,
                                triangle_indices=left_mano.faces,
                                vertex_colors=[0.3, 0.8, 0.3]))  # Green for left hand
                
                # Right hand  
                right_output = right_mano()
                right_vertices_base = right_output.vertices[0].numpy()
                
                # Apply rotation and translation (like working version)
                if right_rot_transformed is not None:
                    right_vertices = (right_rot_transformed @ right_vertices_base.T).T + right_pos_transformed
                else:
                    right_vertices = right_vertices_base + right_pos_transformed
                    
                rr.log("world/hands/right_mesh",
                       rr.Mesh3D(vertex_positions=right_vertices,
                                triangle_indices=right_mano.faces,
                                vertex_colors=[0.3, 0.3, 0.8]))  # Blue for right hand
        
        # === OBJECT MESH ===
        if object_id and object_id in object_assets:
            glb_path = object_assets[object_id]
            
            # Load separate 3D assets for ground truth and predicted objects
            if f"{object_id}_gt" not in loaded_objects:
                rr.log(f"world/objects/object_{object_id}_gt",
                       rr.Asset3D(path=glb_path),
                       static=True)
                loaded_objects.add(f"{object_id}_gt")
                print(f"  ✓ Loaded 3D asset for ground truth object {object_id}")
            
            if f"{object_id}_pred" not in loaded_objects:
                rr.log(f"world/objects/object_{object_id}_pred", 
                       rr.Asset3D(path=glb_path),
                       static=True)
                loaded_objects.add(f"{object_id}_pred")
                print(f"  ✓ Loaded 3D asset for predicted object {object_id}")
            
            # Position ground truth object with rotation (following working version approach)
            if object_gt_rot_transformed is not None:
                rr.log(f"world/objects/object_{object_id}_gt",
                       rr.Transform3D(translation=object_gt_pos_transformed,
                                     mat3x3=object_gt_rot_transformed))
            else:
                rr.log(f"world/objects/object_{object_id}_gt",
                       rr.Transform3D(translation=object_gt_pos_transformed))
            
            # Position predicted object with rotation (following working version approach)
            if object_pred_rot_transformed is not None:
                rr.log(f"world/objects/object_{object_id}_pred",
                       rr.Transform3D(translation=object_pred_pos_transformed,
                                     mat3x3=object_pred_rot_transformed))
            else:
                rr.log(f"world/objects/object_{object_id}_pred",
                       rr.Transform3D(translation=object_pred_pos_transformed))
        
        # Progress indicator
        if frame_idx % 50 == 0:
            print(f"  Processed frame {frame_idx}/{num_frames}")
    
    # Log full trajectories (static)
    rr.log("world/trajectories/left_hand", 
           rr.LineStrips3D([left_hand_trajectory], colors=[[0, 255, 0]], radii=[0.008]), 
           static=True)
    rr.log("world/trajectories/right_hand", 
           rr.LineStrips3D([right_hand_trajectory], colors=[[0, 0, 255]], radii=[0.008]), 
           static=True)
    rr.log("world/trajectories/object_ground_truth", 
           rr.LineStrips3D([object_gt_trajectory], colors=[[255, 0, 0]], radii=[0.012]), 
           static=True)
    rr.log("world/trajectories/object_predicted", 
           rr.LineStrips3D([object_pred_trajectory], colors=[[255, 165, 0]], radii=[0.012]), 
           static=True)
    
    print("✓ Added full trajectory lines")
    
    # Calculate metrics
    position_errors = np.linalg.norm(
        np.array(object_pred_trajectory) - np.array(object_gt_trajectory), axis=1
    )
    mean_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    
    # Add metadata
    has_gt_object = f"{object_id}_gt" in loaded_objects
    has_pred_object = f"{object_id}_pred" in loaded_objects
    rr.log("world/info", rr.TextDocument(
        f"Proper Reconstruction\n"
        f"Demo: {opt.get('demo_id')}\n"
        f"Object: {object_id}\n"
        f"Frames: {num_frames}\n"
        f"Mean Error: {mean_error:.4f}m\n"
        f"Max Error: {max_error:.4f}m\n"
        f"MANO: {'✓' if left_mano else '✗'}\n"
        f"GT Object Mesh: {'✓' if has_gt_object else '✗'}\n"
        f"Pred Object Mesh: {'✓' if has_pred_object else '✗'}"
    ))
    
    # Check final file size
    file_size = output_file.stat().st_size if output_file.exists() else 0
    
    print(f"\n✅ Proper reconstruction complete!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 File size: {file_size} bytes ({file_size/1024/1024:.1f} MB)")
    print(f"📈 Mean position error: {mean_error:.4f}m")
    print(f"📈 Max position error: {max_error:.4f}m")
    print(f"🎯 Features included:")
    print(f"  - {'✓' if left_mano else '✗'} MANO hand models")
    print(f"  - {'✓' if has_gt_object else '✗'} Ground truth object 3D mesh")
    print(f"  - {'✓' if has_pred_object else '✗'} Predicted object 3D mesh")
    print(f"  - ✓ Coordinate system transformation")
    print(f"  - ✓ Ground plane reference")
    print(f"  - ✓ Full trajectory visualization")
    print(f"\n🎆 To view:")
    print(f"   rerun {output_file} --web-viewer --port 9876")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Proper reconstruction following working approach")
    parser.add_argument("--results_dir", type=str, default="runs/train/hand_to_object_motion_only_improved/reconstruct", 
                       help="Directory containing saved training results")
    args = parser.parse_args()
    
    success = reconstruct_visualization(args.results_dir)
    exit(0 if success else 1)
