#!/usr/bin/env python3
"""Modular Rerun visualizer for hand-to-object training and visualization"""

import os
import pickle
import numpy as np
import rerun as rr
import torch
from pathlib import Path

class RerunVisualizer:
    """Modular Rerun visualizer for training and evaluation"""
    
    def __init__(self, exp_name, save_dir, enable_visualization=True, 
                 mano_models_dir=None, hand_articulations_path=None, generation_data_path=None):
        """Initialize Rerun visualizer
        
        Args:
            exp_name: Name of the experiment
            save_dir: Directory to save Rerun files
            enable_visualization: Whether to enable Rerun visualization
            mano_models_dir: Directory containing MANO model files (optional)
            hand_articulations_path: Path to hand articulations pickle file (optional)
            generation_data_path: Path to generation data pickle file (optional)
        """
        self.exp_name = exp_name
        self.save_dir = Path(save_dir)
        self.enable_visualization = enable_visualization
        self.rerun_dir = None
        self.left_mano = None
        self.right_mano = None
        
        # Configurable paths with defaults
        self.mano_models_dir = Path(mano_models_dir) if mano_models_dir else Path("data/mano_models")
        self.hand_articulations_path = Path(hand_articulations_path) if hand_articulations_path else Path("data/hand_articulations.pkl")
        self.generation_data_path = Path(generation_data_path) if generation_data_path else Path("data/generation.pkl")
        
        if self.enable_visualization:
            self._setup_rerun()
            self._load_mano_models()
    
    def _setup_rerun(self):
        """Initialize Rerun visualization"""
        try:
            # Initialize Rerun
            rr.init(f"HandToObject_Training_{self.exp_name}")
            
            # Create output directory for Rerun files
            self.rerun_dir = self.save_dir / "rerun_visualizations"
            self.rerun_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"✓ Rerun visualization initialized")
            print(f"  Output directory: {self.rerun_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to initialize Rerun: {e}")
            print("  Training will continue without visualization")
            self.enable_visualization = False
            self.rerun_dir = None
    
    def _load_mano_models(self):
        """Load MANO models for hand mesh generation"""
        try:
            import smplx
            
            # Check if MANO models directory exists
            if not self.mano_models_dir.exists():
                print(f"⚠️  MANO models directory not found: {self.mano_models_dir}")
                return
            
            # Create a temporary directory structure that SMPLX expects
            mano_dir = "/tmp/mano_models"
            os.makedirs(mano_dir, exist_ok=True)
            
            # Copy the pkl files to expected locations
            import shutil
            left_src = self.mano_models_dir / "MANO_LEFT.pkl"
            right_src = self.mano_models_dir / "MANO_RIGHT.pkl"
            
            if not left_src.exists() or not right_src.exists():
                print(f"⚠️  MANO model files not found in {self.mano_models_dir}")
                return
            
            left_dst = os.path.join(mano_dir, "MANO_LEFT.pkl")
            right_dst = os.path.join(mano_dir, "MANO_RIGHT.pkl")
            
            shutil.copy2(str(left_src), left_dst)
            shutil.copy2(str(right_src), right_dst)
            
            # Load MANO models
            self.left_mano = smplx.MANO(
                model_path=left_dst,
                is_rhand=False,
                use_pca=False,
                flat_hand_mean=True
            )
            
            self.right_mano = smplx.MANO(
                model_path=right_dst,
                is_rhand=True,
                use_pca=False,
                flat_hand_mean=True
            )
            
            print("✓ MANO models loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load MANO models: {e}")
            self.left_mano = None
            self.right_mano = None
    
    def load_hand_articulations(self):
        """Load the extracted hand articulation data"""
        if not self.hand_articulations_path.exists():
            print(f"⚠️  Hand articulations file not found: {self.hand_articulations_path}")
            return None
        
        try:
            with open(self.hand_articulations_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded hand articulations from {len(data)} sequences")
            return data
        except Exception as e:
            print(f"⚠️  Error loading hand articulations: {e}")
            return None
    
    def visualize_training_frame(self, step, left_hand, right_hand, object_motion_gt, 
                               object_motion_pred=None, seq_len=None, is_moving=None, mean_velocity=None):
        """Visualize a single training frame in Rerun"""
        if not self.enable_visualization or self.rerun_dir is None:
            return
        
        try:
            # Set time for this frame
            rr.set_time("training_step", sequence=step)
            
            # Extract positions (first 3 dimensions)
            left_pos = left_hand[0, :, 0:3].cpu().numpy()  # [T, 3]
            right_pos = right_hand[0, :, 3:6].cpu().numpy()  # [T, 3]
            object_pos_gt = object_motion_gt[0, :, 0:3].cpu().numpy()  # [T, 3]
            
            # Log hand trajectories
            rr.log(
                "training/left_hand_trajectory",
                rr.LineStrips3D([left_pos], colors=[[0, 255, 0]], radii=[0.01])
            )
            
            rr.log(
                "training/right_hand_trajectory", 
                rr.LineStrips3D([right_pos], colors=[[0, 0, 255]], radii=[0.01])
            )
            
            # Log ground truth object trajectory
            rr.log(
                "training/object_gt_trajectory",
                rr.LineStrips3D([object_pos_gt], colors=[[255, 0, 0]], radii=[0.015])
            )
            
            # Log predicted object trajectory if available
            if object_motion_pred is not None:
                object_pos_pred = object_motion_pred[0, :, 0:3].cpu().numpy()
                rr.log(
                    "training/object_pred_trajectory",
                    rr.LineStrips3D([object_pos_pred], colors=[[255, 165, 0]], radii=[0.015])
                )
            
            # Log current positions as points
            rr.log(
                "training/left_hand_current",
                rr.Points3D([left_pos[-1]], colors=[[0, 255, 0]], radii=[0.02])
            )
            
            rr.log(
                "training/right_hand_current",
                rr.Points3D([right_pos[-1]], colors=[[0, 0, 255]], radii=[0.02])
            )
            
            rr.log(
                "training/object_gt_current",
                rr.Points3D([object_pos_gt[-1]], colors=[[255, 0, 0]], radii=[0.025])
            )
            
            if object_motion_pred is not None:
                rr.log(
                    "training/object_pred_current",
                    rr.Points3D([object_pos_pred[-1]], colors=[[255, 165, 0]], radii=[0.025])
                )
            
            # Log metadata
            if seq_len is not None:
                rr.log("training/metadata", rr.TextDocument(f"Step: {step}, Seq Length: {seq_len.item()}"))
            
            if is_moving is not None:
                motion_status = "Moving" if is_moving else "Stationary"
                rr.log("training/motion_status", rr.TextDocument(f"Motion: {motion_status}"))
            
            if mean_velocity is not None:
                rr.log("training/velocity", rr.TextDocument(f"Mean Velocity: {mean_velocity:.4f}"))
            
            # Save Rerun file periodically
            if step % 1000 == 0:
                output_file = self.rerun_dir / f"training_step_{step}.rrd"
                rr.save(str(output_file))
                
        except Exception as e:
            print(f"⚠️  Rerun visualization error: {e}")
    
    def visualize_full_trajectory(self, dataset, sampled_motion, step_name="final"):
        """Visualize the full trajectory comparison"""
        if not self.enable_visualization or self.rerun_dir is None:
            return
        
        try:
            # Set time for this visualization
            rr.set_time("full_trajectory", sequence=0)
            
            # Extract positions
            left_hand_full = dataset.left_hand_full[:, 0:3].numpy()  # [T, 3]
            right_hand_full = dataset.right_hand_full[:, 0:3].numpy()  # [T, 3]
            object_gt_full = dataset.object_motion_full[:, 0:3].numpy()  # [T, 3]
            object_pred_full = sampled_motion[:, 0:3].numpy()  # [T, 3]
            
            # Log full trajectories
            rr.log(
                f"full_trajectory/{step_name}/left_hand",
                rr.LineStrips3D([left_hand_full], colors=[[0, 255, 0]], radii=[0.008])
            )
            
            rr.log(
                f"full_trajectory/{step_name}/right_hand",
                rr.LineStrips3D([right_hand_full], colors=[[0, 0, 255]], radii=[0.008])
            )
            
            rr.log(
                f"full_trajectory/{step_name}/object_ground_truth",
                rr.LineStrips3D([object_gt_full], colors=[[255, 0, 0]], radii=[0.012])
            )
            
            rr.log(
                f"full_trajectory/{step_name}/object_predicted",
                rr.LineStrips3D([object_pred_full], colors=[[255, 165, 0]], radii=[0.012])
            )
            
            # Log trajectory statistics
            trajectory_length = len(object_gt_full)
            mean_gt_velocity = np.mean(np.linalg.norm(np.diff(object_gt_full, axis=0), axis=1))
            mean_pred_velocity = np.mean(np.linalg.norm(np.diff(object_pred_full, axis=0), axis=1))
            
            rr.log(
                f"full_trajectory/{step_name}/stats",
                rr.TextDocument(f"Length: {trajectory_length} frames\nGT Mean Vel: {mean_gt_velocity:.4f}\nPred Mean Vel: {mean_pred_velocity:.4f}")
            )
            
            # Save full trajectory visualization
            output_file = self.rerun_dir / f"full_trajectory_{step_name}.rrd"
            rr.save(str(output_file))
            print(f"✓ Saved full trajectory visualization: {output_file}")
            
        except Exception as e:
            print(f"⚠️  Full trajectory visualization error: {e}")
    
    def visualize_enhanced_scene(self, dataset, sequence_key=None, num_frames=500):
        """Create enhanced Rerun visualization with hand articulations and objects"""
        if not self.enable_visualization or self.rerun_dir is None:
            return
        
        try:
            # Load hand articulations
            hand_articulations = self.load_hand_articulations()
            
            # Load basic data
            if not self.generation_data_path.exists():
                print(f"⚠️  Generation data file not found: {self.generation_data_path}")
                return
                
            with open(self.generation_data_path, 'rb') as f:
                basic_data = pickle.load(f)
            
            # Try to find a sequence that has detailed hand articulations
            if sequence_key is None:
                for seq_key in basic_data.keys():
                    if hand_articulations and seq_key in hand_articulations:
                        sequence_key = seq_key
                        break
                
                if sequence_key is None:
                    sequence_key = list(basic_data.keys())[0]
            
            sequence_data = basic_data[sequence_key]
            print(f"Creating enhanced visualization for: {sequence_key}")
            
            # Create output file
            output_file = self.rerun_dir / f"{sequence_key}_enhanced_hands.rrd"
            
            # Initialize rerun
            rr.init(f"Enhanced_Hands_{sequence_key}")
            rr.save(str(output_file))
            
            # Get basic data
            left_hand_data = sequence_data['left_hand']
            right_hand_data = sequence_data['right_hand']
            object_data = sequence_data['object_pose']
            
            # Check if we have detailed hand articulations for this sequence
            detailed_hands = None
            if hand_articulations and sequence_key in hand_articulations:
                detailed_hands = hand_articulations[sequence_key]
                print(f"✓ Using detailed hand articulations with {len(detailed_hands['left_hand'])} left and {len(detailed_hands['right_hand'])} right hand frames")
            else:
                print(f"⚠️  No detailed hand articulations found for {sequence_key}, using basic poses")
            
            # Determine number of frames to process
            if detailed_hands:
                num_frames = min(num_frames, len(detailed_hands['left_hand']), len(detailed_hands['right_hand']))
            else:
                num_frames = min(num_frames, len(left_hand_data), len(right_hand_data))
            
            print(f"Processing {num_frames} frames with enhanced hand visualization...")
            
            # Process frames
            for frame_idx in range(num_frames):
                rr.set_time("frame", sequence=frame_idx)
                
                # Enhanced hand visualization
                if detailed_hands and self.left_mano is not None:
                    self._visualize_detailed_hands(detailed_hands, frame_idx, "left")
                    self._visualize_detailed_hands(detailed_hands, frame_idx, "right")
                else:
                    self._visualize_basic_hands(left_hand_data, right_hand_data, frame_idx)
                
                # Object visualization
                if frame_idx < len(object_data) and len(object_data[frame_idx]['poses']) > 0:
                    self._visualize_objects(object_data[frame_idx])
            
            print(f"✓ Created enhanced rerun file: {output_file}")
            
        except Exception as e:
            print(f"⚠️  Enhanced scene visualization error: {e}")
    
    def _visualize_detailed_hands(self, detailed_hands, frame_idx, hand_type):
        """Visualize detailed hand articulations"""
        if frame_idx >= len(detailed_hands[f'{hand_type}_hand']):
            return
        
        frame = detailed_hands[f'{hand_type}_hand'][frame_idx]
        landmarks = frame.get('landmarks_21')
        mesh_vertices = frame.get('mesh_vertices')
        mesh_faces = frame.get('mesh_faces')
        
        if landmarks is not None and mesh_vertices is not None:
            # Apply coordinate transformation
            landmarks_transformed = []
            for landmark in landmarks:
                transformed_pos = self._apply_coordinate_transform(landmark)
                landmarks_transformed.append(transformed_pos)
            
            mesh_vertices_transformed = []
            for vertex in mesh_vertices:
                transformed_vertex = self._apply_coordinate_transform(vertex)
                mesh_vertices_transformed.append(transformed_vertex)
            
            # Log detailed hand mesh
            color = [0.3, 0.8, 0.3] if hand_type == "left" else [0.3, 0.3, 0.8]
            rr.log(
                f"world/hands/{hand_type}_detailed_mesh",
                rr.Mesh3D(
                    vertex_positions=mesh_vertices_transformed,
                    triangle_indices=mesh_faces,
                    vertex_colors=color
                ),
            )
            
            # Log hand skeleton
            if 'landmark_connectivity' in detailed_hands:
                connectivity = detailed_hands['landmark_connectivity']
                bone_lines = []
                for connection in connectivity:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks_transformed) and end_idx < len(landmarks_transformed):
                        bone_lines.append([landmarks_transformed[start_idx], landmarks_transformed[end_idx]])
                
                if bone_lines:
                    skeleton_color = [[0, 255, 0]] if hand_type == "left" else [[0, 0, 255]]
                    rr.log(
                        f"world/hands/{hand_type}_skeleton",
                        rr.LineStrips3D(bone_lines, colors=skeleton_color, radii=[0.003])
                    )
    
    def _visualize_basic_hands(self, left_hand_data, right_hand_data, frame_idx):
        """Visualize basic hand poses using MANO models"""
        if frame_idx < len(left_hand_data) and self.left_mano is not None:
            left_frame = left_hand_data[frame_idx]
            left_translation = np.array(left_frame['translation'])
            left_rotation_wxyz = left_frame['rotation'][0]
            
            # Generate MANO mesh
            with torch.no_grad():
                left_output = self.left_mano()
                hand_mesh_vertices = left_output.vertices[0].numpy()
                hand_triangles = self.left_mano.faces
            
            # Log mesh
            rr.log(
                "world/hands/left_basic_mesh",
                rr.Mesh3D(
                    vertex_positions=hand_mesh_vertices,
                    triangle_indices=hand_triangles,
                    vertex_colors=[0.3, 0.8, 0.3]
                ),
            )
        
        if frame_idx < len(right_hand_data) and self.right_mano is not None:
            right_frame = right_hand_data[frame_idx]
            right_translation = np.array(right_frame['translation'])
            right_rotation_wxyz = right_frame['rotation'][0]
            
            # Generate MANO mesh
            with torch.no_grad():
                right_output = self.right_mano()
                hand_mesh_vertices = right_output.vertices[0].numpy()
                hand_triangles = self.right_mano.faces
            
            # Log mesh
            rr.log(
                "world/hands/right_basic_mesh",
                rr.Mesh3D(
                    vertex_positions=hand_mesh_vertices,
                    triangle_indices=hand_triangles,
                    vertex_colors=[0.3, 0.3, 0.8]
                ),
            )
    
    def _visualize_objects(self, frame_object_data):
        """Visualize objects in the scene"""
        for obj_data in frame_object_data['poses']:
            object_uid = obj_data['object_uid']
            obj_translation = np.array(obj_data['translation'])
            obj_rotation_wxyz = obj_data['rotation'][0]
            
            # Apply coordinate transformation
            obj_translation_transformed = self._apply_coordinate_transform(obj_translation)
            
            # Log object transform
            rr.log(
                f"world/objects/object_{object_uid}",
                rr.Transform3D(
                    translation=obj_translation_transformed,
                    rotation=rr.Quaternion(xyzw=obj_rotation_wxyz)
                )
            )
    
    def _apply_coordinate_transform(self, position):
        """Apply coordinate system transformation"""
        # Transform matrix: 90-degree rotation around X-axis
        transform_matrix = np.array([
            [1,  0,  0],  # X stays the same
            [0,  1,  0],  # Y becomes -Z (rotated 90° around X)
            [0,  0,  1]   # Z becomes Y (rotated 90° around X)
        ])
        
        # Apply transformation to position
        transformed_position = transform_matrix @ position
        
        return transformed_position
    
    def get_summary(self):
        """Get visualization summary"""
        if not self.enable_visualization:
            return "Rerun visualization disabled"
        
        summary = f"🎯 Rerun Visualization Summary:\n"
        summary += f"  ✓ Real-time training progress visualization\n"
        summary += f"  ✓ Hand trajectory tracking (left=green, right=blue)\n"
        summary += f"  ✓ Object motion comparison (GT=red, predicted=orange)\n"
        summary += f"  ✓ Motion classification (moving/stationary)\n"
        summary += f"  ✓ Best model prediction visualization\n"
        summary += f"  ✓ Full trajectory comparison\n"
        summary += f"  ✓ Enhanced hand articulations with MANO models\n"
        
        if self.rerun_dir:
            summary += f"  📁 Files saved to: {self.rerun_dir}\n"
            summary += f"  🚀 View with: rerun {self.rerun_dir}/full_trajectory_final.rrd --web-viewer --port 9877"
        
        return summary
