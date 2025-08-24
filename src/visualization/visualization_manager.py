#!/usr/bin/env python3
"""Visualization manager for hand-to-object diffusion training"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from .rerun_visualizer import RerunVisualizer


class VisualizationManager:
    """Manages all visualization during training with error handling and performance optimization"""
    
    def __init__(self, opt, dataset, enable_visualization: bool = True):
        """Initialize visualization manager
        
        Args:
            opt: Training options containing visualization parameters
            dataset: Training dataset for visualization context
            enable_visualization: Whether to enable visualization (default: True)
        """
        self.opt = opt
        self.dataset = dataset
        self.visualizer: Optional[RerunVisualizer] = None
        self.enabled = enable_visualization and getattr(opt, 'use_rerun', False)
        
        if self.enabled:
            self._setup_visualization()
    
    def _setup_visualization(self) -> None:
        """Setup Rerun visualization with error handling"""
        try:
            self.visualizer = RerunVisualizer(
                exp_name=self.opt.exp_name,
                save_dir=self.opt.save_dir,
                enable_visualization=True,
                mano_models_dir=getattr(self.opt, 'mano_models_dir', 'data/mano_models'),
                hand_articulations_path=getattr(self.opt, 'hand_articulations_path', 'data/hand_articulations.pkl'),
                generation_data_path=getattr(self.opt, 'generation_data_path', 'data/generation.pkl')
            )
            print("✓ Rerun visualization enabled with MANO models and hand articulations")
        except Exception as e:
            print(f"⚠️  Failed to setup visualization: {e}")
            self.enabled = False
            self.visualizer = None
    
    def is_enabled(self) -> bool:
        """Check if visualization is enabled and ready"""
        return self.enabled and self.visualizer is not None
    
    def visualize_training_step(self, step: int, left_hand: torch.Tensor, 
                              right_hand: torch.Tensor, object_motion: torch.Tensor,
                              **kwargs) -> None:
        """Visualize a single training step with error handling
        
        Args:
            step: Current training step
            left_hand: Left hand pose data [1, T, D]
            right_hand: Right hand pose data [1, T, D]
            object_motion: Object motion data [1, T, D]
            **kwargs: Additional visualization parameters
        """
        if not self.is_enabled():
            return
        
        try:
            self.visualizer.visualize_training_frame(
                step=step,
                left_hand=left_hand,
                right_hand=right_hand,
                object_motion_gt=object_motion,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️  Training step visualization failed: {e}")
    
    def visualize_best_model_prediction(self, step: int, left_hand: torch.Tensor,
                                      right_hand: torch.Tensor, object_motion: torch.Tensor,
                                      seq_len: torch.Tensor, is_moving: bool, 
                                      mean_velocity: float, diffusion_model: torch.nn.Module, 
                                      device: torch.device) -> None:
        """Visualize best model prediction during evaluation
        
        Args:
            step: Current training step
            left_hand: Left hand pose data
            right_hand: Right hand pose data
            object_motion: Object motion data
            seq_len: Sequence length tensor
            is_moving: Whether the object is moving
            mean_velocity: Mean velocity of the object
            diffusion_model: The diffusion model for prediction
            device: Device to run inference on
        """
        if not self.is_enabled():
            return
        
        try:
            # Generate prediction from model
            prediction = self._generate_prediction(
                left_hand, right_hand, seq_len, diffusion_model, device
            )
            
            # Visualize with prediction
            self.visualizer.visualize_training_frame(
                step=step,
                left_hand=left_hand,
                right_hand=right_hand,
                object_motion_gt=object_motion,
                object_motion_pred=prediction,
                seq_len=seq_len,
                is_moving=is_moving,
                mean_velocity=mean_velocity
            )
        except Exception as e:
            print(f"⚠️  Best model prediction visualization failed: {e}")
    
    def visualize_evaluation_samples(self, diffusion_model: torch.nn.Module, 
                                  device: torch.device, num_samples: int = 5) -> None:
        """Visualize evaluation samples with error handling
        
        Args:
            diffusion_model: The diffusion model for evaluation
            device: Device to run inference on
            num_samples: Number of evaluation samples to visualize
        """
        if not self.is_enabled():
            return
        
        try:
            print(f"  Visualizing evaluation samples...")
            eval_samples = min(num_samples, len(self.dataset.window_data))
            
            for i in range(eval_samples):
                self._visualize_single_evaluation_sample(i, diffusion_model, device)
            
            print(f"  ✓ Evaluation visualization completed")
        except Exception as e:
            print(f"  ⚠️  Evaluation visualization failed: {e}")
    
    def visualize_final_results(self, sampled_motion_full: torch.Tensor) -> None:
        """Visualize final training results
        
        Args:
            sampled_motion_full: Full sampled motion trajectory
        """
        if not self.is_enabled():
            return
        
        try:
            # Visualize final full trajectory
            self.visualizer.visualize_full_trajectory(
                self.dataset, sampled_motion_full, step_name="final"
            )
            
            # Create enhanced scene visualization
            print(f"\n🎨 Creating enhanced scene visualization...")
            self.visualizer.visualize_enhanced_scene(
                self.dataset,
                sequence_key=self.dataset.demo_id,
                num_frames=getattr(self.opt, 'enhanced_scene_frames', 500)
            )
            print(f"✓ Enhanced scene visualization completed")
        except Exception as e:
            print(f"⚠️  Final results visualization failed: {e}")
    
    def _generate_prediction(self, left_hand: torch.Tensor, right_hand: torch.Tensor,
                           seq_len: torch.Tensor, diffusion_model: torch.nn.Module,
                           device: torch.device) -> torch.Tensor:
        """Generate prediction from diffusion model
        
        Args:
            left_hand: Left hand pose data
            right_hand: Right hand pose data
            seq_len: Sequence length tensor
            diffusion_model: The diffusion model
            device: Device to run inference on
            
        Returns:
            Predicted object motion
        """
        diffusion_model.eval()
        with torch.no_grad():
            # Prepare input
            hand_poses = torch.cat([left_hand, right_hand], dim=-1)
            object_motion_init = torch.zeros_like(left_hand).to(device)
            
            # Generate padding mask
            actual_seq_len = seq_len + 1
            tmp_mask = torch.arange(self.opt.window + 1, device=device).expand(1, self.opt.window + 1) < actual_seq_len[:, None].repeat(1, self.opt.window + 1)
            padding_mask = tmp_mask[:, None, :]
            
            # Sample from model
            sampled_motion = diffusion_model.sample(object_motion_init, hand_poses, padding_mask=padding_mask)
        
        diffusion_model.train()
        return sampled_motion
    
    def _visualize_single_evaluation_sample(self, sample_idx: int, 
                                          diffusion_model: torch.nn.Module,
                                          device: torch.device) -> None:
        """Visualize a single evaluation sample
        
        Args:
            sample_idx: Index of the evaluation sample
            diffusion_model: The diffusion model
            device: Device to run inference on
        """
        eval_data = self.dataset.window_data[sample_idx]
        eval_left = eval_data['left_hand'].to(device)
        eval_right = eval_data['right_hand'].to(device)
        eval_gt = eval_data['object_motion'].to(device)
        eval_seq_len = eval_data['seq_len'].to(device)
        
        # Generate prediction
        eval_pred = self._generate_prediction(
            eval_left, eval_right, eval_seq_len, diffusion_model, device
        )
        
        # Visualize evaluation sample
        self.visualizer.visualize_training_frame(
            f"eval_{sample_idx}",
            eval_left,
            eval_right,
            eval_gt,
            object_motion_pred=eval_pred,
            seq_len=eval_seq_len,
            is_moving=eval_data['is_moving'],
            mean_velocity=eval_data['mean_velocity']
        )
    
    def get_summary(self) -> str:
        """Get visualization summary
        
        Returns:
            Summary string of visualization status and capabilities
        """
        if not self.is_enabled():
            return "Visualization disabled"
        return self.visualizer.get_summary()
    
    def cleanup(self) -> None:
        """Cleanup visualization resources"""
        if self.visualizer:
            try:
                # Any cleanup needed for Rerun
                pass
            except Exception as e:
                print(f"⚠️  Visualization cleanup failed: {e}")
            finally:
                self.visualizer = None
                self.enabled = False
