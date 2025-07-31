import argparse
import os
import numpy as np
import yaml
import random
import json 

import trimesh 

from tqdm import tqdm
from pathlib import Path

import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

# Import the new dataset
from manip.data.hand_to_object_dataset import HandToObjectDataset

from manip.model.transformer_hand_to_object_diffusion_model import CondGaussianDiffusion 

from matplotlib import pyplot as plt

from manip.vis.trajectory_to_mesh_visualizer import TrajectoryMeshVisualizer

# Add learning rate scheduler import
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=10000,
        results_folder='./results',
        use_wandb=True,  
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-4)

        # Add learning rate scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=train_num_steps, eta_min=1e-6)
        
        # Alternative: ReduceLROnPlateau scheduler
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6)

        self.step = 0
        
        # Add warmup period
        self.warmup_steps = 1000
        self.warmup_lr = train_lr * 0.1  # Start with 10% of target LR

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        self.window = opt.window
        self.use_velocity = getattr(opt, 'use_velocity', False)
        
        self.data_root_folder = self.opt.data_root_folder 

        self.prep_dataloader(window_size=opt.window)

        self.test_on_train = getattr(self.opt, 'test_sample_res_on_train', False)
        self.for_quant_eval = getattr(self.opt, 'for_quant_eval', False)

        # Add gradient clipping
        self.max_grad_norm = getattr(opt, 'max_grad_norm', 1.0)
        
        # Add loss tracking for better monitoring
        self.loss_history = []
        self.val_loss_history = []

    def check_model_nan(self):
        """Check if any model parameters are NaN."""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f'WARNING: NaN detected in model parameter {name}')
                return True
        return False

    def prep_dataloader(self, window_size):
        # Define dataset using the new HandToObjectDataset
        train_dataset = HandToObjectDataset(
            data_path=self.data_root_folder,
            window_size=window_size,
            use_velocity=self.use_velocity,
            sampling_strategy=self.opt.sampling_strategy,
            motion_threshold=self.opt.motion_threshold,
            min_motion_frames=self.opt.min_motion_frames,
            augment=True  # Enable data augmentation for training
        )
        
        # Create validation dataset (using different sampling or subset)
        val_dataset = HandToObjectDataset(
            data_path=self.data_root_folder,
            window_size=window_size,
            use_velocity=self.use_velocity,
            sampling_strategy=self.opt.sampling_strategy,  # Use same strategy for validation
            motion_threshold=self.opt.motion_threshold,
            min_motion_frames=self.opt.min_motion_frames,
            augment=False  # No augmentation for validation
        )

        self.ds = train_dataset 
        self.val_ds = val_dataset
        
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))

        print(f"Training dataset size: {len(self.ds)}")
        print(f"Validation dataset size: {len(self.val_ds)}")
        print(f"Data dimensions: {self.ds.pose_dim}D")
        print(f"Using velocity: {self.use_velocity}")

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            accumulated_loss = 0.0
            accumulated_grad_norm = 0.0
            
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                
                # Extract data from the new dataset format
                condition = data_dict['condition'].cuda()  # [BS, T, 2*D] - left + right hand
                target = data_dict['target'].cuda()        # [BS, T, D] - object trajectory
                
                bs, num_steps, _ = target.shape

                # Generate padding mask - using fixed window size for now
                # In the new dataset, all windows are the same size
                seq_len = torch.full((bs,), num_steps, dtype=torch.long, device=target.device)
                actual_seq_len = seq_len + 1  # Add 1 for noise level timestep
                tmp_mask = torch.arange(self.window+1, device=target.device).expand(bs, self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                padding_mask = tmp_mask[:, None, :]

                with autocast(enabled = self.amp):    
                    loss_diffusion = self.model(target, condition, padding_mask=padding_mask)
                    
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        print(f'  Target range: [{target.min():.6f}, {target.max():.6f}]')
                        print(f'  Condition range: [{condition.min():.6f}, {condition.max():.6f}]')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    # Check for inf loss as well
                    if torch.isinf(loss).item():
                        print('WARNING: Inf loss. Skipping to next data...')
                        print(f'  Target range: [{target.min():.6f}, {target.max():.6f}]')
                        print(f'  Condition range: [{condition.min():.6f}, {condition.max():.6f}]')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    accumulated_loss += loss.item()

            # Check gradients after accumulation and compute gradient norm
            if not nan_exists:
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                if parameters:
                    # Check for NaN or inf gradients BEFORE unscaling
                    has_nan_grad = False
                    for param in parameters:
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print('WARNING: NaN or Inf gradients detected. Skipping step...')
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        nan_exists = True
                        torch.cuda.empty_cache()
                        # Reset gradients and continue
                        self.optimizer.zero_grad()
                        continue
                    
                    # Now safe to unscale
                    self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradient values to prevent extreme values
                    for param in parameters:
                        if param.grad is not None:
                            param.grad.clamp_(-10.0, 10.0)  # Clip gradient values
                    
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(target.device) for p in parameters]), 2.0)
                    
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients after accumulation. Skipping step...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        # Reset gradients and continue
                        self.optimizer.zero_grad()
                        continue
                    else:
                        accumulated_grad_norm = total_norm.item()

            if nan_exists:
                # If we had NaN in loss, skip the optimizer step entirely
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Step the learning rate scheduler
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            # For ReduceLROnPlateau, we'd step with validation loss later
            
            # Apply warmup
            if self.step < self.warmup_steps:
                warmup_factor = min(1.0, self.step / self.warmup_steps)
                target_lr = self.optimizer.param_groups[0]['lr']  # Get the target LR from scheduler
                current_lr = self.warmup_lr + (target_lr - self.warmup_lr) * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            self.ema.update()

            # Track loss history
            current_loss = accumulated_loss / self.gradient_accumulate_every
            self.loss_history.append(current_loss)

            # Log training metrics to wandb
            if self.use_wandb:
                log_dict = {
                    "Train/Loss/Total": current_loss,
                    "Train/Loss/Diffusion": current_loss,
                    "Train/Gradients/Norm": accumulated_grad_norm,
                    "Train/Learning_Rate": self.optimizer.param_groups[0]['lr'],
                    "Train/Step": self.step,
                    "Train/EMA_Decay": self.ema.beta,
                }
                
                # Add loss history statistics
                if len(self.loss_history) > 100:
                    recent_losses = self.loss_history[-100:]
                    log_dict["Train/Loss/Recent_Mean"] = np.mean(recent_losses)
                    log_dict["Train/Loss/Recent_Std"] = np.std(recent_losses)
                    log_dict["Train/Loss/Recent_Min"] = np.min(recent_losses)
                
                # Add memory usage if CUDA is available
                if torch.cuda.is_available():
                    log_dict["Train/GPU_Memory_Allocated_GB"] = torch.cuda.memory_allocated() / 1024**3
                    log_dict["Train/GPU_Memory_Reserved_GB"] = torch.cuda.memory_reserved() / 1024**3
                
                # Add gradient norms for different parameter groups
                param_groups = {
                    'time_mlp': [],
                    'linear_out': [],
                    'motion_transformer': [],
                    'other': []
                }
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if 'time_mlp' in name:
                            param_groups['time_mlp'].append(grad_norm)
                        elif 'linear_out' in name:
                            param_groups['linear_out'].append(grad_norm)
                        elif 'motion_transformer' in name:
                            param_groups['motion_transformer'].append(grad_norm)
                        else:
                            param_groups['other'].append(grad_norm)
                
                # Log average gradient norms for each component
                for group_name, norms in param_groups.items():
                    if norms:
                        log_dict[f"Train/Gradients/{group_name}_avg"] = sum(norms) / len(norms)
                        log_dict[f"Train/Gradients/{group_name}_max"] = max(norms)
                        log_dict[f"Train/Gradients/{group_name}_min"] = min(norms)
                
                # Add scaler scale for AMP monitoring
                if self.amp:
                    log_dict["Train/Scaler_Scale"] = self.scaler.get_scale()
                
                wandb.log(log_dict, step=self.step)

            # Print progress occasionally 
            if self.step % 100 == 0:
                print(f"Step {self.step}: Loss={current_loss:.6f}, Grad_Norm={accumulated_grad_norm:.4f}, LR={self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Check for NaN in model parameters
                if self.check_model_nan():
                    print("WARNING: NaN detected in model parameters. Stopping training.")
                    break

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_condition = val_data_dict['condition'].cuda()
                    val_target = val_data_dict['target'].cuda()

                    bs, num_steps, _ = val_target.shape

                    # Generate padding mask for validation
                    seq_len = torch.full((bs,), num_steps, dtype=torch.long, device=val_target.device)
                    actual_seq_len = seq_len + 1
                    tmp_mask = torch.arange(self.window+1, device=val_target.device).expand(bs, self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                    padding_mask = tmp_mask[:, None, :]

                    # Get validation loss 
                    val_loss_diffusion = self.model(val_target, val_condition, padding_mask=padding_mask)
                    val_loss = val_loss_diffusion 
                    
                    # Track validation loss history
                    self.val_loss_history.append(val_loss.item())
                    
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total": val_loss.item(),
                            "Validation/Loss/Diffusion": val_loss_diffusion.item(),
                        }
                        
                        # Add validation loss statistics
                        if len(self.val_loss_history) > 50:
                            recent_val_losses = self.val_loss_history[-50:]
                            val_log_dict["Validation/Loss/Recent_Mean"] = np.mean(recent_val_losses)
                            val_log_dict["Validation/Loss/Recent_Std"] = np.std(recent_val_losses)
                        
                        wandb.log(val_log_dict, step=self.step)
                    
                    # Step ReduceLROnPlateau scheduler if using it
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss.item())

                    milestone = self.step // self.save_and_sample_every
            
                    bs_for_vis = 1

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        # Sample from the model
                        sampled_trajectories = self.ema.ema_model.sample(
                            val_target[:bs_for_vis], 
                            val_condition[:bs_for_vis], 
                            padding_mask=padding_mask[:bs_for_vis]
                        )

                        self.gen_vis_res(sampled_trajectories, val_data_dict, self.step, vis_tag="pred_object")

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()

        num_sample = 50
        
        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.test_on_train:
                    val_data_dict = next(self.dl)
                else:
                    val_data_dict = next(self.val_dl)
                    
                condition = val_data_dict['condition'].cuda()
                target = val_data_dict['target'].cuda()

                bs, num_steps, _ = target.shape

                # Generate padding mask 
                seq_len = torch.full((bs,), num_steps, dtype=torch.long, device=target.device)
                actual_seq_len = seq_len + 1
                tmp_mask = torch.arange(self.window+1, device=target.device).expand(bs, self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                padding_mask = tmp_mask[:, None, :]

                max_num = 1

                sampled_trajectories = self.ema.ema_model.sample(
                    target[:max_num], 
                    condition[:max_num], 
                    padding_mask=padding_mask[:max_num]
                )

                vis_tag = str(milestone)+"_sample_"+str(s_idx)

                if self.test_on_train:
                    vis_tag = vis_tag + "_on_train"
                
                self.gen_vis_res(sampled_trajectories, val_data_dict, milestone, vis_tag=vis_tag)

    def gen_vis_res(self, sampled_trajectories, data_dict, step, vis_gt=False, vis_tag=None):
        """
        Generate visualization results for sampled object trajectories.
        
        Args:
            sampled_trajectories: [BS, T, D] - sampled object trajectories 
            data_dict: dictionary containing ground truth data
            step: current training step
            vis_gt: whether to visualize ground truth
            vis_tag: tag for saving files
        """
        
        # Get ground truth and other info
        demo_ids = data_dict['demo_id']
        object_ids = data_dict['object_id']
        target_raw = data_dict['target_raw']  # Unnormalized ground truth
        condition = data_dict['condition']    # Hand trajectories
        
        num_seq = sampled_trajectories.shape[0]
        
        # Denormalize the sampled trajectories
        sampled_trajectories_denorm = torch.zeros_like(sampled_trajectories)
        for i in range(num_seq):
            sampled_trajectories_denorm[i] = torch.tensor(
                self.ds.denormalize_data(sampled_trajectories[i].cpu().numpy(), 'object'),
                dtype=torch.float32,
                device=sampled_trajectories.device
            )
        
        # Compute trajectory errors for logging
        position_errors = []
        rotation_errors = []
        
        for i in range(num_seq):
            # Position error (L2 distance per frame)
            pred_pos = sampled_trajectories_denorm[i, :, :3]  # [T, 3]
            gt_pos = target_raw[i, :, :3].to(pred_pos.device)  # [T, 3] - ensure same device
            pos_error = torch.norm(pred_pos - gt_pos, dim=1).mean().item()
            position_errors.append(pos_error)
            
            # Rotation error if available
            if self.ds.pose_dim >= 9:
                if self.ds.use_velocity:
                    # 12D format: pos(3) + vel(3) + rot(6)
                    pred_rot = sampled_trajectories_denorm[i, :, 6:12]
                    gt_rot = target_raw[i, :, 6:12].to(pred_rot.device)
                else:
                    # 9D format: pos(3) + rot(6)
                    pred_rot = sampled_trajectories_denorm[i, :, 3:9]
                    gt_rot = target_raw[i, :, 3:9].to(pred_rot.device)
                
                # Convert 6D rotation to matrices and compute angular error
                pred_rot_mat = transforms.rotation_6d_to_matrix(pred_rot.reshape(-1, 6))
                gt_rot_mat = transforms.rotation_6d_to_matrix(gt_rot.reshape(-1, 6))
                
                # Compute relative rotation and extract angle
                relative_rot = torch.matmul(pred_rot_mat, gt_rot_mat.transpose(-1, -2))
                trace = relative_rot.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
                cos_angle = (trace - 1) / 2
                cos_angle = torch.clamp(cos_angle, -1, 1)
                angle_error = torch.acos(cos_angle).mean().item()
                rotation_errors.append(angle_error)
        
        # Log trajectory errors to wandb
        if self.use_wandb and position_errors:
            error_log_dict = {
                f"Sampling/Position_Error_Mean": np.mean(position_errors),
                f"Sampling/Position_Error_Std": np.std(position_errors),
            }
            if rotation_errors:
                error_log_dict[f"Sampling/Rotation_Error_Mean_Rad"] = np.mean(rotation_errors)
                error_log_dict[f"Sampling/Rotation_Error_Mean_Deg"] = np.degrees(np.mean(rotation_errors))
                error_log_dict[f"Sampling/Rotation_Error_Std_Rad"] = np.std(rotation_errors)
            
            wandb.log(error_log_dict, step=step)
        
        # Save numerical results
        if vis_tag is None:
            vis_tag = f"step_{step}"
            
        save_dir = os.path.join(self.vis_folder, vis_tag)
        os.makedirs(save_dir, exist_ok=True)
        
        for seq_idx in range(num_seq):
            demo_id = demo_ids[seq_idx] if isinstance(demo_ids, (list, tuple)) else demo_ids
            object_id = object_ids[seq_idx] if isinstance(object_ids, (list, tuple)) else object_ids
            
            # Save sampled trajectory
            sampled_path = os.path.join(save_dir, f"sampled_{demo_id}_{object_id}_{seq_idx}.npy")
            np.save(sampled_path, sampled_trajectories_denorm[seq_idx].cpu().numpy())
            
            # Save ground truth trajectory  
            gt_path = os.path.join(save_dir, f"gt_{demo_id}_{object_id}_{seq_idx}.npy")
            np.save(gt_path, target_raw[seq_idx].cpu().numpy())
            
            # Save input hand trajectories
            condition_denorm = torch.zeros_like(condition[seq_idx])
            # Denormalize left and right hand separately
            left_hand = condition[seq_idx, :, :self.ds.pose_dim].cpu().numpy()
            right_hand = condition[seq_idx, :, self.ds.pose_dim:].cpu().numpy()
            
            left_hand_denorm = self.ds.denormalize_data(left_hand, 'left_hand')
            right_hand_denorm = self.ds.denormalize_data(right_hand, 'right_hand')
            
            hands_path = os.path.join(save_dir, f"hands_{demo_id}_{object_id}_{seq_idx}.npz")
            np.savez(hands_path, 
                    left_hand=left_hand_denorm,
                    right_hand=right_hand_denorm)
            
            print(f"Saved visualization results for {demo_id}_{object_id} to {save_dir}")

        # Generate mesh visualizations if not in quantitative evaluation mode
        if not self.for_quant_eval:
            self.generate_mesh_visualizations(save_dir, vis_tag)

    def generate_mesh_visualizations(self, save_dir: str, vis_tag: str):
        """
        Generate mesh visualizations using the TrajectoryMeshVisualizer.
        
        Args:
            save_dir: Directory containing saved trajectory data
            vis_tag: Tag for this visualization
        """
        try:
            # Initialize the mesh visualizer
            visualizer = TrajectoryMeshVisualizer(
                data_root_folder=self.data_root_folder,
                object_geometry_path=getattr(self.ds, 'obj_geo_root_folder', None)
            )
            
            print(f"Generating mesh visualizations for {vis_tag}...")
            
            # Generate mesh visualization from saved results
            video_path = visualizer.visualize_from_trainer_results(
                results_dir=save_dir,
                sequence_name=vis_tag,
                render_video=True
            )
            
            print(f"Mesh visualization complete. Video saved to: {video_path}")
            
            # Log video path to wandb if available
            if self.use_wandb:
                wandb.log({f"Visualization/Video_Path": video_path}, step=int(vis_tag.split('_')[-1]) if 'step_' in vis_tag else 0)
                
        except Exception as e:
            print(f"Warning: Mesh visualization failed: {e}")
            print("Continuing without mesh visualization...")

    def evaluate_model(self, num_eval_samples=100):
        """
        Evaluate the model by computing position and rotation errors.
        """
        self.ema.ema_model.eval()
        
        position_errors = []
        rotation_errors = []
        
        with torch.no_grad():
            for eval_idx in range(num_eval_samples):
                if self.test_on_train:
                    data_dict = next(self.dl)
                else:
                    data_dict = next(self.val_dl)
                    
                condition = data_dict['condition'].cuda()
                target = data_dict['target'].cuda()
                target_raw = data_dict['target_raw'].cuda()
                
                bs, num_steps, _ = target.shape
                
                # Generate padding mask
                seq_len = torch.full((bs,), num_steps, dtype=torch.long, device=target.device)
                actual_seq_len = seq_len + 1
                tmp_mask = torch.arange(self.window+1, device=target.device).expand(bs, self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
                padding_mask = tmp_mask[:, None, :]
                
                # Sample from model
                sampled_trajectories = self.ema.ema_model.sample(
                    target, condition, padding_mask=padding_mask
                )
                
                # Denormalize for evaluation
                for i in range(bs):
                    sampled_denorm = torch.tensor(
                        self.ds.denormalize_data(sampled_trajectories[i].cpu().numpy(), 'object'),
                        dtype=torch.float32, device=target.device
                    )
                    
                    # Extract positions (first 3 dimensions)
                    gt_pos = target_raw[i, :, :3]
                    pred_pos = sampled_denorm[:, :3].to(gt_pos.device)
                    
                    # Position error (L2 distance per frame)
                    pos_error = torch.norm(pred_pos - gt_pos, dim=1).mean().item()
                    position_errors.append(pos_error)
                    
                    # Rotation error (if using rotation data)
                    if self.ds.pose_dim >= 9:  # Has rotation data
                        if self.use_velocity:
                            # 12D format: pos(3) + vel(3) + rot(6)
                            gt_rot = target_raw[i, :, 6:12]
                            pred_rot = sampled_denorm[:, 6:12].to(gt_rot.device)
                        else:
                            # 9D format: pos(3) + rot(6)
                            gt_rot = target_raw[i, :, 3:9]
                            pred_rot = sampled_denorm[:, 3:9].to(gt_rot.device)
                        
                        # Convert 6D rotation to matrices and compute angular error
                        gt_rot_mat = transforms.rotation_6d_to_matrix(gt_rot.reshape(-1, 6))
                        pred_rot_mat = transforms.rotation_6d_to_matrix(pred_rot.reshape(-1, 6))
                        
                        # Compute relative rotation and extract angle
                        relative_rot = torch.matmul(pred_rot_mat, gt_rot_mat.transpose(-1, -2))
                        trace = relative_rot.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
                        cos_angle = (trace - 1) / 2
                        cos_angle = torch.clamp(cos_angle, -1, 1)
                        angle_error = torch.acos(cos_angle).mean().item()
                        rotation_errors.append(angle_error)
        
        mean_pos_error = np.mean(position_errors)
        mean_rot_error = np.mean(rotation_errors) if rotation_errors else 0.0
        
        print(f"Evaluation Results:")
        print(f"  Mean Position Error: {mean_pos_error:.4f}m")
        print(f"  Mean Rotation Error: {mean_rot_error:.4f} rad ({np.degrees(mean_rot_error):.2f}°)")
        
        return mean_pos_error, mean_rot_error

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model dimensions based on data format
    pose_dim = 12 if opt.use_velocity else 9
    repr_dim = pose_dim  # Output dimension (object trajectory)
    input_dim = pose_dim * 2  # Input dimension (left + right hand)
    
    # Use L2 loss for better stability with pred_x0 objective
    loss_type = "l2"  # Changed from "l1" to "l2"
  
    diffusion_model = CondGaussianDiffusion(
        opt, 
        d_feats=repr_dim, 
        d_model=opt.d_model,
        n_dec_layers=opt.n_dec_layers, 
        n_head=opt.n_head, 
        d_k=opt.d_k, 
        d_v=opt.d_v,
        max_timesteps=opt.window+1, 
        out_dim=repr_dim, 
        timesteps=1000,
        objective="pred_x0", 
        loss_type=loss_type,
        batch_size=opt.batch_size,
        # Add P2 loss weight for better timestep weighting
        p2_loss_weight_gamma=1.0,  # Recommended value from DDPM paper
        p2_loss_weight_k=1,
    )
   
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=opt.train_steps,  # Use configurable training steps
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
    )

    trainer.train()

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model dimensions
    pose_dim = 12 if opt.use_velocity else 9
    repr_dim = pose_dim
    
    loss_type = "l1"
    
    diffusion_model = CondGaussianDiffusion(
        opt, 
        d_feats=repr_dim, 
        d_model=opt.d_model,
        n_dec_layers=opt.n_dec_layers, 
        n_head=opt.n_head, 
        d_k=opt.d_k, 
        d_v=opt.d_v,
        max_timesteps=opt.window+1, 
        out_dim=repr_dim, 
        timesteps=1000,
        objective="pred_x0", 
        loss_type=loss_type,
        batch_size=opt.batch_size
    )

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
        use_wandb=False 
    )
    
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # Project and logging
    parser.add_argument('--project', default='runs/train', help='output folder for weights and visualizations')
    parser.add_argument('--wandb_pj_name', type=str, default='hand_object_diffusion', help='wandb project name')
    parser.add_argument('--entity', default='egorecon', help='W&B entity')
    parser.add_argument('--exp_name', default='hand_to_object_exp', help='save to project/exp_name')
    parser.add_argument('--device', default='0', help='cuda device')

    # Data parameters
    parser.add_argument('--data_root_folder', default='data/processed_data.pkl', help='path to processed data pickle file')
    parser.add_argument('--window', type=int, default=120, help='window size for trajectories')
    parser.add_argument('--use_velocity', action='store_true', help='use 12D format with velocity (default: 9D)')
    parser.add_argument('--sampling_strategy', default='motion_only', choices=['balanced', 'motion_only', 'random'], 
                       help='sampling strategy: balanced, motion_only, or random')
    parser.add_argument('--motion_threshold', type=float, default=0.005, 
                       help='threshold for motion detection (meters per frame)')
    parser.add_argument('--min_motion_frames', type=int, default=10, 
                       help='minimum consecutive motion frames to consider a window as moving')

    # Model parameters  
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')  # Reduced from 2e-4
    parser.add_argument('--n_dec_layers', type=int, default=4, help='number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='number of attention heads')
    parser.add_argument('--d_k', type=int, default=256, help='key dimension in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='value dimension in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='model dimension in transformer')
    
    # Training control
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint path to resume from')
    parser.add_argument('--train_steps', type=int, default=100000, help='number of training steps')  # Increased from 100k
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    
    # Testing parameters
    parser.add_argument("--test_sample_res", action="store_true", help="test sampling results")
    parser.add_argument("--test_sample_res_on_train", action="store_true", help="test sampling on training data")
    parser.add_argument("--for_quant_eval", action="store_true", help="quantitative evaluation mode")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
