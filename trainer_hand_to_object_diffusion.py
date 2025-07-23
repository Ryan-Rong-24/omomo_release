import argparse
import os
import numpy as np
import yaml
import random
import json
from tqdm import tqdm
from pathlib import Path
import wandb
import pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.cuda.amp import autocast, GradScaler

from ema_pytorch import EMA

from manip.model.transformer_hand_to_object_diffusion_model import CondGaussianDiffusion
from evaluation_metrics import compute_metrics 
from matplotlib import pyplot as plt
from visualize_training_results_video import create_training_video

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

class HandToObjectDataset(Dataset):
    def __init__(self, data_path, window=128, train=True, velocity_threshold=0.02, balance_ratio=0.5, max_oversample_ratio=5.0):
        self.window = window
        self.data_path = data_path
        self.train = train
        self.velocity_threshold = velocity_threshold
        self.balance_ratio = balance_ratio
        self.max_oversample_ratio = max_oversample_ratio

        print(f"Loading data from {data_path}...")
        processed_data = load_pickle(data_path)
        
        all_demo_ids = list(processed_data.keys())
        random.shuffle(all_demo_ids)
        
        split_ratio = 0.8
        if self.train:
            self.demo_ids = all_demo_ids[:int(len(all_demo_ids) * split_ratio)]
        else:
            self.demo_ids = all_demo_ids[int(len(all_demo_ids) * split_ratio):]

        self.moving_windows = []
        self.stationary_windows = []
        self.pose_dim = 9  # Set early
        self._prepare_interaction_windows(processed_data)

        # Get normalization stats
        all_left_hand_data = [d['left_hand'] for d in self.moving_windows + self.stationary_windows]
        all_right_hand_data = [d['right_hand'] for d in self.moving_windows + self.stationary_windows]
        all_object_motion_data = [d['object_motion'] for d in self.moving_windows + self.stationary_windows]

        self.left_hand_mean, self.left_hand_std = self.get_normalization_stats(all_left_hand_data)
        self.right_hand_mean, self.right_hand_std = self.get_normalization_stats(all_right_hand_data)
        self.object_motion_mean, self.object_motion_std = self.get_normalization_stats(all_object_motion_data)

        # Normalize the data
        for i in range(len(self.moving_windows)):
            self.moving_windows[i]['left_hand'] = self.normalize(self.moving_windows[i]['left_hand'], self.left_hand_mean, self.left_hand_std)
            self.moving_windows[i]['right_hand'] = self.normalize(self.moving_windows[i]['right_hand'], self.right_hand_mean, self.right_hand_std)
            self.moving_windows[i]['object_motion'] = self.normalize(self.moving_windows[i]['object_motion'], self.object_motion_mean, self.object_motion_std)

        for i in range(len(self.stationary_windows)):
            self.stationary_windows[i]['left_hand'] = self.normalize(self.stationary_windows[i]['left_hand'], self.left_hand_mean, self.left_hand_std)
            self.stationary_windows[i]['right_hand'] = self.normalize(self.stationary_windows[i]['right_hand'], self.right_hand_mean, self.right_hand_std)
            self.stationary_windows[i]['object_motion'] = self.normalize(self.stationary_windows[i]['object_motion'], self.object_motion_mean, self.object_motion_std)

        self._balance_dataset()

        # For sequential sampling like in overfit script
        self.current_window_idx = 0

    def get_normalization_stats(self, data):
        # data is a list of tensors
        data = torch.cat(data, dim=0)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        # Add a small epsilon to std to avoid division by zero
        std[std == 0] = 1e-6
        return mean, std

    def normalize(self, data, mean, std):
        return (data - mean) / std

    def denormalize(self, data, mean, std):
        return data * std + mean

    def _prepare_interaction_windows(self, processed_data):
        for demo_id in self.demo_ids:
            demo_data = processed_data[demo_id]
            objects_data = demo_data['objects']
            
            for obj_id in objects_data.keys():
                left_hand_data = demo_data['left_hand']['poses_9d']
                right_hand_data = demo_data['right_hand']['poses_9d']
                object_data = objects_data[obj_id]['poses_9d']

                min_length = min(len(left_hand_data), len(right_hand_data), len(object_data))

                object_positions = object_data[:min_length, :3]
                velocities = np.linalg.norm(np.diff(object_positions, axis=0), axis=1)
                in_motion = velocities > self.velocity_threshold

                # Find contiguous blocks of motion
                motion_blocks = []
                stationary_blocks = []
                
                # Track motion state
                current_state = in_motion[0] if len(in_motion) > 0 else False
                start_idx = 0
                
                for i in range(1, len(in_motion)):
                    if in_motion[i] != current_state:
                        if current_state:
                            motion_blocks.append((start_idx, i))
                        else:
                            stationary_blocks.append((start_idx, i))
                        start_idx = i
                        current_state = in_motion[i]
                
                # Handle the last block
                if current_state:
                    motion_blocks.append((start_idx, len(in_motion)))
                else:
                    stationary_blocks.append((start_idx, len(in_motion)))

                # Process moving windows
                for start, end in motion_blocks:
                    self._extract_windows_from_segment(
                        left_hand_data, right_hand_data, object_data, 
                        start, end, demo_id, obj_id, is_moving=True
                    )
                
                # Process stationary windows
                for start, end in stationary_blocks:
                    self._extract_windows_from_segment(
                        left_hand_data, right_hand_data, object_data, 
                        start, end, demo_id, obj_id, is_moving=False
                    )

    def _extract_windows_from_segment(self, left_hand_data, right_hand_data, object_data, 
                                     start, end, demo_id, obj_id, is_moving=True):
        """Extract windows from a segment, handling both short and long segments."""
        actual_len = end - start
        
        # Skip segments that are too short
        if actual_len < 30:
            return
        
        # For segments shorter than window size, create one window
        if actual_len <= self.window:
            window_dict = self._create_window(
                left_hand_data[start:end], 
                right_hand_data[start:end], 
                object_data[start:end], 
                actual_len, demo_id, obj_id, is_moving
            )
            
            if is_moving:
                self.moving_windows.append(window_dict)
            else:
                self.stationary_windows.append(window_dict)
        else:
            # For long segments, create multiple overlapping windows
            step_size = self.window // 2  # 50% overlap
            
            for window_start in range(0, actual_len - 30 + 1, step_size):
                window_end = min(window_start + self.window, actual_len)
                window_len = window_end - window_start
                
                if window_len < 30:
                    break
                
                segment_start = start + window_start
                segment_end = start + window_end
                
                window_dict = self._create_window(
                    left_hand_data[segment_start:segment_end], 
                    right_hand_data[segment_start:segment_end], 
                    object_data[segment_start:segment_end], 
                    window_len, demo_id, obj_id, is_moving
                )
                
                if is_moving:
                    self.moving_windows.append(window_dict)
                else:
                    self.stationary_windows.append(window_dict)
    
    def _create_window(self, left_hand, right_hand, object_motion, actual_len, demo_id, obj_id, is_moving=True):
        """Create a window dictionary with proper padding."""
        # Ensure actual_len doesn't exceed window size (safety check)
        actual_len = min(actual_len, self.window)
        
        left_hand = torch.tensor(left_hand, dtype=torch.float32)
        right_hand = torch.tensor(right_hand, dtype=torch.float32)
        object_motion = torch.tensor(object_motion, dtype=torch.float32)
        
        # Truncate if too long
        if left_hand.shape[0] > self.window:
            left_hand = left_hand[:self.window]
            right_hand = right_hand[:self.window]
            object_motion = object_motion[:self.window]

        # Center the entire window around the object's position in the first frame.
        origin = torch.zeros(3, dtype=torch.float32)
        if actual_len > 0:
            origin = object_motion[0, :3].clone()
            object_motion[:, :3] -= origin
            left_hand[:, :3] -= origin
            right_hand[:, :3] -= origin
        
        # Pad if necessary
        if actual_len < self.window:
            pad_len = self.window - actual_len
            left_hand = torch.cat([left_hand, torch.zeros(pad_len, self.pose_dim)], dim=0)
            right_hand = torch.cat([right_hand, torch.zeros(pad_len, self.pose_dim)], dim=0)
            object_motion = torch.cat([object_motion, torch.zeros(pad_len, self.pose_dim)], dim=0)
        
        return {
            'left_hand': left_hand,
            'right_hand': right_hand,
            'object_motion': object_motion,
            'seq_len': torch.tensor(actual_len),
            'demo_id': demo_id,
            'obj_id': obj_id,
            'is_moving': is_moving,
            'origin': origin
        }
    
    def _balance_dataset(self):
        """Balance the dataset by sampling moving and stationary windows."""
        print(f"Dataset statistics before balancing:")
        print(f"  Moving windows: {len(self.moving_windows)}")
        print(f"  Stationary windows: {len(self.stationary_windows)}")
        
        if len(self.moving_windows) == 0 or len(self.stationary_windows) == 0:
            print("Warning: No moving or stationary windows found!")
            self.window_data = self.moving_windows + self.stationary_windows
            return
        
        # Calculate target sizes
        total_moving = len(self.moving_windows)
        total_stationary = len(self.stationary_windows)
        
        # Check if we should use less aggressive oversampling
        max_oversample_ratio = getattr(self, 'max_oversample_ratio', 5.0)
        
        if self.balance_ratio == 0.5:
            # For 50% balance, check if oversampling would be too aggressive
            if total_stationary > total_moving * max_oversample_ratio:
                print(f"Warning: Reducing target to avoid excessive oversampling (>{max_oversample_ratio}x)")
                # Instead of matching stationary, limit oversampling
                target_moving = min(total_stationary, int(total_moving * max_oversample_ratio))
                target_stationary = target_moving
            else:
                target_size = min(total_moving, total_stationary)
                target_moving = target_size
                target_stationary = target_size
        else:
            # Custom balance ratio with oversampling limits
            total_target = max(total_moving, total_stationary)
            target_moving = int(total_target * self.balance_ratio)
            target_stationary = int(total_target * (1 - self.balance_ratio))
            
            # Apply oversampling limits
            if target_moving > total_moving * max_oversample_ratio:
                target_moving = int(total_moving * max_oversample_ratio)
                print(f"Limited moving oversampling to {max_oversample_ratio}x")
            
            if target_stationary > total_stationary * max_oversample_ratio:
                target_stationary = int(total_stationary * max_oversample_ratio)
                print(f"Limited stationary oversampling to {max_oversample_ratio}x")
        
        # Sample balanced windows (with replacement if needed)
        if target_moving > 0:
            if target_moving <= total_moving:
                # Sample without replacement
                selected_moving = random.sample(self.moving_windows, target_moving)
            else:
                # Sample with replacement (oversample)
                selected_moving = random.choices(self.moving_windows, k=target_moving)
        else:
            selected_moving = []
            
        if target_stationary > 0:
            if target_stationary <= total_stationary:
                # Sample without replacement
                selected_stationary = random.sample(self.stationary_windows, target_stationary)
            else:
                # Sample with replacement (oversample)
                selected_stationary = random.choices(self.stationary_windows, k=target_stationary)
        else:
            selected_stationary = []
        
        # Combine and shuffle
        self.window_data = selected_moving + selected_stationary
        random.shuffle(self.window_data)
        
        print(f"Dataset statistics after balancing:")
        print(f"  Selected moving windows: {len(selected_moving)}")
        print(f"  Selected stationary windows: {len(selected_stationary)}")
        print(f"  Total windows: {len(self.window_data)}")
        if len(self.window_data) > 0:
            print(f"  Moving ratio: {len(selected_moving) / len(self.window_data) * 100:.1f}%")
        print(f"  Velocity threshold: {self.velocity_threshold}")
        
        # Report oversampling ratios
        if len(selected_moving) > total_moving:
            oversample_ratio = len(selected_moving) / total_moving
            print(f"  Moving oversampling ratio: {oversample_ratio:.1f}x")
        if len(selected_stationary) > total_stationary:
            oversample_ratio = len(selected_stationary) / total_stationary  
            print(f"  Stationary oversampling ratio: {oversample_ratio:.1f}x")

    def sample_window(self, mode='random'):
        """Sample a window from pre-computed windows (like in overfit script)."""
        if len(self.window_data) == 0:
            raise ValueError("No windows available")
        
        if mode == 'random':
            # Random sampling
            idx = random.randint(0, len(self.window_data) - 1)
        elif mode == 'sequential':
            # Sequential sampling
            idx = self.current_window_idx
            self.current_window_idx = (self.current_window_idx + 1) % len(self.window_data)
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
        
        return self.window_data[idx]

    def __len__(self):
        return len(self.window_data)

    def __getitem__(self, index):
        return self.window_data[index]

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        train_dataset,
        val_dataset,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        use_wandb=True,
        use_weighted_loss=False,
        moving_weight=1.0
    ):
        super().__init__()

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0
        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder
        self.vis_folder = results_folder.replace("weights", "vis_res")
        self.opt = opt
        self.window = opt.window
        
        # Store datasets directly (no need for DataLoader cycling)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Weighted loss parameters
        self.use_weighted_loss = use_weighted_loss
        self.moving_weight = moving_weight

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, f'model-{milestone}.pt'))

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            data = torch.load(os.path.join(self.results_folder, f'model-{milestone}.pt'))
        else:
            data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        for idx in range(self.train_num_steps):
            self.optimizer.zero_grad()

            for i in range(self.gradient_accumulate_every):
                # Sample from dataset directly (like overfit script)
                data_dict = self.train_dataset.sample_window(mode='random')
                
                left_hand = data_dict['left_hand'].cuda().unsqueeze(0)  # Add batch dim
                right_hand = data_dict['right_hand'].cuda().unsqueeze(0)
                object_motion = data_dict['object_motion'].cuda().unsqueeze(0)
                seq_len = data_dict['seq_len'].cuda().unsqueeze(0)

                hand_poses = torch.cat([left_hand, right_hand], dim=-1)
                
                # Generate padding mask (like overfit script)
                actual_seq_len = seq_len + 1
                tmp_mask = torch.arange(self.window+1, device='cuda').expand(hand_poses.shape[0], self.window+1) < actual_seq_len[:, None]
                padding_mask = tmp_mask[:, None, :]

                with autocast(enabled=self.amp):
                    loss = self.model(object_motion, hand_poses, padding_mask=padding_mask)
                    
                    # Apply weighted loss if enabled
                    if self.use_weighted_loss and 'is_moving' in data_dict:
                        is_moving = data_dict['is_moving']
                        if is_moving:
                            loss = loss * self.moving_weight
                    
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                if self.use_wandb:
                    # Log additional metrics
                    log_dict = {"Train/Loss": loss.item()}
                    if 'is_moving' in data_dict:
                        log_dict["Train/Is_Moving"] = float(data_dict['is_moving'])
                    wandb.log(log_dict)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()

            if self.step % self.save_and_sample_every == 0:
                self.ema.ema_model.eval()
                with torch.no_grad():
                    # Sample from validation dataset
                    val_data_dict = self.val_dataset.sample_window(mode='random')
                    left_hand = val_data_dict['left_hand'].cuda().unsqueeze(0)
                    right_hand = val_data_dict['right_hand'].cuda().unsqueeze(0)
                    object_motion = val_data_dict['object_motion'].cuda().unsqueeze(0)
                    seq_len = val_data_dict['seq_len'].cuda().unsqueeze(0)
                    
                    hand_poses = torch.cat([left_hand, right_hand], dim=-1)
                    
                    # Generate validation padding mask (like overfit script)
                    actual_seq_len = seq_len + 1
                    tmp_mask = torch.arange(self.window+1, device='cuda').expand(hand_poses.shape[0], self.window+1) < actual_seq_len[:, None]
                    padding_mask = tmp_mask[:, None, :]
                    
                    val_loss = self.model(object_motion, hand_poses, padding_mask=padding_mask)
                    if self.use_wandb:
                        wandb.log({"Validation/Loss": val_loss.item()})
                
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

            self.step += 1
        
        print('Training complete')
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

        mpjpe_list = []
        
        # Use validation dataset for testing
        val_dataset = self.val_dataset
        num_test_samples = min(100, len(val_dataset))  # Test on subset

        # For enhanced visualization - collect trajectory segments
        trajectory_segments = []

        with torch.no_grad():
            moving_samples_processed = 0
            max_moving_samples = 50  # Limit to 50 moving examples for visualization
            
            for s_idx in range(num_test_samples):
                data_dict = val_dataset.sample_window(mode='sequential')
                
                # Skip stationary examples - only visualize moving ones
                if not data_dict['is_moving']:
                    continue
                
                # Stop after processing enough moving examples
                if moving_samples_processed >= max_moving_samples:
                    break
                
                left_hand = data_dict['left_hand'].cuda().unsqueeze(0)
                right_hand = data_dict['right_hand'].cuda().unsqueeze(0)
                object_motion_gt = data_dict['object_motion'].cuda().unsqueeze(0)
                seq_len = data_dict['seq_len'].cuda().unsqueeze(0)
                origin = data_dict['origin'].cuda().unsqueeze(0)

                hand_poses = torch.cat([left_hand, right_hand], dim=-1)
                
                actual_seq_len = seq_len + 1
                tmp_mask = torch.arange(self.window+1, device='cuda').expand(hand_poses.shape[0], self.window+1) < actual_seq_len[:, None]
                padding_mask = tmp_mask[:, None, :]

                all_res_list = self.ema.ema_model.sample(object_motion_gt, hand_poses, padding_mask=padding_mask)

                # Denormalize the output
                all_res_list = val_dataset.denormalize(all_res_list, val_dataset.object_motion_mean.to(all_res_list.device), val_dataset.object_motion_std.to(all_res_list.device))
                object_motion_gt = val_dataset.denormalize(object_motion_gt, val_dataset.object_motion_mean.to(object_motion_gt.device), val_dataset.object_motion_std.to(object_motion_gt.device))

                # Store trajectory segment info before restoring absolute coordinates
                segment_info = {
                    'demo_id': data_dict['demo_id'],
                    'obj_id': data_dict['obj_id'], 
                    'is_moving': data_dict['is_moving'],
                    'origin': origin.cpu().numpy(),
                    'seq_len': seq_len.cpu().numpy(),
                    's_idx': s_idx
                }

                # Add the origin back to the predictions and ground truth
                all_res_list[:, :, :3] += origin.unsqueeze(1)
                object_motion_gt[:, :, :3] += origin.unsqueeze(1)

                for i in range(all_res_list.shape[0]):
                    pred_motion = all_res_list[i, :seq_len[i]]
                    gt_motion = object_motion_gt[i, :seq_len[i]]
                    
                    # Denormalize hand poses for visualization
                    left_hand_denorm = val_dataset.denormalize(left_hand[i, :seq_len[i]], 
                                                             val_dataset.left_hand_mean.to(left_hand.device), 
                                                             val_dataset.left_hand_std.to(left_hand.device))
                    right_hand_denorm = val_dataset.denormalize(right_hand[i, :seq_len[i]], 
                                                              val_dataset.right_hand_mean.to(right_hand.device), 
                                                              val_dataset.right_hand_std.to(right_hand.device))
                    
                    # Restore hand poses to original coordinate system
                    left_hand_denorm[:, :3] += origin[i]
                    right_hand_denorm[:, :3] += origin[i]
                    hand_poses_restored = torch.cat([left_hand_denorm, right_hand_denorm], dim=-1)
                    
                    # Using position from 9d representation
                    pred_pos = pred_motion[:, :3]
                    gt_pos = gt_motion[:, :3]
                    
                    mpjpe = torch.norm(pred_pos - gt_pos, dim=-1).mean().item()
                    mpjpe_list.append(mpjpe)

                    # Create a directory for each sample
                    vis_tag = f"{milestone}_sidx_{s_idx}_ex_{i}"
                    dest_folder = os.path.join(self.vis_folder, vis_tag)
                    os.makedirs(dest_folder, exist_ok=True)

                    # Save the results for visualization (original system)
                    np.save(os.path.join(dest_folder, "sampled_motion.npy"), pred_motion.cpu().numpy())
                    np.save(os.path.join(dest_folder, "ground_truth_object.npy"), gt_motion.cpu().numpy())
                    np.save(os.path.join(dest_folder, "input_hand_poses.npy"), hand_poses_restored.cpu().numpy())
                    
                    # Save enhanced context for better visualization
                    # Create centered versions for comparison (only subtract origin from position components)
                    centered_gt_motion = gt_motion.clone()
                    centered_pred_motion = pred_motion.clone()
                    centered_gt_motion[:, :3] -= origin[i]  # Only center the position (first 3 dims)
                    centered_pred_motion[:, :3] -= origin[i]  # Only center the position (first 3 dims)
                    
                    context_info = {
                        'demo_id': data_dict['demo_id'],
                        'obj_id': data_dict['obj_id'],
                        'is_moving': bool(data_dict['is_moving']),
                        'origin': origin[i].cpu().numpy(),
                        'actual_length': int(seq_len[i]),
                        'window_idx': s_idx,
                        'batch_idx': i,
                        'milestone': milestone,
                        # Save centered versions for comparison
                        'centered_gt_motion': centered_gt_motion.cpu().numpy(),
                        'centered_pred_motion': centered_pred_motion.cpu().numpy(),
                        'motion_type': 'moving' if data_dict['is_moving'] else 'stationary'
                    }
                    
                    # Save context as JSON for easier inspection
                    import json
                    with open(os.path.join(dest_folder, "context_info.json"), 'w') as f:
                        json.dump(context_info, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
                    
                    # Also save as pickle for programmatic access
                    with open(os.path.join(dest_folder, "context_info.pkl"), 'wb') as f:
                        pickle.dump(context_info, f)

                    # Create enhanced visualization
                    self.create_enhanced_visualization(dest_folder, context_info)
                    
                    # Generate the standard visualization video as well
                    create_training_video(dest_folder)

                # Store for potential trajectory reconstruction
                trajectory_segments.append({
                    'segment_info': segment_info,
                    'data_dict': data_dict
                })
                
                moving_samples_processed += 1

        mean_mpjpe = np.mean(mpjpe_list) if mpjpe_list else 0.0
        print(f"Processed {moving_samples_processed} moving examples")
        print(f"Mean MPJPE: {mean_mpjpe}")
        
        # Save trajectory segments for potential reconstruction
        segments_save_path = os.path.join(self.vis_folder, f"trajectory_segments_{milestone}.pkl")
        with open(segments_save_path, 'wb') as f:
            pickle.dump(trajectory_segments, f)
        print(f"Saved trajectory segments to: {segments_save_path}")

    def create_enhanced_visualization(self, dest_folder, context_info):
        """Create enhanced visualization showing motion in original coordinate system"""
        try:
            import matplotlib.pyplot as plt
            
            # Load the motion data
            sampled_motion = np.load(os.path.join(dest_folder, "sampled_motion.npy"))
            gt_motion = np.load(os.path.join(dest_folder, "ground_truth_object.npy"))
            hand_poses = np.load(os.path.join(dest_folder, "input_hand_poses.npy"))
            
            # Split hand poses
            left_hand = hand_poses[:, :9]
            right_hand = hand_poses[:, 9:]
            
            # Extract positions (first 3 dimensions)
            sampled_pos = sampled_motion[:, :3]
            gt_pos = gt_motion[:, :3]
            left_hand_pos = left_hand[:, :3]  
            right_hand_pos = right_hand[:, :3]
            
            # Create comprehensive visualization
            fig = plt.figure(figsize=(16, 12))
            
            # Top-down view (X-Y plane) - Most important for seeing motion
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.plot(gt_pos[:, 0], gt_pos[:, 1], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
            ax1.plot(sampled_pos[:, 0], sampled_pos[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.8)
            ax1.plot(left_hand_pos[:, 0], left_hand_pos[:, 1], 'b-', linewidth=1, label='Left Hand', alpha=0.6)
            ax1.plot(right_hand_pos[:, 0], right_hand_pos[:, 1], 'm-', linewidth=1, label='Right Hand', alpha=0.6)
            ax1.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=80, marker='o', label='GT Start', zorder=5)
            ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='darkgreen', s=80, marker='s', label='GT End', zorder=5)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'Top View - Original Coordinate System\n{context_info["motion_type"].title()} Motion')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Side view (X-Z plane) 
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.plot(gt_pos[:, 0], gt_pos[:, 2], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
            ax2.plot(sampled_pos[:, 0], sampled_pos[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
            ax2.plot(left_hand_pos[:, 0], left_hand_pos[:, 2], 'b-', linewidth=1, label='Left Hand', alpha=0.6)
            ax2.plot(right_hand_pos[:, 0], right_hand_pos[:, 2], 'm-', linewidth=1, label='Right Hand', alpha=0.6)
            ax2.scatter(gt_pos[0, 0], gt_pos[0, 2], c='green', s=80, marker='o', zorder=5)
            ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 2], c='darkgreen', s=80, marker='s', zorder=5)
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Z (m)')
            ax2.set_title('Side View (X-Z)')
            ax2.grid(True, alpha=0.3)
            
            # Y-Z view (front view)
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.plot(gt_pos[:, 1], gt_pos[:, 2], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
            ax3.plot(sampled_pos[:, 1], sampled_pos[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.8)
            ax3.plot(left_hand_pos[:, 1], left_hand_pos[:, 2], 'b-', linewidth=1, label='Left Hand', alpha=0.6)
            ax3.plot(right_hand_pos[:, 1], right_hand_pos[:, 2], 'm-', linewidth=1, label='Right Hand', alpha=0.6)
            ax3.scatter(gt_pos[0, 1], gt_pos[0, 2], c='green', s=80, marker='o', zorder=5)
            ax3.scatter(gt_pos[-1, 1], gt_pos[-1, 2], c='darkgreen', s=80, marker='s', zorder=5)
            ax3.set_xlabel('Y (m)')
            ax3.set_ylabel('Z (m)')
            ax3.set_title('Front View (Y-Z)')
            ax3.grid(True, alpha=0.3)
            
            # Position error over time
            ax4 = fig.add_subplot(2, 3, 4)
            position_errors = np.linalg.norm(gt_pos - sampled_pos, axis=1)
            ax4.plot(position_errors, 'purple', linewidth=2)
            ax4.set_xlabel('Frame')
            ax4.set_ylabel('Position Error (m)')
            ax4.set_title(f'Position Error Over Time\nMean: {np.mean(position_errors):.4f}m')
            ax4.grid(True, alpha=0.3)
            
            # Distance from window origin over time
            ax5 = fig.add_subplot(2, 3, 5)
            origin = context_info['origin']
            gt_dist_from_origin = np.linalg.norm(gt_pos - origin, axis=1) 
            pred_dist_from_origin = np.linalg.norm(sampled_pos - origin, axis=1)
            ax5.plot(gt_dist_from_origin, 'g-', linewidth=2, label='GT Distance')
            ax5.plot(pred_dist_from_origin, 'r--', linewidth=2, label='Pred Distance')
            ax5.set_xlabel('Frame')
            ax5.set_ylabel('Distance from Window Origin (m)')
            ax5.set_title('Distance from Window Start Position')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Velocity comparison
            ax6 = fig.add_subplot(2, 3, 6)
            if len(gt_pos) > 1:
                gt_velocity = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
                pred_velocity = np.linalg.norm(np.diff(sampled_pos, axis=0), axis=1)
                frames = range(1, len(gt_pos))
                ax6.plot(frames, gt_velocity, 'g-', linewidth=2, label='GT Velocity')
                ax6.plot(frames, pred_velocity, 'r--', linewidth=2, label='Pred Velocity')
                ax6.set_xlabel('Frame')
                ax6.set_ylabel('Velocity (m/frame)')
                ax6.set_title('Velocity Over Time')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            
            plt.suptitle(f'Enhanced Motion Visualization - {context_info["demo_id"]} - {context_info["obj_id"]}\n'
                        f'Motion Type: {context_info["motion_type"].title()} | Length: {context_info["actual_length"]} frames', 
                        fontsize=14)
            
            plt.tight_layout()
            
            # Save the enhanced visualization
            enhanced_vis_path = os.path.join(dest_folder, "enhanced_visualization.png")
            plt.savefig(enhanced_vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Enhanced visualization saved: {enhanced_vis_path}")
            
        except Exception as e:
            print(f"Error creating enhanced visualization: {e}")
            import traceback
            traceback.print_exc()

def run_train(opt, device):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Create separate train and validation datasets
    train_dataset = HandToObjectDataset(
        opt.data_path, 
        window=opt.window, 
        train=True,
        velocity_threshold=opt.velocity_threshold,
        balance_ratio=opt.balance_ratio,
        max_oversample_ratio=opt.max_oversample_ratio
    )
    
    val_dataset = HandToObjectDataset(
        opt.data_path, 
        window=opt.window, 
        train=False,
        velocity_threshold=opt.velocity_threshold,
        balance_ratio=opt.balance_ratio,
        max_oversample_ratio=opt.max_oversample_ratio
    )
    
    repr_dim = train_dataset.pose_dim
    input_dim = train_dataset.pose_dim * 2

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
        d_input_feats=input_dim,
        timesteps=1000,
        objective="pred_x0",
        loss_type="l1",
        batch_size=opt.batch_size
    ).to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_dataset,
        val_dataset,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=opt.num_steps,
        gradient_accumulate_every=2,
        amp=True,
        results_folder=str(wdir),
        use_wandb=opt.use_wandb,
        use_weighted_loss=opt.use_weighted_loss,
        moving_weight=opt.moving_weight
    )

    trainer.train()

def run_sample(opt, device):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # For sampling, we only need validation dataset
    val_dataset = HandToObjectDataset(
        opt.data_path, 
        window=opt.window, 
        train=False,
        velocity_threshold=opt.velocity_threshold,
        balance_ratio=opt.balance_ratio,
        max_oversample_ratio=opt.max_oversample_ratio
    )
    repr_dim = val_dataset.pose_dim
    input_dim = val_dataset.pose_dim * 2

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
        d_input_feats=input_dim,
        timesteps=1000,
        objective="pred_x0",
        loss_type="l1",
        batch_size=opt.batch_size
    ).to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        val_dataset,  # Use val_dataset for both
        val_dataset,
        train_batch_size=opt.batch_size,
        results_folder=str(wdir),
        use_wandb=False,
        use_weighted_loss=opt.use_weighted_loss,
        moving_weight=opt.moving_weight
    )
    
    trainer.cond_sample_res()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/processed_data.pkl')
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='runs/train_hand_to_object')
    parser.add_argument('--wandb_pj_name', type=str, default='hand_object_diffusion')
    parser.add_argument('--entity', type=str, default='egorecon')
    parser.add_argument('--exp_name', type=str, default='initial_training')
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--test_sample_res', action='store_true')
    
    # Dataset balancing parameters
    parser.add_argument('--velocity_threshold', type=float, default=0.02, help='Velocity threshold for motion detection')
    parser.add_argument('--balance_ratio', type=float, default=0.5, help='Ratio of moving windows (0.5 = equal balance)')
    
    # Weighted loss parameters (alternative to aggressive oversampling)
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted loss instead of oversampling')
    parser.add_argument('--moving_weight', type=float, default=10.0, help='Weight multiplier for moving window losses')
    parser.add_argument('--max_oversample_ratio', type=float, default=5.0, help='Maximum oversampling ratio to prevent instability')
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.save_dir, opt.exp_name)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
