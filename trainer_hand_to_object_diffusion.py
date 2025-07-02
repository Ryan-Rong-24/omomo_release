import argparse
import os
import numpy as np
import yaml
import random
import json 
import pickle

import trimesh 

from tqdm import tqdm
from pathlib import Path

import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

from manip.model.transformer_hand_to_object_diffusion_model import CondGaussianDiffusion

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_rotation_error(pred_rot_6d, gt_rot_6d):
    """Compute rotation error between 6D rotation representations."""
    # Convert 6D to rotation matrices
    pred_rot_mat = transforms.rotation_6d_to_matrix(pred_rot_6d.reshape(-1, 6))
    gt_rot_mat = transforms.rotation_6d_to_matrix(gt_rot_6d.reshape(-1, 6))
    
    # Compute relative rotation
    rel_rot = torch.matmul(pred_rot_mat, gt_rot_mat.transpose(-2, -1))
    
    # Convert to angle difference
    trace = rel_rot.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    return angle * 180 / np.pi  # Convert to degrees

def compute_hand_object_metrics(pred_object, gt_object, pred_hands, gt_hands, seq_len):
    """Compute comprehensive metrics for hand-to-object task."""
    metrics = {}
    
    # Only compute on valid sequence length
    valid_pred_object = pred_object[:seq_len]
    valid_gt_object = gt_object[:seq_len]
    valid_pred_hands = pred_hands[:seq_len]
    valid_gt_hands = gt_hands[:seq_len]
    
    # Object position error (first 3 dimensions are translation)
    obj_pos_error = torch.norm(valid_pred_object[:, :3] - valid_gt_object[:, :3], dim=-1)
    metrics['object_pos_mse'] = obj_pos_error.mean().item()
    metrics['object_pos_max'] = obj_pos_error.max().item()
    
    # Object rotation error (dimensions 3:9 are 6D rotation)
    if valid_pred_object.shape[-1] >= 9:
        rot_error = compute_rotation_error(valid_pred_object[:, 3:9], valid_gt_object[:, 3:9])
        metrics['object_rot_error'] = rot_error.mean().item()
        metrics['object_rot_max'] = rot_error.max().item()
    
    # Hand consistency (how well the hands relate to predicted object)
    left_hand_pred = valid_pred_hands[:, :9]
    right_hand_pred = valid_pred_hands[:, 9:18]
    left_hand_gt = valid_gt_hands[:, :9]
    right_hand_gt = valid_gt_hands[:, 9:18]
    
    # Hand position consistency
    lhand_pos_error = torch.norm(left_hand_pred[:, :3] - left_hand_gt[:, :3], dim=-1)
    rhand_pos_error = torch.norm(right_hand_pred[:, :3] - right_hand_gt[:, :3], dim=-1)
    
    metrics['lhand_pos_error'] = lhand_pos_error.mean().item()
    metrics['rhand_pos_error'] = rhand_pos_error.mean().item()
    metrics['hand_pos_error'] = (lhand_pos_error.mean() + rhand_pos_error.mean()).item() / 2
    
    # Overall trajectory smoothness
    obj_vel = torch.diff(valid_pred_object[:, :3], dim=0)
    obj_acc = torch.diff(obj_vel, dim=0)
    metrics['object_smoothness'] = torch.norm(obj_acc, dim=-1).mean().item()
    
    return metrics

def compute_trajectory_metrics(pred_trajectory, gt_trajectory, seq_len):
    """Compute trajectory-level metrics."""
    valid_pred = pred_trajectory[:seq_len]
    valid_gt = gt_trajectory[:seq_len]
    
    # Overall MSE
    mse = torch.nn.functional.mse_loss(valid_pred, valid_gt)
    
    # Per-dimension errors
    pos_mse = torch.nn.functional.mse_loss(valid_pred[:, :3], valid_gt[:, :3])
    rot_mse = torch.nn.functional.mse_loss(valid_pred[:, 3:], valid_gt[:, 3:]) if valid_pred.shape[-1] > 3 else 0
    
    return {
        'trajectory_mse': mse.item(),
        'position_mse': pos_mse.item(),
        'rotation_mse': rot_mse.item() if isinstance(rot_mse, torch.Tensor) else rot_mse
    }

class HandToObjectDataset:
    """Dataset class for hand-to-object diffusion training."""
    
    def __init__(self, data_path, demo_id=None, target_object_id=None, window=64, train=True):
        self.window = window
        self.data_path = data_path
        self.train = train
        
        # Load the processed data
        print(f"Loading data from {data_path}...")
        processed_data = load_pickle(data_path)
        
        # Select demonstration
        if demo_id is None:
            demo_id = list(processed_data.keys())[0]
        
        self.demo_id = demo_id
        demo_data = processed_data[demo_id]
        print(f"Using demonstration: {demo_id}")
        
        # Extract trajectories
        left_hand_data = demo_data['left_hand']['poses_9d']  # [T, 9]
        right_hand_data = demo_data['right_hand']['poses_9d']  # [T, 9]
        
        # Select target object
        objects_data = demo_data['objects']
        if target_object_id is None:
            target_object_id = list(objects_data.keys())[0]
        
        self.target_object_id = target_object_id
        object_data = objects_data[target_object_id]['poses_9d']  # [T, 9]
        print(f"Using target object: {target_object_id}")
        
        # Find common length
        min_length = min(len(left_hand_data), len(right_hand_data), len(object_data))
        print(f"Original trajectory lengths - Left: {len(left_hand_data)}, Right: {len(right_hand_data)}, Object: {len(object_data)}")
        print(f"Using common length: {min_length}")
        
        # Store full trajectories
        self.left_hand_full = torch.tensor(left_hand_data[:min_length], dtype=torch.float32)
        self.right_hand_full = torch.tensor(right_hand_data[:min_length], dtype=torch.float32)
        self.object_motion_full = torch.tensor(object_data[:min_length], dtype=torch.float32)
        self.full_length = min_length
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
        # Pre-compute all windows
        self.window_data = []
        self._prepare_windows()
        
        print(f"Dataset initialized: {self.full_length} frames, window size: {self.window}")
        print(f"Created {len(self.window_data)} overlapping windows")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics for object motion."""
        # Compute statistics on object motion
        obj_pos = self.object_motion_full[:, :3]  # [T, 3]
        obj_rot = self.object_motion_full[:, 3:]  # [T, 6]
        
        self.obj_pos_mean = obj_pos.mean(dim=0)
        self.obj_pos_std = obj_pos.std(dim=0)
        self.obj_rot_mean = obj_rot.mean(dim=0)
        self.obj_rot_std = obj_rot.std(dim=0)
        
        # Hand statistics
        hand_data = torch.cat([self.left_hand_full, self.right_hand_full], dim=-1)  # [T, 18]
        self.hand_mean = hand_data.mean(dim=0)
        self.hand_std = hand_data.std(dim=0)
        
        print(f"Normalization stats computed - Obj pos std: {self.obj_pos_std.mean():.4f}, Obj rot std: {self.obj_rot_std.mean():.4f}")
    
    def normalize_object_motion(self, obj_motion):
        """Normalize object motion data."""
        normalized = obj_motion.clone()
        normalized[:, :3] = (obj_motion[:, :3] - self.obj_pos_mean) / (self.obj_pos_std + 1e-8)
        if obj_motion.shape[-1] > 3:
            normalized[:, 3:] = (obj_motion[:, 3:] - self.obj_rot_mean) / (self.obj_rot_std + 1e-8)
        return normalized
    
    def denormalize_object_motion(self, normalized_obj_motion):
        """Denormalize object motion data."""
        denormalized = normalized_obj_motion.clone()
        denormalized[:, :3] = normalized_obj_motion[:, :3] * (self.obj_pos_std + 1e-8) + self.obj_pos_mean
        if normalized_obj_motion.shape[-1] > 3:
            denormalized[:, 3:] = normalized_obj_motion[:, 3:] * (self.obj_rot_std + 1e-8) + self.obj_rot_mean
        return denormalized
    
    def normalize_hand_poses(self, hand_poses):
        """Normalize hand pose data."""
        return (hand_poses - self.hand_mean) / (self.hand_std + 1e-8)
    
    def denormalize_hand_poses(self, normalized_hand_poses):
        """Denormalize hand pose data."""
        return normalized_hand_poses * (self.hand_std + 1e-8) + self.hand_mean
    
    def _prepare_windows(self):
        """Pre-compute all overlapping windows."""
        step_size = self.window // 2  # 50% overlap
        
        for start_idx in range(0, self.full_length, step_size):
            end_idx = start_idx + self.window
            
            if end_idx > self.full_length:
                end_idx = self.full_length
            
            actual_len = end_idx - start_idx
            
            # Skip windows that are too short
            if actual_len < 30:
                continue
            
            # Extract window
            left_hand = self.left_hand_full[start_idx:end_idx]
            right_hand = self.right_hand_full[start_idx:end_idx]
            object_motion = self.object_motion_full[start_idx:end_idx]
            
            # Pad if necessary to reach window size
            if actual_len < self.window:
                pad_len = self.window - actual_len
                left_hand = torch.cat([left_hand, torch.zeros(pad_len, 9)], dim=0)
                right_hand = torch.cat([right_hand, torch.zeros(pad_len, 9)], dim=0)
                object_motion = torch.cat([object_motion, torch.zeros(pad_len, 9)], dim=0)
            
            # Store window data
            window_dict = {
                'left_hand': left_hand.unsqueeze(0),  # [1, T, 9]
                'right_hand': right_hand.unsqueeze(0),  # [1, T, 9]
                'object_motion': object_motion.unsqueeze(0),  # [1, T, 9]
                'seq_len': actual_len,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            self.window_data.append(window_dict)
    
    def __len__(self):
        return len(self.window_data)
    
    def __getitem__(self, index):
        """Get a specific window for DataLoader compatibility."""
        window_dict = self.window_data[index]
        
        # Prepare data dict similar to HandFootManipDataset format
        left_hand = window_dict['left_hand'].squeeze(0)  # [T, 9]
        right_hand = window_dict['right_hand'].squeeze(0)  # [T, 9]
        object_motion = window_dict['object_motion'].squeeze(0)  # [T, 9]
        
        # Normalize the data
        hand_poses = torch.cat([left_hand, right_hand], dim=-1)  # [T, 18]
        normalized_hand_poses = self.normalize_hand_poses(hand_poses)
        normalized_object_motion = self.normalize_object_motion(object_motion)
        
        data_dict = {
            'left_hand': normalized_hand_poses[:, :9],  # [T, 9]
            'right_hand': normalized_hand_poses[:, 9:18],  # [T, 9]
            'hand_poses': normalized_hand_poses,  # [T, 18]
            'object_motion': normalized_object_motion,  # [T, 9]
            'raw_object_motion': object_motion,  # [T, 9] - keep raw for evaluation
            'raw_hand_poses': hand_poses,  # [T, 18] - keep raw for evaluation
            'seq_len': window_dict['seq_len'],
            'demo_id': self.demo_id,
            'object_id': self.target_object_id
        }
        
        return data_dict

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
        save_and_sample_every=40000,
        results_folder='./results',
        use_wandb=True,
    ):
        super().__init__()

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity,
                      name=opt.exp_name, dir=opt.save_dir)

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
        self.data_path = opt.data_path

        # For evaluation
        self.test_on_train = getattr(opt, 'test_sample_res_on_train', False)
        self.for_quant_eval = getattr(opt, 'for_quant_eval', False)

        self.prep_dataloader(window_size=opt.window)

    def prep_dataloader(self, window_size):
        # Define dataset - for simplicity, use same data for train and val
        # In practice, you'd want separate train/val splits
        train_dataset = HandToObjectDataset(
            self.data_path, 
            demo_id=self.opt.demo_id,
            target_object_id=self.opt.target_object_id,
            window=window_size, 
            train=True
        )
        val_dataset = HandToObjectDataset(
            self.data_path,
            demo_id=self.opt.demo_id, 
            target_object_id=self.opt.target_object_id,
            window=window_size,
            train=False
        )

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size,
                                      shuffle=True, pin_memory=True, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size,
                                           shuffle=False, pin_memory=True, num_workers=4))

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

            nan_exists = False
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                
                # Extract hand poses and object motion (already normalized)
                hand_poses = data_dict['hand_poses'].cuda()  # BS X T X 18
                object_motion = data_dict['object_motion'].cuda()  # BS X T X 9
                seq_len = data_dict['seq_len']  # BS

                bs, num_steps, _ = object_motion.shape

                # Generate padding mask 
                actual_seq_len = seq_len + 1  # BS, + 1 since we need additional timestep for noise level
                tmp_mask = torch.arange(self.window+1).expand(object_motion.shape[0],
                                                             self.window+1) < actual_seq_len[:, None]
                padding_mask = tmp_mask[:, None, :].to(object_motion.device)  # BS X 1 X (T+1)

                with autocast(enabled=self.amp):    
                    loss_diffusion = self.model(object_motion, hand_poses, padding_mask=padding_mask)
                    loss = loss_diffusion

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(object_motion.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                        }
                        wandb.log(log_dict)

                    if idx % 10 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    
                    hand_poses = val_data_dict['hand_poses'].cuda()
                    val_object_motion = val_data_dict['object_motion'].cuda()
                    seq_len = val_data_dict['seq_len']

                    bs, num_steps, _ = val_object_motion.shape

                    # Generate padding mask 
                    actual_seq_len = seq_len + 1
                    tmp_mask = torch.arange(self.window+1).expand(val_object_motion.shape[0],
                                                                 self.window+1) < actual_seq_len[:, None]
                    padding_mask = tmp_mask[:, None, :].to(val_object_motion.device)

                    # Get validation loss 
                    val_loss_diffusion = self.model(val_object_motion, hand_poses, padding_mask=padding_mask)
                    val_loss = val_loss_diffusion 
                    
                    if self.use_wandb:
                        val_log_dict = {
                            "Validation/Loss/Total Loss": val_loss.item(),
                            "Validation/Loss/Diffusion Loss": val_loss_diffusion.item(),
                        }
                        wandb.log(val_log_dict)

                    milestone = self.step // self.save_and_sample_every
            
                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

                        # Sample and save results
                        sampled_object_motion = self.ema.ema_model.sample(
                            torch.zeros_like(val_object_motion), hand_poses, padding_mask=padding_mask
                        )
                        
                        # Save sampled results with evaluation
                        self.save_sample_results(sampled_object_motion, val_data_dict, self.step)

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def save_sample_results(self, sampled_motion, data_dict, step):
        """Save sampled results to files with evaluation metrics."""
        os.makedirs(self.vis_folder, exist_ok=True)
        
        # Denormalize for evaluation and saving
        sampled_motion_denorm = self.ds.denormalize_object_motion(sampled_motion.cpu())
        gt_motion_denorm = data_dict['raw_object_motion']
        hand_poses_denorm = data_dict['raw_hand_poses']
        
        # Save sampled motion
        save_path = os.path.join(self.vis_folder, f'sampled_motion_step_{step}.npy')
        np.save(save_path, sampled_motion_denorm.numpy())
        
        # Save ground truth for comparison
        gt_path = os.path.join(self.vis_folder, f'gt_motion_step_{step}.npy') 
        np.save(gt_path, gt_motion_denorm.numpy())
        
        # Save hand poses
        hand_path = os.path.join(self.vis_folder, f'hand_poses_step_{step}.npy')
        np.save(hand_path, hand_poses_denorm.numpy())
        
        # Compute and log evaluation metrics
        seq_len = data_dict['seq_len'][0].item()
        metrics = compute_hand_object_metrics(
            sampled_motion_denorm[0], gt_motion_denorm[0], 
            hand_poses_denorm[0], hand_poses_denorm[0], seq_len
        )
        
        traj_metrics = compute_trajectory_metrics(
            sampled_motion_denorm[0], gt_motion_denorm[0], seq_len
        )
        
        # Log metrics
        if self.use_wandb:
            log_dict = {f"Sample/Metrics/{k}": v for k, v in metrics.items()}
            log_dict.update({f"Sample/Trajectory/{k}": v for k, v in traj_metrics.items()})
            wandb.log(log_dict, step=step)
        
        print(f"Saved sample results to {self.vis_folder}")
        print(f"Object pos MSE: {metrics['object_pos_mse']:.4f}, Hand pos error: {metrics['hand_pos_error']:.4f}")

    def cond_sample_res(self):
        """Conditional sampling with comprehensive evaluation."""
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")
        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()

        num_sample = 50 if not self.for_quant_eval else 10
        
        # Collect metrics
        all_obj_pos_errors = []
        all_obj_rot_errors = []
        all_hand_errors = []
        all_traj_mses = []
        
        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.test_on_train:
                    val_data_dict = next(self.dl)
                else:
                    val_data_dict = next(self.val_dl)
                
                hand_poses = val_data_dict['hand_poses'].cuda()
                object_motion = val_data_dict['object_motion'].cuda()
                seq_len = val_data_dict['seq_len']

                # Generate padding mask 
                actual_seq_len = seq_len + 1
                tmp_mask = torch.arange(self.window+1).expand(object_motion.shape[0],
                                                             self.window+1) < actual_seq_len[:, None]
                padding_mask = tmp_mask[:, None, :].to(object_motion.device)

                # Sample from the model
                sampled_motion = self.ema.ema_model.sample(
                    torch.zeros_like(object_motion), hand_poses, padding_mask=padding_mask
                )

                # Denormalize for evaluation
                sampled_motion_denorm = self.ds.denormalize_object_motion(sampled_motion.cpu())
                gt_motion_denorm = val_data_dict['raw_object_motion']
                hand_poses_denorm = val_data_dict['raw_hand_poses']

                # Compute metrics for each sequence in batch
                for b_idx in range(sampled_motion.shape[0]):
                    seq_len_curr = seq_len[b_idx].item()
                    
                    # Compute metrics
                    metrics = compute_hand_object_metrics(
                        sampled_motion_denorm[b_idx], gt_motion_denorm[b_idx],
                        hand_poses_denorm[b_idx], hand_poses_denorm[b_idx], seq_len_curr
                    )
                    
                    traj_metrics = compute_trajectory_metrics(
                        sampled_motion_denorm[b_idx], gt_motion_denorm[b_idx], seq_len_curr
                    )
                    
                    all_obj_pos_errors.append(metrics['object_pos_mse'])
                    all_obj_rot_errors.append(metrics.get('object_rot_error', 0))
                    all_hand_errors.append(metrics['hand_pos_error'])
                    all_traj_mses.append(traj_metrics['trajectory_mse'])

                # Save example results
                if s_idx < 5:  # Save first 5 examples
                    vis_tag = f"{milestone}_sample_{s_idx}"
                    if self.test_on_train:
                        vis_tag += "_on_train"
                    
                    # Save results
                    save_path = os.path.join(self.vis_folder, f'{vis_tag}_sampled.npy')
                    np.save(save_path, sampled_motion_denorm.numpy())
                    
                    gt_path = os.path.join(self.vis_folder, f'{vis_tag}_gt.npy')
                    np.save(gt_path, gt_motion_denorm.numpy())
                    
                    hand_path = os.path.join(self.vis_folder, f'{vis_tag}_hands.npy')
                    np.save(hand_path, hand_poses_denorm.numpy())

        # Compute summary statistics
        final_metrics = {
            'object_pos_mse_mean': np.mean(all_obj_pos_errors),
            'object_pos_mse_std': np.std(all_obj_pos_errors),
            'object_rot_error_mean': np.mean(all_obj_rot_errors),
            'object_rot_error_std': np.std(all_obj_rot_errors),
            'hand_pos_error_mean': np.mean(all_hand_errors),
            'hand_pos_error_std': np.std(all_hand_errors),
            'trajectory_mse_mean': np.mean(all_traj_mses),
            'trajectory_mse_std': np.std(all_traj_mses),
            'num_samples': len(all_obj_pos_errors)
        }
        
        print("*" * 60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("*" * 60)
        print(f"Number of samples: {final_metrics['num_samples']}")
        print(f"Object Position MSE: {final_metrics['object_pos_mse_mean']:.4f} ± {final_metrics['object_pos_mse_std']:.4f}")
        print(f"Object Rotation Error (deg): {final_metrics['object_rot_error_mean']:.4f} ± {final_metrics['object_rot_error_std']:.4f}")
        print(f"Hand Position Error: {final_metrics['hand_pos_error_mean']:.4f} ± {final_metrics['hand_pos_error_std']:.4f}")
        print(f"Trajectory MSE: {final_metrics['trajectory_mse_mean']:.4f} ± {final_metrics['trajectory_mse_std']:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(self.vis_folder, f'evaluation_metrics_{milestone}.json')
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        if self.use_wandb:
            log_dict = {f"Eval/{k}": v for k, v in final_metrics.items()}
            wandb.log(log_dict)

    def sample_full_trajectory(self):
        """Sample full trajectory using sliding windows."""
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")
        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()

        print("Sampling full trajectory using sliding windows...")
        
        # Use the full dataset trajectory
        full_length = self.ds.full_length
        left_hand_full = self.ds.left_hand_full.cuda()
        right_hand_full = self.ds.right_hand_full.cuda()
        
        # Normalize hand poses
        hand_poses_full = torch.cat([left_hand_full, right_hand_full], dim=-1)
        hand_poses_full_norm = self.ds.normalize_hand_poses(hand_poses_full).cuda()
        
        # Initialize output trajectory
        sampled_trajectory = torch.zeros_like(self.ds.object_motion_full).cuda()
        weight_map = torch.zeros(full_length).cuda()
        
        window_size = self.window
        overlap = window_size // 2
        step_size = window_size - overlap
        num_windows = (full_length - overlap + step_size - 1) // step_size
        
        with torch.no_grad():
            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = min(start_idx + window_size, full_length)
                actual_window_size = end_idx - start_idx
                
                print(f"Processing window {i+1}/{num_windows}: frames {start_idx}-{end_idx}")
                
                # Extract normalized hand poses window
                hand_poses_window = hand_poses_full_norm[start_idx:end_idx].unsqueeze(0)
                
                # Pad if necessary
                if actual_window_size < window_size:
                    pad_len = window_size - actual_window_size
                    hand_poses_window = torch.cat([hand_poses_window, torch.zeros(1, pad_len, 18).cuda()], dim=1)
                
                # Prepare input
                object_motion_init = torch.zeros(1, window_size, 9).cuda()
                
                # Generate padding mask
                seq_len_tensor = torch.tensor([actual_window_size + 1]).cuda()
                tmp_mask = torch.arange(window_size + 1, device='cuda').expand(1, window_size + 1) < seq_len_tensor[:, None].repeat(1, window_size + 1)
                padding_mask = tmp_mask[:, None, :]
                
                # Sample window
                sampled_window = self.ema.ema_model.sample(object_motion_init, hand_poses_window, padding_mask=padding_mask)
                sampled_window = sampled_window[0, :actual_window_size]
                
                # Add to output with overlap handling
                sampled_trajectory[start_idx:end_idx] += sampled_window
                weight_map[start_idx:end_idx] += 1.0
        
        # Average overlapping regions
        sampled_trajectory = sampled_trajectory / weight_map.unsqueeze(-1)
        
        # Denormalize the trajectory
        sampled_trajectory_denorm = self.ds.denormalize_object_motion(sampled_trajectory.cpu())
        
        # Compute full trajectory metrics
        gt_trajectory = self.ds.object_motion_full
        full_metrics = compute_trajectory_metrics(sampled_trajectory_denorm, gt_trajectory, full_length)
        
        print(f"✅ Full trajectory sampling completed: {sampled_trajectory_denorm.shape}")
        print(f"Full trajectory MSE: {full_metrics['trajectory_mse']:.4f}")
        print(f"Position MSE: {full_metrics['position_mse']:.4f}")
        print(f"Rotation MSE: {full_metrics['rotation_mse']:.4f}")
        
        # Save results
        os.makedirs(self.vis_folder, exist_ok=True) 
        save_path = os.path.join(self.vis_folder, f'sampled_full_trajectory_{milestone}.npy')
        np.save(save_path, sampled_trajectory_denorm.numpy())
        
        gt_path = os.path.join(self.vis_folder, f'gt_full_trajectory_{milestone}.npy')
        np.save(gt_path, gt_trajectory.numpy())
        
        hand_path = os.path.join(self.vis_folder, f'hand_poses_full_{milestone}.npy')
        np.save(hand_path, hand_poses_full.cpu().numpy())
        
        # Save metrics
        metrics_path = os.path.join(self.vis_folder, f'full_trajectory_metrics_{milestone}.json')
        with open(metrics_path, 'w') as f:
            json.dump(full_metrics, f, indent=2)
        
        print(f"Saved results to {self.vis_folder}")

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 9  # Object motion dimension (3D translation + 6D rotation)
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
        train_num_steps=opt.num_steps,
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

    # Define model 
    repr_dim = 9
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
        train_num_steps=opt.num_steps,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
        use_wandb=False 
    )
    
    # Use comprehensive evaluation instead of basic sampling
    trainer.cond_sample_res()
    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='output folder for weights and visualizations')
    parser.add_argument('--wandb_pj_name', type=str, default='egorecon', help='wandb project name')
    parser.add_argument('--entity', default='egorecon', help='W&B entity')
    parser.add_argument('--exp_name', default='hand_to_object_diffusion', help='save to project/exp_name')
    parser.add_argument('--device', default='0', help='cuda device')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/processed_data.pkl', help='Path to processed data pickle file')
    parser.add_argument('--demo_id', type=str, default=None, help='Specific demo ID to use (if None, use first available)')
    parser.add_argument('--target_object_id', type=str, default=None, help='Specific object ID to track (if None, use first available)')

    # Model parameters
    parser.add_argument('--window', type=int, default=64, help='Training window size and model max sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='Transformer model dimension')
    parser.add_argument('--n_dec_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_k', type=int, default=64, help='Key dimension')
    parser.add_argument('--d_v', type=int, default=64, help='Value dimension')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_steps', type=int, default=100000, help='Number of training steps')

    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint path')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true", help="Test sampling results")
    parser.add_argument("--test_sample_res_on_train", action="store_true", help="Test sampling on training data")
    parser.add_argument("--for_quant_eval", action="store_true", help="Run quantitative evaluation")
    parser.add_argument("--sample_full_trajectory", action="store_true", help="Sample full trajectory using sliding windows")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    
    if opt.test_sample_res:
        run_sample(opt, device)
    elif opt.sample_full_trajectory:
        # Prepare Directories
        save_dir = Path(opt.save_dir)
        wdir = save_dir / 'weights'

        # Define model 
        repr_dim = 9
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
            train_num_steps=opt.num_steps,
            gradient_accumulate_every=2,
            ema_decay=0.995,
            amp=True,
            results_folder=str(wdir),
            use_wandb=False 
        )
        
        trainer.sample_full_trajectory()
        torch.cuda.empty_cache()
    else:
        run_train(opt, device) 