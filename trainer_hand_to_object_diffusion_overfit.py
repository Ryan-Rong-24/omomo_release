import argparse
import os
import numpy as np
import yaml
import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import wandb
import pickle
import random

from manip.model.transformer_hand_to_object_diffusion_model import CondGaussianDiffusion

def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

class HandToObjectDataset:
    """Similar to HandFootManipDataset structure."""
    
    def __init__(self, data_path, demo_id=None, target_object_id=None, window=64, use_velocity=True):
        self.window = window
        self.data_path = data_path
        self.use_velocity = use_velocity
        
        # Load the processed data
        print(f"Loading data from {data_path}...")
        processed_data = load_pickle(data_path)
        
        # Select demonstration
        if demo_id is None:
            demo_id = list(processed_data.keys())[0]
        
        self.demo_id = demo_id
        demo_data = processed_data[demo_id]
        print(f"Using demonstration: {demo_id}")
        
        # Choose data format based on use_velocity
        pose_key = 'poses_12d' if use_velocity else 'poses_9d'
        pose_dim = 12 if use_velocity else 9
        print(f"Using {pose_key} data (dimension: {pose_dim})")
        
        # Extract trajectories
        left_hand_data = demo_data['left_hand'][pose_key]  # [T, 9/12]
        right_hand_data = demo_data['right_hand'][pose_key]  # [T, 9/12]
        
        # Select target object
        objects_data = demo_data['objects']
        if target_object_id is None:
            target_object_id = list(objects_data.keys())[0]
        
        self.target_object_id = target_object_id
        object_data = objects_data[target_object_id][pose_key]  # [T, 9/12]
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
        self.pose_dim = pose_dim
        
        # Pre-compute all windows
        self.window_data = []
        self._prepare_windows()
        
        # For sequential sampling
        self.current_window_idx = 0
        
        print(f"Dataset initialized: {self.full_length} frames, window size: {self.window}")
        print(f"Created {len(self.window_data)} overlapping windows")
        print(f"Window step size: {self.window // 2} (50% overlap)")
        print(f"Data dimensions: Left hand {self.left_hand_full.shape}, Right hand {self.right_hand_full.shape}, Object {self.object_motion_full.shape}")
    
    def _prepare_windows(self):
        """Pre-compute all overlapping windows"""
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
                left_hand = torch.cat([left_hand, torch.zeros(pad_len, self.pose_dim)], dim=0)
                right_hand = torch.cat([right_hand, torch.zeros(pad_len, self.pose_dim)], dim=0)
                object_motion = torch.cat([object_motion, torch.zeros(pad_len, self.pose_dim)], dim=0)
            
            # Store window data
            window_dict = {
                'left_hand': left_hand.unsqueeze(0),  # [1, T, 9/12]
                'right_hand': right_hand.unsqueeze(0),  # [1, T, 9/12]
                'object_motion': object_motion.unsqueeze(0),  # [1, T, 9/12]
                'seq_len': torch.tensor([actual_len]),
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            self.window_data.append(window_dict)
    
    def __len__(self):
        return len(self.window_data)
    
    def __getitem__(self, index):
        """Get a specific window (for DataLoader compatibility)."""
        return self.window_data[index]
    
    def sample_window(self, mode='random'):
        """Sample a window from pre-computed windows."""
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

def sample_full_trajectory_inpaint(diffusion_model, dataset, device, window_size=64, overlap=32):
    print(f"\nSampling full trajectory using inpainting (window={window_size}, overlap={overlap})...")
    full_length = dataset.full_length
    left_hand_full = dataset.left_hand_full.to(device)
    right_hand_full = dataset.right_hand_full.to(device)
    hand_poses = torch.cat([left_hand_full, right_hand_full], dim=-1).unsqueeze(0)

    step_size = window_size - overlap
    num_windows = (full_length - overlap + step_size - 1) // step_size

    sampled_trajectory = torch.zeros_like(dataset.object_motion_full).to(device)

    diffusion_model.eval()
    with torch.no_grad():
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, full_length)
            actual_window_size = end_idx - start_idx

            hand_window = hand_poses[:, start_idx:end_idx, :]

            if actual_window_size < window_size:
                pad_len = window_size - actual_window_size
                hand_window = torch.cat([hand_window, torch.zeros(1, pad_len, hand_window.shape[2]).to(device)], dim=1)

            shape = (1, window_size, dataset.pose_dim)
            sampled_window = diffusion_model.long_ddim_sample(shape, hand_window)
            sampled_trajectory[start_idx:end_idx] = sampled_window[0, :actual_window_size]

    return sampled_trajectory.cpu()

def sample_full_trajectory_sliding_window(diffusion_model, dataset, device, window_size=64, overlap=32):
    """
    Sample the full trajectory using sliding windows with overlap.
    """
    print(f"\nSampling full trajectory using sliding windows (window={window_size}, overlap={overlap})...")
    
    full_length = dataset.full_length
    left_hand_full = dataset.left_hand_full.to(device)
    right_hand_full = dataset.right_hand_full.to(device)
    
    # Initialize output trajectory
    sampled_trajectory = torch.zeros_like(dataset.object_motion_full).to(device)
    weight_map = torch.zeros(full_length).to(device)
    
    step_size = window_size - overlap
    num_windows = (full_length - overlap + step_size - 1) // step_size
    
    diffusion_model.eval()
    with torch.no_grad():
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, full_length)
            actual_window_size = end_idx - start_idx
            
            print(f"Processing window {i+1}/{num_windows}: frames {start_idx}-{end_idx}")
            
            # Extract window
            left_hand_window = left_hand_full[start_idx:end_idx].unsqueeze(0)
            right_hand_window = right_hand_full[start_idx:end_idx].unsqueeze(0)
            
            # Pad if necessary
            if actual_window_size < window_size:
                pad_len = window_size - actual_window_size
                left_hand_window = torch.cat([left_hand_window, torch.zeros(1, pad_len, dataset.pose_dim).to(device)], dim=1)
                right_hand_window = torch.cat([right_hand_window, torch.zeros(1, pad_len, dataset.pose_dim).to(device)], dim=1)
            
            # Prepare input
            hand_poses = torch.cat([left_hand_window, right_hand_window], dim=-1)
            object_motion_init = torch.zeros(1, window_size, dataset.pose_dim).to(device)
            
            # Generate padding mask
            seq_len_tensor = torch.tensor([actual_window_size + 1]).to(device)
            tmp_mask = torch.arange(window_size + 1, device=device).expand(1, window_size + 1) < seq_len_tensor[:, None].repeat(1, window_size + 1)
            padding_mask = tmp_mask[:, None, :]
            
            # Sample window
            sampled_window = diffusion_model.sample(object_motion_init, hand_poses, padding_mask=padding_mask)
            sampled_window = sampled_window[0, :actual_window_size]
            
            # Add to output with overlap handling
            sampled_trajectory[start_idx:end_idx] += sampled_window
            weight_map[start_idx:end_idx] += 1.0
    
    # Average overlapping regions
    sampled_trajectory = sampled_trajectory / weight_map.unsqueeze(-1)
    
    print(f"Full trajectory sampling completed: {sampled_trajectory.shape}")
    return sampled_trajectory.cpu()

def evaluate_model(diffusion_model, dataset, device, num_eval_windows=10):
    """
    Evaluate the model by sampling from multiple windows and computing trajectory accuracy.
    Returns mean position error and max position error.
    """
    diffusion_model.eval()
    
    total_position_errors = []
    total_max_errors = []
    
    with torch.no_grad():
        # Evaluate on a subset of windows
        eval_windows = min(num_eval_windows, len(dataset.window_data))
        
        for i in range(eval_windows):
            # Get a window
            data_dict = dataset.window_data[i]
            
            # Move to device
            left_hand = data_dict['left_hand'].to(device)
            right_hand = data_dict['right_hand'].to(device)
            object_motion_gt = data_dict['object_motion'].to(device)
            seq_len = data_dict['seq_len'].to(device)
            
            # Prepare input
            hand_poses = torch.cat([left_hand, right_hand], dim=-1)  # [1, T, 18/24]
            object_motion_init = torch.zeros_like(object_motion_gt).to(device)
            
            # Generate padding mask
            actual_seq_len = seq_len + 1
            tmp_mask = torch.arange(dataset.window + 1, device=device).expand(1, dataset.window + 1) < actual_seq_len[:, None].repeat(1, dataset.window + 1)
            padding_mask = tmp_mask[:, None, :]
            
            # Sample from model
            sampled_motion = diffusion_model.sample(object_motion_init, hand_poses, padding_mask=padding_mask)
            
            # Compute position errors (only for valid sequence length)
            valid_len = seq_len.item()
            
            # Extract positions (first 3 dimensions of each pose)
            if dataset.use_velocity:
                # For 12D: translation (0:3), velocity (3:6), rotation (6:12)
                gt_positions = object_motion_gt[0, :valid_len, 0:3]  # [T, 3]
                pred_positions = sampled_motion[0, :valid_len, 0:3]  # [T, 3]
            else:
                # For 9D: translation (0:3), rotation (3:9)
                gt_positions = object_motion_gt[0, :valid_len, 0:3]  # [T, 3]
                pred_positions = sampled_motion[0, :valid_len, 0:3]  # [T, 3]
            
            # Compute L2 distance per frame
            position_errors = torch.norm(pred_positions - gt_positions, dim=1)  # [T]
            
            # Store metrics
            mean_error = position_errors.mean().item()
            max_error = position_errors.max().item()
            
            total_position_errors.append(mean_error)
            total_max_errors.append(max_error)
    
    # Compute overall metrics
    overall_mean_error = sum(total_position_errors) / len(total_position_errors)
    overall_max_error = max(total_max_errors)
    
    diffusion_model.train()
    
    return overall_mean_error, overall_max_error

def train_overfit(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Load dataset
    try:
        dataset = HandToObjectDataset(
            opt.data_path, 
            demo_id=opt.demo_id, 
            target_object_id=opt.target_object_id,
            window=opt.window,
            use_velocity=opt.use_velocity
        )
        print(f"  Successfully loaded dataset from demo {dataset.demo_id}, object {dataset.target_object_id}")
        use_real_data = True
    except Exception as e:
        print(f"Failed to load real data: {e}")
        print("Cannot proceed without real data for overfitting.")
        return

    # Define model - use window size for model architecture
    repr_dim = dataset.pose_dim  # Output dimension (3D translation + 6D rotation)
    input_dim = dataset.pose_dim * 2  # Input dimension (2 hands × pose_dim each)
   
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
        batch_size=1
    )
   
    diffusion_model.to(device)

    # Initialize optimizer and scaler
    optimizer = Adam(diffusion_model.parameters(), lr=opt.learning_rate)
    scaler = GradScaler(enabled=True)

    # Initialize wandb
    if opt.use_wandb:
        wandb.init(
            config=opt,
            project=opt.wandb_pj_name,
            entity=opt.entity,
            name=opt.exp_name,
            dir=opt.save_dir
        )

    # Track best model based on evaluation metrics
    best_eval_error = float('inf')
    best_model_state = None
    best_step = 0
    
    # Training loop
    print("Starting training loop...")
    print(f"Total windows available: {len(dataset.window_data)}")
    print(f"Each epoch will see all {len(dataset.window_data)} windows once")
    print(f"Using {opt.sampling_mode} sampling mode")
    
    windows_seen = set()
    
    for step in range(opt.num_steps):
        optimizer.zero_grad()

        # Sample a window from the dataset 
        data_dict = dataset.sample_window(mode=opt.sampling_mode)
        
        # Track which windows we've seen
        start_idx = data_dict['start_idx']
        windows_seen.add(start_idx)
        
        # Move to device
        left_hand = data_dict['left_hand'].to(device)
        right_hand = data_dict['right_hand'].to(device)
        object_motion = data_dict['object_motion'].to(device)
        seq_len = data_dict['seq_len'].to(device)
        
        # Prepare input 
        hand_poses = torch.cat([left_hand, right_hand], dim=-1)  # [1, T, 18/24]
        
        # Generate padding mask
        actual_seq_len = seq_len + 1
        tmp_mask = torch.arange(opt.window+1, device=device).expand(1, opt.window+1) < actual_seq_len[:, None].repeat(1, opt.window+1)
        padding_mask = tmp_mask[:, None, :]

        with autocast(enabled=True):
            loss = diffusion_model(object_motion, hand_poses, padding_mask=padding_mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Evaluate model periodically
        if step % 1000 == 0 and step > 0:
            print(f"\n  Evaluating model at step {step}...")
            eval_mean_error, eval_max_error = evaluate_model(diffusion_model, dataset, device, num_eval_windows=5)
            
            print(f"  Evaluation - Mean position error: {eval_mean_error:.4f}m, Max error: {eval_max_error:.4f}m")
            
            # Track best model based on evaluation metrics
            if eval_mean_error < best_eval_error:
                best_eval_error = eval_mean_error
                best_step = step
                # Save best model state 
                best_model_state = {
                    'model': diffusion_model.state_dict().copy(),
                    'optimizer': optimizer.state_dict().copy(),
                    'scaler': scaler.state_dict().copy(),
                    'step': step,
                    'loss': loss.item(),
                    'eval_mean_error': eval_mean_error,
                    'eval_max_error': eval_max_error
                }
                print(f"  New best model at step {step} with eval error {eval_mean_error:.4f}m (previous: {best_eval_error:.4f}m)")
            
            if opt.use_wandb:
                wandb.log({
                    "eval/mean_position_error": eval_mean_error,
                    "eval/max_position_error": eval_max_error,
                    "eval/best_mean_error": best_eval_error,
                    "eval/best_step": best_step
                }, step=step)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}, Best eval error: {best_eval_error:.4f}m (step {best_step})")
            
            if opt.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/best_eval_error": best_eval_error,
                    "train/best_step": best_step,
                }, step=step)

        # Save checkpoint
        if step % 1000 == 0:
            checkpoint = {
                'step': step,
                'model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'dataset_info': {
                    'demo_id': dataset.demo_id,
                    'object_id': dataset.target_object_id,
                    'window_size': opt.window,
                    'full_length': dataset.full_length,
                    'num_windows': len(dataset.window_data)
                }
            }
            torch.save(checkpoint, os.path.join(wdir, f'model-{step}.pt'))

    print("Training completed!")
    print(f"Best model: step {best_step} with eval error {best_eval_error:.4f}m")

    # Save best model
    if best_model_state is not None:
        best_model_state['dataset_info'] = {
            'demo_id': dataset.demo_id,
            'object_id': dataset.target_object_id,
            'window_size': opt.window,
            'full_length': dataset.full_length
        }
        best_model_path = os.path.join(wdir, 'best_model.pt')
        torch.save(best_model_state, best_model_path)
        print(f"Saved best model to {best_model_path}")

        # Load best model for inference
        print(f"Loading best model (step {best_step}, eval error {best_eval_error:.4f}m) for inference...")
        diffusion_model.load_state_dict(best_model_state['model'])
        
        # Final comprehensive evaluation
        print(f"\nFinal evaluation on best model...")
        final_mean_error, final_max_error = evaluate_model(diffusion_model, dataset, device, num_eval_windows=len(dataset.window_data))
        print(f"Final evaluation results:")
        print(f"  Mean position error: {final_mean_error:.4f} meters")
        print(f"  Max position error: {final_max_error:.4f} meters")
        print(f"  Evaluated on {len(dataset.window_data)} windows")
    else:
        print("No best model found, using final model for inference")
        final_mean_error, final_max_error = evaluate_model(diffusion_model, dataset, device, num_eval_windows=len(dataset.window_data))
        print(f"Final model evaluation:")
        print(f"  Mean position error: {final_mean_error:.4f} meters")
        print(f"  Max position error: {final_max_error:.4f} meters")

    # Test sampling - generate full trajectory using sliding windows
    print("\nTesting sampling on full trajectory...")
    sampled_motion_full = sample_full_trajectory_inpaint(
        diffusion_model, dataset, device, 
        window_size=opt.window, 
        overlap=opt.window//2
    )
    
    # Save results - full trajectory
    save_path = os.path.join(opt.save_dir, 'sampled_motion.npy')
    np.save(save_path, sampled_motion_full.unsqueeze(0).numpy())  # Add batch dim for compatibility
    print(f"Saved full sampled motion to {save_path}")

    # Save input hand poses and ground truth - full trajectory
    hand_poses_full = torch.cat([dataset.left_hand_full, dataset.right_hand_full], dim=-1)
    hand_poses_path = os.path.join(opt.save_dir, 'input_hand_poses.npy')
    np.save(hand_poses_path, hand_poses_full.unsqueeze(0).numpy())
    print(f"Saved full input hand poses to {hand_poses_path}")
    
    gt_object_path = os.path.join(opt.save_dir, 'ground_truth_object.npy')
    np.save(gt_object_path, dataset.object_motion_full.unsqueeze(0).numpy())
    print(f"Saved full ground truth object motion to {gt_object_path}")

    # Save training summary
    summary_path = os.path.join(opt.save_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n")
        f.write(f"Total steps: {opt.num_steps}\n")
        f.write(f"Best step: {best_step}\n")
        f.write(f"Best eval error: {best_eval_error:.4f}m\n")
        f.write(f"Final loss: {loss.item():.6f}\n")
        f.write(f"Final mean position error: {final_mean_error:.4f}m\n")
        f.write(f"Final max position error: {final_max_error:.4f}m\n")
        f.write(f"Demo ID: {dataset.demo_id}\n")
        f.write(f"Object ID: {dataset.target_object_id}\n")
        f.write(f"Trajectory length: {dataset.full_length} frames\n")
        f.write(f"Window size: {opt.window}\n")
        f.write(f"Use velocity: {opt.use_velocity}\n")
        f.write(f"Data dimension: {dataset.pose_dim}D\n")
    print(f"Saved training summary to {summary_path}")

    if opt.use_wandb:
        wandb.log({
            "final/best_eval_error": best_eval_error,
            "final/best_step": best_step,
            "final/final_loss": loss.item(),
            "final/final_mean_error": final_mean_error,
            "final/final_max_error": final_max_error
        })
        wandb.finish()

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/processed_data_with_velocity.pkl', help='Path to processed data pickle file')
    parser.add_argument('--demo_id', type=str, default=None, help='Specific demo ID to use (if None, use first available)')
    parser.add_argument('--target_object_id', type=str, default=None, help='Specific object ID to track (if None, use first available)')
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['random', 'sequential'], 
                        help='Window sampling mode: random (better performance) or sequential')
    parser.add_argument('--use_velocity', action='store_true', default=False, help='Use 12D data with velocity (default: False)')
    
    # Model parameters
    parser.add_argument('--window', type=int, default=128, help='Training window size and model max sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='Transformer model dimension')
    parser.add_argument('--n_dec_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_k', type=int, default=64, help='Key dimension')
    parser.add_argument('--d_v', type=int, default=64, help='Value dimension')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--guidance_weight', type=float, default=0.0, help='Guidance weight')
    parser.add_argument('--guidance_mode', action='store_true', help='Enable guidance mode')
    
    # Logging parameters
    parser.add_argument('--save_dir', type=str, default='runs/overfit_ddim', help='Directory to save results')
    parser.add_argument('--wandb_pj_name', type=str, default='egorecon', help='Wandb project name')
    parser.add_argument('--entity', type=str, default='egorecon', help='Wandb entity name')
    parser.add_argument('--exp_name', type=str, default='overfit_with_velocity', help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_overfit(opt, device) 