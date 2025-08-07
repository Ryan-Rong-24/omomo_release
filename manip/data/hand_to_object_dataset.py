#!/usr/bin/env python3

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms


def load_pickle(path):
    """Load and return the object stored in a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def to_tensor(array, dtype=torch.float32):
    """Convert array to tensor with specified dtype."""
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


class HandToObjectDataset(Dataset):
    """
    Dataset for hand-to-object trajectory denoising.
    
    Loads HOT3D processed data and creates windows for training.
    Inputs: Left hand trajectory + Right hand trajectory (both T x 9 or T x 12, default: 9D)
    Output: Object trajectory (T x 9 or T x 12, default: 9D)
    """
    
    def __init__(
        self,
        data_path,
        window_size=120,
        use_velocity=False,  # Use 12D format (pos + vel + rot) vs 9D (pos + rot) - default is 9D
        single_demo=None,   # For overfitting: specify demo_id
        single_object=None, # For overfitting: specify object_id  
        motion_threshold=0.005,  # Threshold for considering motion vs stationary
        sampling_strategy='balanced',  # 'balanced', 'motion_only', 'random'
        min_motion_frames=10,  # Minimum consecutive motion frames to consider
        augment=False,  # Whether in training mode (for data augmentation)
        split='train',  # 'train', 'val', or 'all'
        val_split_ratio=0.2,  # Fraction of windows to use for validation
        split_seed=42,  # Seed for reproducible splits
    ):
        self.data_path = data_path
        self.window_size = window_size
        self.use_velocity = use_velocity
        self.motion_threshold = motion_threshold
        self.sampling_strategy = sampling_strategy
        self.min_motion_frames = min_motion_frames
        self.augment = augment  # Add training mode flag
        self.split = split
        self.val_split_ratio = val_split_ratio
        self.split_seed = split_seed
        
        # Load processed data
        print(f"Loading data from {data_path}...")
        self.processed_data = load_pickle(data_path)
        print(f"Found {len(self.processed_data)} demonstrations")
        
        # Choose data format
        self.pose_key = 'poses_12d' if use_velocity else 'poses_9d'
        self.pose_dim = 12 if use_velocity else 9
        print(f"Using {self.pose_key} format (dimension: {self.pose_dim})")
        
        # Filter for single demo/object if specified (for overfitting)
        if single_demo or single_object:
            self.processed_data = self._filter_data(single_demo, single_object)
            print(f"Filtered to {len(self.processed_data)} demonstrations for overfitting")
        
        # Create windows
        self.windows = self._create_windows()
        
        # Apply train/validation split
        self.windows = self._apply_train_val_split()
        print(f"Created {len(self.windows)} {self.split} windows")
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
    def _filter_data(self, single_demo=None, single_object=None):
        """Filter data for overfitting on specific demo/object."""
        filtered_data = {}
        
        for demo_id, demo_data in self.processed_data.items():
            if single_demo and demo_id != single_demo:
                continue
                
            if single_object and 'objects' in demo_data:
                # Keep only the specified object
                if single_object in demo_data['objects']:
                    filtered_demo = {
                        'left_hand': demo_data.get('left_hand'),
                        'right_hand': demo_data.get('right_hand'),
                        'objects': {single_object: demo_data['objects'][single_object]}
                    }
                    filtered_data[demo_id] = filtered_demo
            else:
                filtered_data[demo_id] = demo_data
                
        return filtered_data
    
    def _create_windows(self):
        """Create sliding windows from trajectory data."""
        windows = []
        
        for demo_id, demo_data in self.processed_data.items():
            if 'objects' not in demo_data:
                continue
                
            left_hand = demo_data.get('left_hand', {}).get(self.pose_key, None)
            right_hand = demo_data.get('right_hand', {}).get(self.pose_key, None)
            
            if left_hand is None or right_hand is None:
                print(f"Warning: Missing hand data for demo {demo_id}")
                continue
                
            for obj_id, obj_data in demo_data['objects'].items():
                object_traj = obj_data.get(self.pose_key, None)
                if object_traj is None:
                    continue
                    
                # Find the overlapping time range for all trajectories
                min_len = min(len(left_hand), len(right_hand), len(object_traj))
                
                if min_len < self.window_size:
                    print(f"Warning: Trajectory too short for demo {demo_id}, obj {obj_id}: {min_len}")
                    continue
                
                # Trim all trajectories to same length
                left_hand_trimmed = left_hand[:min_len]
                right_hand_trimmed = right_hand[:min_len]
                object_traj_trimmed = object_traj[:min_len]
                
                # Create sliding windows
                for start_idx in range(0, min_len - self.window_size + 1, self.window_size // 2):
                    end_idx = start_idx + self.window_size
                    
                    window_data = {
                        'demo_id': demo_id,
                        'object_id': obj_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'left_hand': left_hand_trimmed[start_idx:end_idx],
                        'right_hand': right_hand_trimmed[start_idx:end_idx], 
                        'object': object_traj_trimmed[start_idx:end_idx],
                    }
                    
                    # Check if this is a motion window
                    is_motion = self._is_motion_window(object_traj_trimmed[start_idx:end_idx])
                    window_data['is_motion'] = is_motion
                    
                    windows.append(window_data)
        
        # Apply sampling strategy
        windows = self._apply_sampling_strategy(windows)
        
        return windows
    
    def _is_motion_window(self, object_traj):
        """Determine if this window contains significant object motion."""
        if len(object_traj) < 2:
            return False
            
        # Extract positions (first 3 dimensions)
        positions = object_traj[:, :3]
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        # Check if there are consecutive frames above threshold
        above_threshold = velocities > self.motion_threshold
        
        # Find longest consecutive sequence
        max_consecutive = 0
        current_consecutive = 0
        
        for is_moving in above_threshold:
            if is_moving:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive >= self.min_motion_frames
    
    def _apply_sampling_strategy(self, windows):
        """Apply sampling strategy to balance motion vs stationary windows."""
        motion_windows = [w for w in windows if w['is_motion']]
        stationary_windows = [w for w in windows if not w['is_motion']]
        
        print(f"Original windows: {len(motion_windows)} motion, {len(stationary_windows)} stationary")
        
        if self.sampling_strategy == 'motion_only':
            return motion_windows
        elif self.sampling_strategy == 'balanced':
            # Balance by taking equal numbers, up to the smaller set size
            min_count = min(len(motion_windows), len(stationary_windows))
            if min_count == 0:
                return windows  # Return all if one category is empty
            
            # Randomly sample to balance
            np.random.seed(42)  # For reproducibility
            motion_indices = np.random.choice(len(motion_windows), min_count, replace=False)
            stationary_indices = np.random.choice(len(stationary_windows), min_count, replace=False)
            
            balanced_windows = [motion_windows[i] for i in motion_indices]
            balanced_windows.extend([stationary_windows[i] for i in stationary_indices])
            
            print(f"Balanced to: {len(balanced_windows)} windows ({min_count} motion, {min_count} stationary)")
            return balanced_windows
        else:  # 'random'
            return windows
    
    def _apply_train_val_split(self):
        """Apply train/validation split to windows."""
        if self.split == 'all':
            return self.windows
            
        # Use seed for reproducible splits
        np.random.seed(self.split_seed)
        
        # Shuffle windows for random split
        indices = np.arange(len(self.windows))
        np.random.shuffle(indices)
        
        # Split indices
        n_val = int(len(self.windows) * self.val_split_ratio)
        if self.split == 'val':
            selected_indices = indices[:n_val]
        else:  # 'train'
            selected_indices = indices[n_val:]
        
        # Return selected windows
        selected_windows = [self.windows[i] for i in selected_indices]
        
        print(f"Split info: {len(selected_windows)}/{len(self.windows)} windows for {self.split}")
        return selected_windows
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics for the dataset."""
        all_left_hand = []
        all_right_hand = []
        all_objects = []
        
        for window in self.windows:
            all_left_hand.append(window['left_hand'])
            all_right_hand.append(window['right_hand'])
            all_objects.append(window['object'])
        
        if not all_left_hand:
            print("Warning: No windows found for normalization")
            self.stats = {}
            return
            
        # Stack all data
        all_left_data = np.vstack(all_left_hand)  # [N*T, D]
        all_right_data = np.vstack(all_right_hand)  # [N*T, D]
        all_object_data = np.vstack(all_objects)  # [N*T, D]
        
        # Use standard normalization approach
        left_mean = np.mean(all_left_data, axis=0)
        left_std = np.std(all_left_data, axis=0)
        right_mean = np.mean(all_right_data, axis=0)
        right_std = np.std(all_right_data, axis=0)
        object_mean = np.mean(all_object_data, axis=0)
        object_std = np.std(all_object_data, axis=0)
        
        # Ensure minimum std to prevent division by very small values
        min_std = 1e-6
        left_std = np.maximum(left_std, min_std)
        right_std = np.maximum(right_std, min_std)
        object_std = np.maximum(object_std, min_std)
        
        self.stats = {
            'left_hand_mean': left_mean,
            'left_hand_std': left_std,
            'right_hand_mean': right_mean,
            'right_hand_std': right_std,
            'object_mean': object_mean,
            'object_std': object_std,
        }
        
        print("Computed normalization statistics")
        print(f"Object trajectory - Mean range: [{object_mean.min():.3f}, {object_mean.max():.3f}]")
        print(f"Object trajectory - Std range: [{object_std.min():.3f}, {object_std.max():.3f}]")
    
    def normalize_data(self, data, data_type):
        """Normalize data using computed statistics."""
        if not self.stats:
            return data
            
        mean = self.stats[f'{data_type}_mean']
        std = self.stats[f'{data_type}_std']
        
        return (data - mean) / std
    
    def denormalize_data(self, data, data_type):
        """Denormalize data using computed statistics."""
        if not self.stats:
            return data
            
        mean = self.stats[f'{data_type}_mean']
        std = self.stats[f'{data_type}_std']
        
        return data * std + mean
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Convert to tensors
        left_hand = to_tensor(window['left_hand'])  # [T, D]
        right_hand = to_tensor(window['right_hand'])  # [T, D]
        object_traj = to_tensor(window['object'])  # [T, D]
        
        # Apply data augmentation during training
        if self.augment:
            left_hand, right_hand, object_traj = self._augment_data(left_hand, right_hand, object_traj)
        
        # Normalize
        left_hand_norm = to_tensor(self.normalize_data(left_hand.numpy(), 'left_hand'))
        right_hand_norm = to_tensor(self.normalize_data(right_hand.numpy(), 'right_hand'))
        object_norm = to_tensor(self.normalize_data(object_traj.numpy(), 'object'))
        
        # Concatenate hand inputs for conditioning
        condition = torch.cat([left_hand_norm, right_hand_norm], dim=-1)  # [T, 2*D]
        
        return {
            'condition': condition,  # [T, 2*D] - left and right hand trajectories
            'target': object_norm,   # [T, D] - object trajectory to denoise
            'target_raw': object_traj,  # [T, D] - unnormalized for evaluation
            'demo_id': window['demo_id'],
            'object_id': str(window['object_id']),
            'is_motion': window['is_motion']
        }
    
    def _augment_data(self, left_hand, right_hand, object_traj):
        """Apply data augmentation to improve training stability."""
        # Small random noise to positions (first 3 dimensions)
        if np.random.random() < 0.3:  # 30% chance
            noise_scale = 0.001  # 1mm noise
            pos_noise = torch.randn_like(left_hand[:, :3]) * noise_scale
            left_hand[:, :3] += pos_noise
            right_hand[:, :3] += pos_noise
            object_traj[:, :3] += pos_noise
        
        # Small random scaling to rotations (if using rotation data)
        if self.pose_dim >= 9 and np.random.random() < 0.2:  # 20% chance
            scale_factor = 0.95 + 0.1 * torch.rand(1).item()  # Scale between 0.95-1.05
            if self.use_velocity:
                # 12D format: pos(3) + vel(3) + rot(6)
                left_hand[:, 6:12] *= scale_factor
                right_hand[:, 6:12] *= scale_factor
                object_traj[:, 6:12] *= scale_factor
            else:
                # 9D format: pos(3) + rot(6)
                left_hand[:, 3:9] *= scale_factor
                right_hand[:, 3:9] *= scale_factor
                object_traj[:, 3:9] *= scale_factor
        
        return left_hand, right_hand, object_traj


# Convenience function for creating datasets
def create_hand_to_object_dataset(
    data_path,
    window_size=120,
    use_velocity=False,
    single_demo=None,
    single_object=None,
    motion_threshold=0.01,
    sampling_strategy='balanced',
    split='train',
    val_split_ratio=0.2,
    augment=False
):
    """
    Convenience function to create HandToObjectDataset.
    
    Args:
        data_path: Path to processed data pickle file
        window_size: Size of trajectory windows
        use_velocity: Whether to use 12D (with velocity) or 9D format (default: 9D)
        single_demo: For overfitting, specify single demo ID
        single_object: For overfitting, specify single object ID
        motion_threshold: Threshold for motion detection
        sampling_strategy: 'balanced', 'motion_only', or 'random'
        split: 'train', 'val', or 'all'
        val_split_ratio: Fraction of data for validation
        augment: Whether to apply data augmentation
    """
    return HandToObjectDataset(
        data_path=data_path,
        window_size=window_size,
        use_velocity=use_velocity,
        single_demo=single_demo,
        single_object=single_object,
        motion_threshold=motion_threshold,
        sampling_strategy=sampling_strategy,
        split=split,
        val_split_ratio=val_split_ratio,
        augment=augment
    )


if __name__ == "__main__":
    # Test the dataset (default 9D format)
    dataset = create_hand_to_object_dataset(
        data_path="data/processed_data.pkl",
        window_size=120,
        use_velocity=False,  # Default: 9D format (pos + rot)
        single_demo="P0001_10a27bf7",  # For overfitting test
        single_object="37787722328019"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shapes:")
        print(f"  Condition: {sample['condition'].shape}")  # [T, 2*D]
        print(f"  Target: {sample['target'].shape}")  # [T, D]
        print(f"  Demo ID: {sample['demo_id']}")
        print(f"  Object ID: {sample['object_id']}")
        print(f"  Is motion: {sample['is_motion']}") 