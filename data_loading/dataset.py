"""
dataset.py — PyTorch Dataset for bone SDF data.

Handles loading pre-voxelized SDF volumes and sampling (coordinate, SDF value)
pairs for training the INR bone atlas. Follows the CINeMA Data class interface.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class BoneDataset(Dataset):
    """
    PyTorch Dataset for bone SDF training data.
    
    Each subject is a pre-voxelized SDF volume (.npy file) along with metadata.
    Returns coordinate-value pairs for INR training, following the CINeMA interface:
        (coords, values, conditions, idx_df)
    
    Args:
        args: Configuration dictionary
        split: 'train' or 'val'
        subject_list: Optional list of subject dicts (overrides auto-discovery)
    """
    
    def __init__(self, args, split='train', subject_list=None):
        self.args = args
        self.split = split
        self.dataset_args = args['dataset']
        self.world_bbox = np.array(self.dataset_args['world_bbox'], dtype=np.float32)
        self.n_surface_ratio = self.dataset_args.get('surface_sample_ratio', 0.5)
        
        if subject_list is not None:
            self.subjects = subject_list
        else:
            self.subjects = self._discover_subjects()
        
        # Apply train/val split if subject_ids provided
        if 'subject_ids' in self.dataset_args and self.dataset_args['subject_ids'] is not None:
            split_ids = self.dataset_args['subject_ids'].get(split, None)
            if split_ids is not None:
                self.subjects = [s for s in self.subjects if s['subject_id'] in split_ids]
        
        print(f"[BoneDataset] {split}: {len(self.subjects)} subjects loaded")
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        """
        Returns:
            coords: (N, 3) normalized coordinates in [-1, 1]
            values: (N, 1) SDF values at those coordinates
            conditions: (N, C) condition vector (e.g., fracture type)
            idx_df: (N, 1) subject index for latent code lookup
        """
        subject = self.subjects[idx]
        sdf_volume = self._load_sdf(subject['sdf_path'])
        
        # Sample coordinates and SDF values
        coords, values = self._sample_coords_and_values(sdf_volume)
        
        # Build condition vector
        conditions = self._build_conditions(subject)
        
        coords = torch.tensor(coords, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        conditions = conditions.unsqueeze(0).expand(coords.shape[0], -1)
        idx_df = torch.tensor(idx, dtype=torch.int32).unsqueeze(0).expand(coords.shape[0], 1)
        
        return coords, values, conditions, idx_df
    
    def collate_fn(self, batch, shuffle=True):
        """Custom collate: concatenate all subjects and shuffle coordinates."""
        coords = torch.cat([b[0] for b in batch], dim=0)
        values = torch.cat([b[1] for b in batch], dim=0)
        conditions = torch.cat([b[2] for b in batch], dim=0)
        idx_df = torch.cat([b[3] for b in batch], dim=0)
        
        if shuffle:
            perm = torch.randperm(coords.shape[0])
            coords = coords[perm]
            values = values[perm]
            conditions = conditions[perm]
            idx_df = idx_df[perm]
        
        return coords, values, conditions, idx_df
    
    def _load_sdf(self, path):
        """Load pre-computed SDF volume."""
        sdf = np.load(path)
        return sdf.astype(np.float32)
    
    def _sample_coords_and_values(self, sdf_volume):
        """
        Sample (coordinate, SDF value) pairs from the volume.
        
        Uses a hybrid strategy:
        1. Surface-biased sampling: sample near the zero level-set (bone surface)
           for efficient training on sparse bone geometry
        2. Uniform sampling: sample uniformly in the bounding box for global coverage
        
        Returns:
            coords: (N, 3) in [-1, 1]
            values: (N, 1)
        """
        resolution = sdf_volume.shape[0]
        n_total = self.args.get('n_coords_per_subject', 50000)
        n_surface = int(n_total * self.n_surface_ratio)
        n_uniform = n_total - n_surface
        
        # Create coordinate grid for this volume
        grid = np.linspace(-1, 1, resolution)
        
        # --- Surface-biased sampling ---
        # Find voxels near the zero level-set (|SDF| < threshold)
        surface_threshold = self.dataset_args.get('surface_threshold', 0.05)
        near_surface = np.abs(sdf_volume) < surface_threshold
        surface_indices = np.argwhere(near_surface)
        
        if len(surface_indices) > 0:
            # Sample from near-surface voxels
            if len(surface_indices) > n_surface:
                choice = np.random.choice(len(surface_indices), n_surface, replace=False)
            else:
                choice = np.random.choice(len(surface_indices), n_surface, replace=True)
            
            surf_idx = surface_indices[choice]
            surf_coords = np.stack([grid[surf_idx[:, i]] for i in range(3)], axis=-1)
            # Add small noise for sub-voxel variation
            noise = np.random.uniform(-0.5/resolution, 0.5/resolution, surf_coords.shape)
            surf_coords += noise
            surf_coords = np.clip(surf_coords, -1, 1)
            # Get SDF values via trilinear interpolation (nearest for speed)
            surf_values = sdf_volume[surf_idx[:, 0], surf_idx[:, 1], surf_idx[:, 2]]
        else:
            # Fallback: all uniform if no surface found
            n_uniform = n_total
            n_surface = 0
            surf_coords = np.zeros((0, 3), dtype=np.float32)
            surf_values = np.zeros((0,), dtype=np.float32)
        
        # --- Uniform sampling ---
        uniform_coords = np.random.uniform(-1, 1, (n_uniform, 3)).astype(np.float32)
        # Map to grid indices for value lookup
        uniform_idx = ((uniform_coords + 1) / 2 * (resolution - 1)).astype(int)
        uniform_idx = np.clip(uniform_idx, 0, resolution - 1)
        uniform_values = sdf_volume[uniform_idx[:, 0], uniform_idx[:, 1], uniform_idx[:, 2]]
        
        # Combine
        if n_surface > 0:
            coords = np.concatenate([surf_coords, uniform_coords], axis=0)
            values = np.concatenate([surf_values, uniform_values], axis=0)
        else:
            coords = uniform_coords
            values = uniform_values
        
        return coords.astype(np.float32), values[:, None].astype(np.float32)
    
    def _build_conditions(self, subject):
        """Build condition vector from subject metadata."""
        conditions = []
        cond_config = self.dataset_args.get('conditions', {})
        
        for key, enabled in cond_config.items():
            if enabled and key in subject:
                conditions.append(float(subject[key]))
        
        if len(conditions) == 0:
            return torch.zeros(0, dtype=torch.float32)
        
        return torch.tensor(conditions, dtype=torch.float32)
    
    def _discover_subjects(self):
        """
        Auto-discover subjects from the processed data directory.
        Looks for .npy SDF files in the processed_data/ subdirectory.
        """
        processed_dir = self.dataset_args.get('processed_dir', 
                                               os.path.join(self.dataset_args['data_dir'], 'processed'))
        
        subjects = []
        if not os.path.exists(processed_dir):
            print(f"  Warning: processed directory {processed_dir} does not exist. "
                  f"Run prepare_data.py first.")
            return subjects
        
        npy_files = sorted(glob.glob(os.path.join(processed_dir, '*.npy')))
        for i, f in enumerate(npy_files):
            basename = os.path.splitext(os.path.basename(f))[0]
            subjects.append({
                'subject_id': basename,
                'sdf_path': f,
                'type': 'patient' if 'patient' in basename else ('gt' if 'gt' in basename.lower() else 'template'),
                'n_fragments': 1,
            })
        
        return subjects
    
    def get_condition_values(self, condition_key, normed=True, device=None):
        """
        Get all values of a condition_key from subjects.
        Compatible with CINeMA interface.
        """
        values = []
        for s in self.subjects:
            values.append(s.get(condition_key, 0.0))
        values = torch.tensor(values, dtype=torch.float32, device=device)
        return values, torch.arange(len(self.subjects), device=device)
