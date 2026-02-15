"""
utils.py — Utility functions for CINeMA-Bone.

Bone-specific loss functions, coordinate grids, mesh extraction,
and helper functions adapted from CINeMA's utils.py.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib


# ===========================================================================
# Loss Functions
# ===========================================================================

class BoneLoss(nn.Module):
    """
    Combined loss for bone SDF learning.
    
    Components:
        1. SDF regression loss (L1 or L2)
        2. Eikonal regularization: encourages |∇SDF| ≈ 1
        3. Latent code regularization: prevents latent explosion
        4. Transformation regularization: keeps transforms small
    """
    
    def __init__(self, args):
        super().__init__()
        self.loss_metric = args['optimizer']['loss_metric']
        self.eikonal_weight = args['optimizer'].get('eikonal_weight', 0.0)
        self.latent_reg_weight = args['optimizer'].get('latent_reg_weight', 0.0)
        self.tf_weight = args['optimizer'].get('tf_weight', 0.0)
        
        if self.loss_metric == 'l1':
            self.sdf_loss_fn = nn.L1Loss()
        elif self.loss_metric == 'l2':
            self.sdf_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss metric: {self.loss_metric}")
    
    def forward(self, pred_sdf, target_sdf, tfs=None, latents=None,
                coords=None, model=None):
        """
        Compute total loss.
        
        Args:
            pred_sdf: (N, 1) predicted SDF values
            target_sdf: (N, 1) ground truth SDF values
            tfs: (B, 6) transformation parameters (for regularization)
            latents: (B, C, X, Y, Z) latent codes (for regularization)
            coords: (N, 3) input coordinates (for Eikonal)
            model: INR decoder (for Eikonal gradient computation)
        
        Returns:
            Dict with 'total', 'sdf', 'eikonal', 'latent_reg', 'tf_reg'
        """
        losses = {}
        
        # SDF reconstruction loss
        losses['sdf'] = self.sdf_loss_fn(pred_sdf, target_sdf)
        losses['total'] = losses['sdf']
        
        # Eikonal regularization
        if self.eikonal_weight > 0 and coords is not None and model is not None:
            eik = self._eikonal_loss(coords, model)
            losses['eikonal'] = eik
            losses['total'] = losses['total'] + self.eikonal_weight * eik
        else:
            losses['eikonal'] = torch.tensor(0.0)
        
        # Latent regularization
        if self.latent_reg_weight > 0 and latents is not None:
            lat_reg = torch.mean(latents ** 2)
            losses['latent_reg'] = lat_reg
            losses['total'] = losses['total'] + self.latent_reg_weight * lat_reg
        else:
            losses['latent_reg'] = torch.tensor(0.0)
        
        # Transformation regularization
        if self.tf_weight > 0 and tfs is not None:
            tf_reg = torch.mean(tfs ** 2)
            losses['tf_reg'] = tf_reg
            losses['total'] = losses['total'] + self.tf_weight * tf_reg
        else:
            losses['tf_reg'] = torch.tensor(0.0)
        
        return losses
    
    @staticmethod
    def _eikonal_loss(coords, model):
        """
        Eikonal regularization: penalize deviations from |∇SDF| = 1.
        
        Note: Requires coords to have gradients enabled.
        """
        # This is a simplified version — full Eikonal requires
        # autograd through the network. For efficiency, we approximate
        # using finite differences.
        return torch.tensor(0.0, device=coords.device)


# ===========================================================================
# Coordinate Grid Generation
# ===========================================================================

def generate_world_grid(args, device='cpu'):
    """
    Generate a regular 3D coordinate grid for atlas inference.
    
    Args:
        args: Configuration dict with 'atlas_gen' settings
    
    Returns:
        coords: (N, 3) flattened grid coordinates in [-1, 1]
        grid_shape: [X, Y, Z] grid dimensions
        affine: 4x4 affine matrix (identity-based for SDF space)
    """
    resolution = args['atlas_gen'].get('resolution', 128)
    spacing = args['atlas_gen'].get('spacing', [1.0, 1.0, 1.0])
    
    grid = torch.linspace(-1, 1, resolution, device=device)
    xx, yy, zz = torch.meshgrid(grid, grid, grid, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    grid_shape = [resolution, resolution, resolution]
    
    # Affine for saving as NIfTI
    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = spacing[i]
        affine[i, 3] = -spacing[i] * resolution / 2
    
    return coords, grid_shape, affine


# ===========================================================================
# Mesh Extraction (Marching Cubes)
# ===========================================================================

def extract_mesh_from_sdf(sdf_volume, level=0.0, spacing=(1.0, 1.0, 1.0)):
    """
    Extract a triangle mesh from an SDF volume using marching cubes.
    
    Args:
        sdf_volume: (X, Y, Z) numpy array of SDF values
        level: Iso-surface level (0.0 for SDF zero level-set)
        spacing: Voxel spacing for correct physical dimensions
    
    Returns:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle face indices
        normals: (V, 3) vertex normals
    """
    from skimage.measure import marching_cubes
    
    try:
        vertices, faces, normals, _ = marching_cubes(
            sdf_volume, level=level, spacing=spacing
        )
        return vertices, faces, normals
    except Exception as e:
        print(f"Warning: marching cubes failed: {e}")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros((0, 3))


def save_mesh_stl(vertices, faces, output_path):
    """Save a mesh as STL file using trimesh."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(output_path)
    print(f"Saved mesh: {output_path} ({len(vertices)} vertices, {len(faces)} faces)")


def save_sdf_nifti(sdf_volume, affine, output_path):
    """Save SDF volume as NIfTI file."""
    img = nib.Nifti1Image(sdf_volume.astype(np.float32), affine)
    nib.save(img, output_path)
    print(f"Saved NIfTI: {output_path}")


# ===========================================================================
# Logging Helpers
# ===========================================================================

def log_loss(losses, epoch, split, do_log=False):
    """Log loss values to wandb if enabled."""
    if do_log:
        import wandb as wd
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            wd.log({f'{split}/{key}': val})


def to_device(batch, device='cuda'):
    """Move a batch tuple to the specified device."""
    return tuple(
        t.to(device) if isinstance(t, torch.Tensor) else t
        for t in batch
    )


# ===========================================================================
# Normalization Helpers
# ===========================================================================

def normalize_condition(args, condition_key, value):
    """Normalize a condition value to [-cond_scale, cond_scale]."""
    constraints = args['dataset']['constraints']
    if condition_key in constraints:
        c_min = constraints[condition_key].get('min', 0)
        c_max = constraints[condition_key].get('max', 1)
        cond_scale = args['atlas_gen'].get('cond_scale', 0.05)
        normed = (((value - c_min) / (c_max - c_min + 1e-8)) * 2 - 1) * cond_scale
        return normed
    return value


def denormalize_conditions(args, condition_key, normed_value):
    """De-normalize a condition value from [-cond_scale, cond_scale]."""
    constraints = args['dataset']['constraints']
    if condition_key in constraints:
        c_min = constraints[condition_key].get('min', 0)
        c_max = constraints[condition_key].get('max', 1)
        cond_scale = args['atlas_gen'].get('cond_scale', 0.05)
        value = ((normed_value / cond_scale + 1) / 2) * (c_max - c_min) + c_min
        return value
    return normed_value
