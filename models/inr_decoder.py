"""
inr_decoder.py — Bone INR Decoder for SDF representation.

Adapted from CINeMA's INR_Decoder. Simplified for single-channel SDF output
(instead of multi-modal brain tissue). Retains rigid transformation and
spatial modulation mechanisms for the auto-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.siren import Siren


def embed2affine(params):
    """
    Convert 6-DOF embedding to rotation matrix + translation vector.
    
    Uses the exponential map (Rodrigues' formula) for rotation.
    
    Args:
        params: (N, 6) tensor — first 3 are rotation (axis-angle), last 3 are translation
    
    Returns:
        R: (N, 3, 3) rotation matrices
        t: (N, 3) translation vectors
    """
    rot_params = params[:, :3]
    t = params[:, 3:6]
    
    # Axis-angle to rotation matrix via Rodrigues
    theta = torch.norm(rot_params, dim=-1, keepdim=True).unsqueeze(-1)  # (N, 1, 1)
    theta = theta.clamp(min=1e-8)
    
    k = rot_params / theta.squeeze(-1)  # Normalized axis (N, 3)
    
    K = torch.zeros(params.shape[0], 3, 3, device=params.device, dtype=params.dtype)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    
    I = torch.eye(3, device=params.device, dtype=params.dtype).unsqueeze(0)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.bmm(K, K)
    
    return R, t


class BoneINRDecoder(nn.Module):
    """
    Implicit Neural Representation decoder for bone SDF.
    
    Given 3D coordinates and a per-subject latent code, outputs SDF values.
    Includes learnable rigid transformations for per-subject alignment.
    
    Architecture:
        coords → [rigid transform] → SIREN(coords, latent_modulation) → SDF
    """
    
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        args_inr = args['inr_decoder']
        self.device = device
        self.out_dim = sum(args_inr['out_dim'])  # Should be 1 for SDF
        
        # Spatial modulation CNN (optional)
        self.modulator = Modulator(
            args_inr['latent_dim'],
            kernel_size=args_inr.get('cnn_kernel_size', 0)
        )
        
        # Condition dimensions (may be 0 if no explicit conditions)
        cond_dims = args_inr.get('cond_dims', 0)
        
        # SIREN network
        self.sr_net = Siren(
            in_size=args_inr['in_dim'],
            lat_size=args_inr['latent_dim'][0] + cond_dims,
            out_size=self.out_dim,
            hidden_size=args_inr['hidden_size'],
            num_layers=args_inr['num_hidden_layers'],
            f_om=args_inr['omega'][0],
            h_om=args_inr['omega'][1],
            outermost_linear=True,
            modulated_layers=args_inr['modulated_layers']
        )
    
    def forward(self, coords, latent_vecs, condition_vecs=None,
                tfs=None, idcs_df=None):
        """
        Forward pass: coordinates → SDF values.
        
        Args:
            coords: (N, 3) world coordinates
            latent_vecs: (B, C, X, Y, Z) per-subject latent codes
            condition_vecs: (N, C_cond) condition vectors (optional)
            tfs: (B, 6) per-subject transformation parameters
            idcs_df: (N,) subject index for each coordinate
        
        Returns:
            output: (N, 1) predicted SDF values
        """
        # Apply rigid transformation
        if tfs is not None:
            coords = self.transform(coords, tfs)
        
        # Spatial modulation of latent codes
        modulations = self.modulator(latent_vecs)[idcs_df]
        
        # Interpolate latent code at each coordinate
        modulations_interp = self.spatial_interpolation(
            coords, modulations, condition_vecs
        )
        
        # SIREN forward
        output = self.sr_net((coords, modulations_interp))
        
        return output
    
    def inference(self, coords, latent_vec, condition_vec=None,
                  img_shape=None, tfs=None, step_size=100000):
        """
        Full volume inference — generates SDF on a regular grid.
        
        Args:
            coords: (M, 3) grid coordinates
            latent_vec: (1, C, X, Y, Z) single subject latent code
            condition_vec: (C_cond,) condition vector
            img_shape: [X, Y, Z] output volume shape
            tfs: (1, 6) transformation parameters
            step_size: batch size for chunked inference
        
        Returns:
            sdf_volume: (X, Y, Z) SDF tensor
        """
        output = torch.empty((coords.shape[0], self.out_dim),
                             device=self.device)
        
        # Apply transform to all coords
        if tfs is not None:
            coords = self.transform(coords, tfs.expand(coords.shape[0], -1))
        
        # Process in chunks to avoid OOM
        for i in range(0, coords.shape[0], step_size):
            c = coords[i:i + step_size]
            idcs_df = torch.zeros(c.shape[0], dtype=torch.long,
                                  device=self.device)
            
            if condition_vec is not None:
                cv = condition_vec.expand(c.shape[0], -1)
            else:
                cv = None
            
            output[i:i + step_size] = self.forward(
                c, latent_vec, cv, idcs_df=idcs_df
            )
        
        # Reshape to volume
        if img_shape is not None:
            sdf_volume = output.squeeze(-1).reshape(img_shape)
        else:
            sdf_volume = output
        
        return sdf_volume
    
    @staticmethod
    def transform(coords, tfs, inverse=False):
        """
        Apply rigid transformation to coordinates.
        
        Args:
            coords: (N, 3) coordinates
            tfs: (N, 6) transformation parameters
            inverse: If True, apply inverse transform
        
        Returns:
            Transformed coordinates (N, 3)
        """
        R, t = embed2affine(tfs)
        
        if inverse:
            R = R.inverse()
            t = -torch.einsum('nij,nj->ni', R, t)
        
        coords = torch.einsum('nxy,ny->nx', R, coords) + t
        return coords
    
    @staticmethod
    def spatial_interpolation(coords, latents, condition_vecs=None):
        """
        Spatially interpolate the 4D latent code to 1D per coordinate.
        
        Uses grid_sample for trilinear interpolation of the latent 
        volume at each coordinate location.
        
        Args:
            coords: (N, 3) coordinates in [-1, 1]
            latents: (N, C, X, Y, Z) latent volumes
            condition_vecs: (N, C_cond) optional condition vectors
        
        Returns:
            Interpolated latent features (N, C + C_cond)
        """
        coords_5d = coords[:, None, None, None, :]  # (N, 1, 1, 1, 3)
        latents_interp = F.grid_sample(
            latents, coords_5d, mode='bilinear',
            align_corners=True, padding_mode='border'
        ).squeeze()
        
        # Handle case where batch dim is squeezed
        if latents_interp.dim() == 1:
            latents_interp = latents_interp.unsqueeze(0)
        
        if condition_vecs is not None and condition_vecs.shape[-1] > 0:
            latents_interp = torch.cat(
                (latents_interp, condition_vecs), dim=-1
            )
        
        return latents_interp


class Modulator(nn.Module):
    """
    Spatial modulator for latent codes using 3D convolutions.
    
    Processes the 5D latent volume (B, C, X, Y, Z) with a Conv3d layer
    to enable spatial modulation patterns.
    
    Args:
        latent_dims: [C, X, Y, Z] latent dimensions
        kernel_size: CNN kernel size (0 = identity / no CNN)
    """
    
    def __init__(self, latent_dims, kernel_size=3):
        super().__init__()
        if kernel_size > 0:
            self.conv = nn.Conv3d(
                latent_dims[0], latent_dims[0],
                kernel_size, padding='same'
            )
        else:
            self.conv = nn.Identity()
    
    def forward(self, latent_vecs):
        """
        Args:
            latent_vecs: (B, C, X, Y, Z)
        Returns:
            Modulated latent volume (B, C, X, Y, Z)
        """
        return self.conv(latent_vecs)
