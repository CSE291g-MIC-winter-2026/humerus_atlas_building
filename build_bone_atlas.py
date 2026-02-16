"""
build_bone_atlas.py — Core training loop for bone SDF atlas building.

Adapted from CINeMA's AtlasBuilder. Implements the auto-decoder architecture:
jointly optimizes the INR network weights (θ), per-subject latent codes (z_i),
and per-subject rigid transformations (R_i, t_i) to learn a population atlas
of proximal humerus anatomy from fractured bone fragment data.
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from models.inr_decoder import BoneINRDecoder
from data_loading.dataset import BoneDataset
from utils import (
    BoneLoss, generate_world_grid, extract_mesh_from_sdf,
    save_mesh_stl, save_sdf_nifti, log_loss, to_device
)


class BoneAtlasBuilder:
    """
    Auto-decoder atlas builder for bone SDF.
    
    Training loop:
        For each epoch:
            For each batch of coordinate samples across subjects:
                1. Look up per-subject latent codes z_i
                2. Apply per-subject rigid transforms (R_i, t_i) to coordinates
                3. Decode: f_θ(x, z_i) → predicted SDF
                4. Compute loss against ground-truth SDF
                5. Backprop into θ, z_i, (R_i, t_i)
    
    After training:
        - The mean latent (z ≈ 0) represents the canonical humerus
        - Individual z_i encode patient-specific deviations
        - Fracture locations can be mapped to atlas space via learned transforms
    """
    
    def __init__(self, args, train=True):
        self.args = args
        self.device = args['device']
        self.loss_criterion = BoneLoss(args).to(self.device)
        
        self._init_atlas_training()
        
        if train:
            self.train_on_data()
    
    # =======================================================================
    # Training
    # =======================================================================
    
    def train_on_data(self):
        """Main training loop."""
        loss_hist_epochs = []
        start_time = time.time()
        
        for epoch in range(self.args['epochs']['train']):
            if self.args['optimizer'].get('re_init_latents', False):
                self.re_init_latents()
            
            epoch_losses, n_updates = self._train_epoch(epoch, split='train')
            loss_hist_epochs.append(epoch_losses['total'])
            
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}: "
                f"total={epoch_losses['total']:.6f} | "
                f"sdf={epoch_losses['sdf']:.6f} | "
                f"latent_reg={epoch_losses['latent_reg']:.6f} | "
                f"tf_reg={epoch_losses['tf_reg']:.6f} | "
                f"total(5-ep-avg)={np.mean(loss_hist_epochs[-5:]):.6f} | "
                f"time={elapsed:.1f}s"
            )
            
            # Validation and atlas generation
            if (epoch + 1) % self.args['validate_every'] == 0 or \
               (epoch + 1) == self.args['epochs']['train']:
                self._validate(epoch)
            
            if n_updates > 0:
                self._update_scheduler(split='train')
        
        # Final atlas generation
        print("\n=== Training Complete ===")
        self.generate_atlas(epoch=self.args['epochs']['train'] - 1)
        self.save_state(self.args['epochs']['train'] - 1)
        
        return np.mean(loss_hist_epochs)
    
    def _train_epoch(self, epoch, split='train'):
        """Train for one epoch."""
        self.inr_decoder[split].train()
        loss_hist = {
            'total': [],
            'sdf': [],
            'latent_reg': [],
            'tf_reg': [],
        }
        
        n_updates = 0
        for batch in self.dataloaders[split]:
            batch_losses, batch_updates = self._train_batch(batch, epoch, split)
            n_updates += batch_updates
            for key in loss_hist:
                loss_hist[key].append(batch_losses[key])
        
        return {
            key: (float(np.mean(vals)) if len(vals) > 0 else 0.0)
            for key, vals in loss_hist.items()
        }, n_updates
    
    def _train_batch(self, batch, epoch, split='train'):
        """Process one batch of coordinate samples."""
        n_smpls = self.args['n_samples']
        coords_batch, values_batch, conditions_batch, idx_df_batch = to_device(
            batch, self.device
        )
        
        loss_hist = {
            'total': [],
            'sdf': [],
            'latent_reg': [],
            'tf_reg': [],
        }
        
        n_updates = 0
        for smpls in range(0, idx_df_batch.shape[0], n_smpls):
            self.optimizers[split].zero_grad()
            
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            conditions = conditions_batch[smpls:smpls + n_smpls]

            with torch.autocast(device_type=self.device,
                                enabled=self.args['amp']):
                # Forward pass through INR decoder
                values_pred = self.inr_decoder[split](
                    coords, self.latents[split], conditions,
                    self.transformations[split][idx_df],
                    idcs_df=idx_df
                )
                
                # Compute loss
                losses = self.loss_criterion(
                    values_pred, values,
                    tfs=self.transformations[split][idx_df],
                    latents=self.latents[split]
                )
            
            # Backward pass
            if self.args['amp']:
                self.grad_scalers[split].scale(losses['total']).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                losses['total'].backward()
                self.optimizers[split].step()
            n_updates += 1
            
            for key in loss_hist:
                val = losses[key]
                if isinstance(val, torch.Tensor):
                    val = val.detach().item()
                loss_hist[key].append(float(val))
            log_loss(losses, epoch, split, self.args['logging'])
        
        return {
            key: (float(np.mean(vals)) if len(vals) > 0 else 0.0)
            for key, vals in loss_hist.items()
        }, n_updates
    
    # =======================================================================
    # Validation
    # =======================================================================
    
    def _validate(self, epoch_train):
        """Validate and generate reconstructions."""
        print(f"\n--- Validation at epoch {epoch_train} ---")
        
        # Generate atlas
        if self.args.get('generate_atlas', True):
            self.generate_atlas(epoch=epoch_train)
        
        # Reconstruct training subjects
        self._generate_subject_reconstructions(
            idcs_df=list(range(min(3, len(self.datasets['train'])))),
            epoch=epoch_train, split='train'
        )
        
        # Fit latent codes to validation set
        if len(self.datasets.get('val', [])) > 0:
            self._init_validation()
            for epoch_val in range(self.args['epochs']['val']):
                _, n_updates = self._train_epoch(epoch=epoch_val, split='val')
                if n_updates > 0:
                    self._update_scheduler(split='val')
            
            self._generate_subject_reconstructions(
                idcs_df=list(range(min(3, len(self.datasets['val'])))),
                epoch=epoch_train, split='val'
            )
        
        self.save_state(epoch_train)
    
    def _generate_subject_reconstructions(self, idcs_df, epoch, split):
        """Generate and save reconstructed SDF volumes for specified subjects."""
        out_dir = os.path.join(self.args['output_dir'], split)
        os.makedirs(out_dir, exist_ok=True)
        
        grid_coords, grid_shape, affine = generate_world_grid(
            self.args, device=self.device
        )
        
        self.inr_decoder[split].eval()
        
        for idx in idcs_df:
            subject_id = self.datasets[split].subjects[idx]['subject_id']
            
            with torch.no_grad():
                tfs = self.transformations[split][idx, None]
                cond = self.datasets[split]._build_conditions(
                    self.datasets[split].subjects[idx]
                ).to(self.device)
                
                sdf_vol = self.inr_decoder[split].inference(
                    grid_coords,
                    self.latents[split][idx:idx + 1],
                    cond,
                    grid_shape,
                    tfs
                )
            
            sdf_np = sdf_vol.detach().cpu().numpy()
            
            # Save as NIfTI
            nifti_path = os.path.join(
                out_dir, f"{subject_id}_ep{epoch}_sdf.nii.gz"
            )
            save_sdf_nifti(sdf_np, affine, nifti_path)
            
            # Save as mesh (marching cubes)
            try:
                vertices, faces, normals = extract_mesh_from_sdf(sdf_np, level=0.0)
                if len(vertices) > 0:
                    mesh_path = os.path.join(
                        out_dir, f"{subject_id}_ep{epoch}.stl"
                    )
                    save_mesh_stl(vertices, faces, mesh_path)
            except Exception as e:
                print(f"  Warning: mesh extraction failed for {subject_id}: {e}")
        
        self.inr_decoder[split].train()
    
    # =======================================================================
    # Atlas Generation
    # =======================================================================
    
    def generate_atlas(self, epoch=0):
        """
        Generate the population-mean atlas from z=0 latent code.
        
        The mean latent (z≈0) represents the canonical average humerus.
        """
        print("Generating atlas...")
        self.inr_decoder['train'].eval()
        
        grid_coords, grid_shape, affine = generate_world_grid(
            self.args, device=self.device
        )
        
        # Mean latent: weighted average of all training latents
        mean_latent = self._get_mean_latent()
        
        # Zero condition vector
        cond_dims = self.args['inr_decoder'].get('cond_dims', 0)
        condition_vec = torch.zeros(cond_dims, device=self.device)
        
        with torch.no_grad():
            sdf_vol = self.inr_decoder['train'].inference(
                grid_coords, mean_latent, condition_vec,
                grid_shape, tfs=None
            )
        
        sdf_np = sdf_vol.detach().cpu().numpy()
        
        # Save atlas SDF
        atlas_dir = os.path.join(self.args['output_dir'], 'atlas')
        os.makedirs(atlas_dir, exist_ok=True)
        
        nifti_path = os.path.join(atlas_dir, f"atlas_sdf_ep{epoch}.nii.gz")
        save_sdf_nifti(sdf_np, affine, nifti_path)
        
        # Extract and save atlas mesh
        try:
            vertices, faces, normals = extract_mesh_from_sdf(
                sdf_np, level=0.0,
                spacing=self.args['atlas_gen'].get('spacing', [1.0, 1.0, 1.0])
            )
            if len(vertices) > 0:
                mesh_path = os.path.join(atlas_dir, f"atlas_mesh_ep{epoch}.stl")
                save_mesh_stl(vertices, faces, mesh_path)
                print(f"Atlas mesh: {len(vertices)} vertices, {len(faces)} faces")
        except Exception as e:
            print(f"Warning: Atlas mesh extraction failed: {e}")
        
        self.inr_decoder['train'].train()
        return sdf_np
    
    def _get_mean_latent(self, split='train'):
        """
        Compute the mean latent code across all training subjects.
        
        In CINeMA, this uses Gaussian-weighted regression over a temporal
        axis. For bone (no temporal axis), we simply average all latents.
        """
        latents = self.latents[split]
        mean_latent = latents.mean(dim=0, keepdim=True)
        return mean_latent
    
    # =======================================================================
    # Initialization
    # =======================================================================
    
    def _init_atlas_training(self):
        """Initialize all training components."""
        self.datasets, self.dataloaders = {}, {}
        self.inr_decoder, self.latents, self.transformations = {}, {}, {}
        self.optimizers, self.grad_scalers = {}, {}
        self.schedulers = {}
        
        chkp_path = self.args['load_model']['path']
        if len(chkp_path) > 0:
            self.load_checkpoint(chkp_path, self.args['load_model']['epoch'])
        else:
            self._init_dataloading(split='train')
            self._init_inr(split='train')
            self._init_transformations(split='train')
            self._init_latents(split='train')
        
        self._init_optimizer(split='train')
        self._init_dataloading(split='val')
    
    def _init_validation(self):
        """Reinitialize validation components (avoid information leakage)."""
        torch.manual_seed(self.args['seed'])
        self._init_latents(split='val')
        self._init_transformations(split='val')
        self._init_optimizer(split='val')
        self.inr_decoder['val'] = copy.deepcopy(self.inr_decoder['train'])
        self.inr_decoder['val'].eval()
    
    def _init_dataloading(self, split='train'):
        """Initialize dataset and dataloader."""
        shuffle = (split == 'train')
        
        self.datasets[split] = BoneDataset(self.args, split=split)
        
        if len(self.datasets[split]) == 0:
            print(f"  Warning: No subjects found for {split} split")
            return
        
        self.dataloaders[split] = DataLoader(
            self.datasets[split],
            batch_size=self.args['batch_size'],
            num_workers=self.args['num_workers'],
            shuffle=shuffle,
            collate_fn=self.datasets[split].collate_fn,
            pin_memory=True
        )
        
        print(f"Initialized dataloader for {split}: "
              f"{len(self.datasets[split])} subjects")
    
    def _init_inr(self, state_dict=None, split='train'):
        """Initialize INR decoder network."""
        # Count active conditions
        cond_config = self.args['dataset'].get('conditions', {})
        self.args['inr_decoder']['cond_dims'] = sum(
            1 for v in cond_config.values() if v
        )
        
        self.inr_decoder[split] = BoneINRDecoder(
            self.args, self.device
        ).to(self.device)
        
        if state_dict is not None:
            self.inr_decoder[split].load_state_dict(state_dict)
        
        n_params = sum(p.numel() for p in self.inr_decoder[split].parameters())
        print(f"INR decoder: {n_params:,} parameters")
    
    def _init_transformations(self, tfs=None, split='train'):
        """Initialize per-subject rigid transformation parameters."""
        n_subjects = len(self.datasets[split])
        tf_dim = max(self.args['inr_decoder']['tf_dim'], 6)
        
        shape = (n_subjects, tf_dim)
        if tfs is None:
            tfs = torch.zeros(shape, device=self.device)
        else:
            tfs = tfs.to(self.device)
        
        if self.args['inr_decoder']['tf_dim'] > 0:
            self.transformations[split] = nn.Parameter(tfs)
        else:
            self.transformations[split] = tfs  # Fixed at zero
    
    def _init_latents(self, lats=None, split='train'):
        """Initialize per-subject latent codes from N(0, 0.01)."""
        n_subjects = len(self.datasets[split])
        latent_dim = self.args['inr_decoder']['latent_dim']
        
        shape = (n_subjects, *latent_dim)
        if lats is None:
            lats = torch.normal(0, 0.01, size=shape, device=self.device)
        else:
            lats = lats.to(self.device)
        
        self.latents[split] = nn.Parameter(lats)
    
    def re_init_latents(self, split='train'):
        """Re-initialize latent codes and transformations."""
        self.latents[split].data.normal_(0, 0.01)
        self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()
    
    def _init_optimizer(self, split='train'):
        """Initialize optimizer with separate learning rates per parameter group."""
        params = [
            {
                'name': f'latents_{split}',
                'params': self.latents[split],
                'lr': self.args['optimizer']['lr_latent'],
                'weight_decay': self.args['optimizer']['latent_weight_decay']
            }
        ]
        
        if self.args['inr_decoder']['tf_dim'] > 0:
            params.append({
                'name': f'transformations_{split}',
                'params': self.transformations[split],
                'lr': self.args['optimizer']['lr_tf'],
                'weight_decay': self.args['optimizer']['tf_weight_decay']
            })
        
        if split == 'train':
            params.append({
                'name': 'inr_decoder',
                'params': self.inr_decoder[split].parameters(),
                'lr': self.args['optimizer']['lr_inr'],
                'weight_decay': self.args['optimizer']['inr_weight_decay']
            })
        
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args['amp'] else None
        
        if self.args['optimizer']['scheduler']['type'] == 'cosine':
            self.schedulers[split] = CosineAnnealingLR(
                self.optimizers[split],
                T_max=self.args['epochs'][split],
                eta_min=self.args['optimizer']['scheduler']['eta_min']
            )
        else:
            self.schedulers[split] = None
    
    def _update_scheduler(self, split='train'):
        if self.schedulers.get(split) is not None:
            self.schedulers[split].step()
    
    # =======================================================================
    # Checkpointing
    # =======================================================================
    
    def save_state(self, epoch, split='train'):
        """Save model checkpoint."""
        if not self.args['save_model']:
            return
        
        log_dir = self.args['output_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'latents': self.latents[split].detach().cpu(),
            'transformations': self.transformations[split].detach().cpu()
                if isinstance(self.transformations[split], nn.Parameter)
                else self.transformations[split].cpu(),
            'inr_decoder': self.inr_decoder[split].state_dict(),
            'args': self.args
        }
        
        path = os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, chkp_path, epoch=None):
        """Load a saved checkpoint."""
        if epoch is not None:
            chkp_path = os.path.join(chkp_path, f'checkpoint_epoch_{epoch}.pth')
        
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f"Checkpoint not found: {chkp_path}")
        
        chkp = torch.load(chkp_path, weights_only=False, map_location=self.device)
        
        self._init_dataloading(split='train')
        self._init_inr(chkp['inr_decoder'], split='train')
        self._init_transformations(chkp['transformations'])
        self._init_latents(chkp['latents'])
        
        print(f"Loaded checkpoint: {chkp_path}")
    
    # =======================================================================
    # Access Methods (for fracture atlas module)
    # =======================================================================
    
    def get_latents(self, split='train'):
        """Return learned latent codes."""
        return self.latents[split].detach()
    
    def get_transformations(self, split='train'):
        """Return learned rigid transformations."""
        return self.transformations[split].detach()
    
    def get_decoder(self):
        """Return the trained INR decoder."""
        return self.inr_decoder['train']
