"""
density_map.py — Fracture Density Map Generator (Subproject 4).

Aggregates fracture locations across the patient population in atlas space
to produce a 3D fracture density/frequency map. This map shows where
fractures most commonly occur on the proximal humerus.

Workflow:
    1. Load trained atlas model (checkpoint)
    2. For each patient: compare fragment SDF to complete bone SDF
    3. Identify fracture boundaries (where fragments end)
    4. Transform fracture locations into atlas space using learned (R, t)
    5. Accumulate on atlas grid → fracture frequency volume
    6. Normalize by number of patients → probability density
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from utils import generate_world_grid, extract_mesh_from_sdf, save_sdf_nifti


def compute_fracture_density_map(
    atlas_builder,
    split='train',
    resolution=128,
    fracture_threshold=0.3
):
    """
    Compute the fracture density map by analyzing fragment boundaries
    in atlas space.
    
    For each subject with multiple fragments, fracture lines appear where
    the individual fragment SDFs transition from inside to outside.
    In atlas space, this reveals population-level fracture patterns.
    
    Args:
        atlas_builder: Trained BoneAtlasBuilder instance
        split: Which data split to analyze
        resolution: Resolution of the density map
        fracture_threshold: SDF transition threshold for fracture detection
    
    Returns:
        density_map: (resolution, resolution, resolution) numpy array
        atlas_sdf: (resolution, resolution, resolution) atlas SDF
    """
    device = atlas_builder.device
    args = atlas_builder.args
    
    # Generate atlas grid
    grid_coords, grid_shape, affine = generate_world_grid(args, device=device)
    
    # Get atlas (mean) SDF for reference
    atlas_sdf = atlas_builder.generate_atlas()
    
    # Accumulate fracture indicators
    fracture_count = np.zeros(grid_shape, dtype=np.float32)
    n_subjects = 0
    
    dataset = atlas_builder.datasets[split]
    decoder = atlas_builder.inr_decoder[split]
    decoder.eval()
    
    for idx in range(len(dataset)):
        subject = dataset.subjects[idx]
        
        # Skip template (no fracture)
        if subject.get('subject_type') == 'template':
            continue
        
        print(f"  Analyzing subject {subject['subject_id']}...")
        
        with torch.no_grad():
            # Subject-specific reconstruction in atlas space
            tfs = atlas_builder.transformations[split][idx, None]
            cond = dataset._build_conditions(subject).to(device)
            
            subject_sdf = decoder.inference(
                grid_coords,
                atlas_builder.latents[split][idx:idx + 1],
                cond, grid_shape, tfs
            ).detach().cpu().numpy()
        
        # Detect fracture lines:
        # Fracture boundaries are where the subject's bone surface
        # deviates significantly from the atlas reference
        # (fragments have edges where the intact bone would be continuous)
        
        # Method: find voxels that are inside the atlas bone (SDF < 0)
        # but outside the subject's reconstruction (SDF > 0)
        # These are the "missing" regions near fractures
        
        atlas_inside = (atlas_sdf < 0).astype(np.float32)
        subject_inside = (subject_sdf < 0).astype(np.float32)
        
        # Fracture zone: atlas says bone, subject says no bone
        fracture_zone = atlas_inside * (1 - subject_inside)
        
        # Also detect transition regions (near-surface differences)
        sdf_diff = np.abs(atlas_sdf - subject_sdf)
        near_surface = (np.abs(atlas_sdf) < fracture_threshold)
        fracture_transition = near_surface * (sdf_diff > fracture_threshold * 0.5)
        
        # Combine detections
        fracture_indicator = np.maximum(fracture_zone, fracture_transition)
        fracture_count += fracture_indicator
        n_subjects += 1
    
    # Normalize to probability density
    if n_subjects > 0:
        density_map = fracture_count / n_subjects
    else:
        density_map = fracture_count
    
    decoder.train()
    
    return density_map, atlas_sdf


def save_fracture_density(density_map, atlas_sdf, affine, output_dir,
                          epoch=0):
    """
    Save fracture density map and overlaid atlas mesh.
    
    Args:
        density_map: (X, Y, Z) fracture density volume [0, 1]
        atlas_sdf: (X, Y, Z) atlas SDF volume
        affine: 4x4 NIfTI affine matrix
        output_dir: Output directory
        epoch: Epoch number for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save density as NIfTI
    density_path = os.path.join(output_dir, f'fracture_density_ep{epoch}.nii.gz')
    save_sdf_nifti(density_map, affine, density_path)
    
    # Save atlas SDF
    atlas_path = os.path.join(output_dir, f'atlas_sdf_ep{epoch}.nii.gz')
    save_sdf_nifti(atlas_sdf, affine, atlas_path)
    
    # Save density as numpy for Napari
    np.save(os.path.join(output_dir, 'fracture_density.npy'), density_map)
    np.save(os.path.join(output_dir, 'atlas_sdf.npy'), atlas_sdf)
    
    # Extract atlas mesh with density coloring
    try:
        vertices, faces, normals = extract_mesh_from_sdf(atlas_sdf, level=0.0)
        
        if len(vertices) > 0:
            # Sample density values at mesh vertices
            _assign_density_colors(
                vertices, faces, density_map, atlas_sdf,
                os.path.join(output_dir, f'atlas_fracture_overlay_ep{epoch}.ply')
            )
    except Exception as e:
        print(f"Warning: Could not create overlay mesh: {e}")
    
    # Print statistics
    max_density = density_map.max()
    mean_density = density_map[atlas_sdf < 0].mean() if (atlas_sdf < 0).any() else 0
    print(f"\nFracture Density Statistics:")
    print(f"  Max density: {max_density:.4f}")
    print(f"  Mean density (inside bone): {mean_density:.4f}")
    print(f"  Saved to: {output_dir}")


def _assign_density_colors(vertices, faces, density_map, atlas_sdf,
                            output_path):
    """
    Create a colored mesh with fracture density as vertex colors.
    
    Uses a red-to-yellow colormap for density visualization.
    """
    import trimesh
    
    resolution = density_map.shape[0]
    
    # Map vertex positions to volume indices
    # Vertices are in physical space, need to map back to volume indices
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_range = v_max - v_min
    v_range[v_range == 0] = 1.0
    
    v_normalized = (vertices - v_min) / v_range  # [0, 1]
    v_indices = (v_normalized * (resolution - 1)).astype(int)
    v_indices = np.clip(v_indices, 0, resolution - 1)
    
    # Sample density at vertices
    density_at_vertices = density_map[
        v_indices[:, 0], v_indices[:, 1], v_indices[:, 2]
    ]
    
    # Normalize density to [0, 1] for coloring
    d_max = density_at_vertices.max()
    if d_max > 0:
        density_norm = density_at_vertices / d_max
    else:
        density_norm = density_at_vertices
    
    # Color map: bone (white/grey) → fracture zone (red → yellow)
    colors = np.zeros((len(vertices), 4), dtype=np.uint8)
    colors[:, 3] = 255  # Alpha
    
    for i in range(len(vertices)):
        d = density_norm[i]
        if d < 0.1:
            # Low density: bone color (light grey)
            colors[i, :3] = [220, 220, 210]
        else:
            # High density: red → yellow gradient
            r = 255
            g = int(min(255, 50 + d * 200))
            b = int(max(0, 50 - d * 50))
            colors[i, :3] = [r, g, b]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                            vertex_colors=colors)
    mesh.export(output_path)
    print(f"Saved colored overlay mesh: {output_path}")
