"""
napari_plugin.py — Interactive Napari viewer for bone atlas and fracture density.

Provides a GUI for:
    - Visualizing the learned bone atlas as a 3D surface
    - Sliding through latent space to see morphological variation
    - Overlaying fracture density as a colored heatmap
    - Loading and comparing individual fragment meshes

Usage:
    python -m visualization.napari_plugin --checkpoint <path> [--density <path>]
    
    Or from Python:
        from visualization.napari_plugin import launch_viewer
        launch_viewer(checkpoint_path='output/bone_xxx/checkpoint_epoch_49.pth')

Requirements:
    pip install napari[all] magicgui
"""

import os
import sys
import argparse
import numpy as np
import torch
import trimesh

try:
    import napari
    from magicgui import magicgui
    from magicgui.widgets import FloatSlider, PushButton, CheckBox
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False
    print("Warning: napari/magicgui not installed. Install with: pip install napari[all] magicgui")

from utils import generate_world_grid, extract_mesh_from_sdf


def load_trained_model(checkpoint_path, device='cpu'):
    """
    Load a trained CINeMA-Bone checkpoint.
    
    Returns:
        decoder: Trained BoneINRDecoder
        latents: (N, C, X, Y, Z) tensor of learned latent codes
        args: Training configuration dict
    """
    from models.inr_decoder import BoneINRDecoder
    
    checkpoint = torch.load(checkpoint_path, weights_only=False,
                            map_location=device)
    args = checkpoint['args']
    args['device'] = device
    
    # Rebuild decoder
    decoder = BoneINRDecoder(args, device).to(device)
    decoder.load_state_dict(checkpoint['inr_decoder'])
    decoder.eval()
    
    latents = checkpoint['latents'].to(device)
    
    return decoder, latents, args


def decode_latent_to_mesh(decoder, latent_vec, args, device='cpu'):
    """
    Decode a latent vector to a mesh via SDF → marching cubes.
    
    Args:
        decoder: Trained BoneINRDecoder
        latent_vec: (1, C, X, Y, Z) latent code
        args: Config dict
        device: Compute device
    
    Returns:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle faces
    """
    grid_coords, grid_shape, _ = generate_world_grid(args, device=device)
    
    cond_dims = args['inr_decoder'].get('cond_dims', 0)
    condition_vec = torch.zeros(cond_dims, device=device)
    
    with torch.no_grad():
        sdf_vol = decoder.inference(
            grid_coords, latent_vec, condition_vec,
            grid_shape, tfs=None
        )
    
    sdf_np = sdf_vol.detach().cpu().numpy()
    
    vertices, faces, normals = extract_mesh_from_sdf(sdf_np, level=0.0)
    
    return vertices, faces, sdf_np


def launch_viewer(checkpoint_path, density_path=None, device='cpu'):
    """
    Launch the interactive Napari atlas viewer.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        density_path: Optional path to fracture density .npy file
        device: Compute device
    """
    if not HAS_NAPARI:
        print("Error: napari is required. Install with: pip install napari[all] magicgui")
        return
    
    print("Loading model...")
    decoder, latents, args = load_trained_model(checkpoint_path, device)
    
    # Compute mean and std of latent codes for slider range
    latent_mean = latents.mean(dim=0, keepdim=True)
    latent_std = latents.std(dim=0).mean().item()
    
    # Generate initial atlas mesh (mean latent)
    print("Generating atlas mesh...")
    vertices, faces, atlas_sdf = decode_latent_to_mesh(
        decoder, latent_mean, args, device
    )
    
    # Load fracture density if available
    fracture_density = None
    if density_path and os.path.exists(density_path):
        print("Loading fracture density map...")
        fracture_density = np.load(density_path)
    
    # Create Napari viewer
    viewer = napari.Viewer(title="CINeMA-Bone Atlas Viewer", ndisplay=3)
    
    # Add atlas surface
    if len(vertices) > 0:
        surface_data = (vertices, faces)
        surface_layer = viewer.add_surface(
            surface_data,
            name='Atlas (Mean Humerus)',
            colormap='gray',
            opacity=0.8,
        )
    
    # Add atlas SDF as volume (for cross-section viewing)
    viewer.add_image(
        atlas_sdf,
        name='Atlas SDF Volume',
        colormap='RdBu',
        contrast_limits=[-1, 1],
        visible=False,
        rendering='attenuated_mip',
    )
    
    # Add fracture density overlay
    if fracture_density is not None:
        viewer.add_image(
            fracture_density,
            name='Fracture Density',
            colormap='hot',
            contrast_limits=[0, fracture_density.max()],
            opacity=0.6,
            visible=True,
            rendering='attenuated_mip',
            blending='additive',
        )
    
    # ===================================================================
    # Interactive Widgets via magicgui
    # ===================================================================
    
    @magicgui(
        call_button="Update Atlas",
        latent_scale={
            "widget_type": "FloatSlider",
            "min": -3.0, "max": 3.0, "step": 0.1,
            "label": "Latent Scale (σ)",
        },
        pc_index={
            "widget_type": "Slider",
            "min": 0, "max": min(9, latents.shape[0] - 1),
            "label": "Subject Index",
        },
        mode={
            "widget_type": "ComboBox",
            "choices": ["Mean", "Interpolate to Subject", "Random Sample"],
            "label": "Exploration Mode",
        }
    )
    def explore_latent_space(
        latent_scale: float = 0.0,
        pc_index: int = 0,
        mode: str = "Mean"
    ):
        """Explore the latent space by modifying the latent code."""
        with torch.no_grad():
            if mode == "Mean":
                z = latent_mean.clone()
            elif mode == "Interpolate to Subject":
                # Interpolate between mean and subject latent
                subject_z = latents[pc_index:pc_index + 1]
                alpha = (latent_scale + 3.0) / 6.0  # Map [-3, 3] to [0, 1]
                z = (1 - alpha) * latent_mean + alpha * subject_z
            elif mode == "Random Sample":
                # Sample from N(mean, scale * std)
                noise = torch.randn_like(latent_mean) * latent_std * abs(latent_scale)
                z = latent_mean + noise
            else:
                z = latent_mean.clone()
            
            # Add scaled deviation from mean
            if mode == "Mean" and latent_scale != 0.0:
                # Use PCA to find the first principal component of variation
                # Center the latents
                centered_latents = latents - latent_mean
                # Flatten for PCA: (N, D)
                N = latents.shape[0]
                flat_latents = centered_latents.reshape(N, -1)
                
                # SVD to get principal components
                # U, S, Vh = torch.linalg.svd(flat_latents, full_matrices=False)
                # First PC is Vh[0] (direction of max variance)
                # But for speed/stability with small N, we can just use the direction of max variance
                # or just use the first subject's difference as a proxy if SVD is too heavy
                # Let's do proper SVD
                try:
                    _, _, Vh = torch.linalg.svd(flat_latents, full_matrices=False)
                    principal_dir = Vh[0] # (D,)
                    principal_dir = principal_dir.reshape(latent_mean.shape) # (1, C, X, Y, Z)
                    
                    # Scale by the amount of variance in this direction (singular value?)
                    # Or just treat the slider as "sigma" units along this unit vector
                    # Total variance along PC1 is S[0]^2 / (N-1)
                    # std_dev along PC1 is S[0] / sqrt(N-1)
                    # We want z + scale * std_dev * unit_vector
                    # z + scale * (S[0]/sqrt(N-1)) * Vh[0]
                    # Actually, let's just use the scalar std deviation magnitude we calculated earlier
                    # to keep the scale intuitive, but apply it in the principal direction
                    
                    z = z + latent_scale * latent_std * principal_dir * 1.0 # Standard 1-sigma scaling
                except Exception as e:
                    print(f"PCA failed: {e}, falling back to random direction")
                    z = z + latent_scale * latent_std * torch.randn_like(z)
            
            # Decode and update mesh
            new_verts, new_faces, new_sdf = decode_latent_to_mesh(
                decoder, z, args, device
            )
            
            if len(new_verts) > 0:
                # Update surface layer
                for layer in viewer.layers:
                    if layer.name == 'Atlas (Mean Humerus)':
                        layer.data = (new_verts, new_faces)
                        break
                
                # Update SDF volume
                for layer in viewer.layers:
                    if layer.name == 'Atlas SDF Volume':
                        layer.data = new_sdf
                        break
        
        print(f"Updated: mode={mode}, scale={latent_scale:.2f}")
    
    @magicgui(
        call_button="Load Fragment STL",
        stl_path={"widget_type": "FileEdit", "mode": "r",
                   "label": "Fragment STL File",
                   "filter": "*.stl"},
    )
    def load_fragment(stl_path: str = ""):
        """Load an individual fragment mesh for comparison."""
        if not stl_path or not os.path.exists(stl_path):
            print(f"File not found: {stl_path}")
            return
        
        try:
            mesh = trimesh.load(stl_path, force='mesh')
            # Center and normalize
            mesh.vertices -= mesh.vertices.mean(axis=0)
            extent = mesh.vertices.max() - mesh.vertices.min()
            mesh.vertices *= 2.0 / (extent + 1e-8)
            
            name = os.path.basename(stl_path)
            viewer.add_surface(
                (mesh.vertices, mesh.faces),
                name=f'Fragment: {name}',
                colormap='cyan',
                opacity=0.5,
            )
            print(f"Loaded fragment: {name}")
        except Exception as e:
            print(f"Error loading {stl_path}: {e}")
    
    @magicgui(
        call_button="Update Density Settings",
        show_density={"widget_type": "CheckBox", "value": True,
                       "label": "Show Fracture Density"},
        density_opacity={"widget_type": "FloatSlider",
                         "min": 0.0, "max": 1.0, "step": 0.05,
                         "value": 0.8, "label": "Opacity"},
        density_threshold={"widget_type": "FloatSlider",
                           "min": 0.0, "max": 1.0, "step": 0.05,
                           "value": 0.2, "label": "Min Threshold"},
    )
    def toggle_density(show_density: bool = True,
                        density_opacity: float = 0.8,
                        density_threshold: float = 0.2):
        """Control fracture density visualization parameters."""
        for layer in viewer.layers:
            if layer.name == 'Fracture Density':
                layer.visible = show_density
                layer.opacity = density_opacity
                # Update contrast limits to clip values below threshold
                # Assuming max value is around 1.0 (probability) or normalized
                current_max = layer.contrast_limits[1]
                layer.contrast_limits = [current_max * density_threshold, current_max]
    
    # Add widgets to viewer
    viewer.window.add_dock_widget(
        explore_latent_space, name="Latent Space Explorer",
        area='right'
    )
    viewer.window.add_dock_widget(
        load_fragment, name="Fragment Loader",
        area='right'
    )
    if fracture_density is not None:
        viewer.window.add_dock_widget(
            toggle_density, name="Density Overlay",
            area='right'
        )
    
    print("\n✓ Napari viewer launched!")
    print("  Use the right sidebar widgets to explore the atlas.")
    
    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="CINeMA-Bone Napari Atlas Viewer"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained model checkpoint (.pth)'
    )
    parser.add_argument(
        '--density', type=str, default=None,
        help='Path to fracture density map (.npy)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device for inference (cpu or cuda)'
    )
    
    args = parser.parse_args()
    launch_viewer(args.checkpoint, args.density, args.device)


if __name__ == '__main__':
    main()
