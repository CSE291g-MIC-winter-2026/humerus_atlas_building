"""
fracture_analyzer.py — Statistical analysis of fracture patterns.

Extracts the mean humerus mesh from the atlas, overlays fracture density,
identifies high-density fracture zones, and exports for 3D visualization.
"""

import os
import numpy as np
from utils import extract_mesh_from_sdf, save_mesh_stl


def analyze_fracture_patterns(density_map, atlas_sdf, output_dir,
                               percentile_threshold=90):
    """
    Analyze fracture density patterns on the atlas.
    
    Identifies high-density fracture zones and computes statistics.
    
    Args:
        density_map: (X, Y, Z) fracture density volume
        atlas_sdf: (X, Y, Z) atlas SDF volume
        output_dir: Output directory for results
        percentile_threshold: Percentile for high-density zone detection
    
    Returns:
        Dict of analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Mask to bone interior
    bone_mask = atlas_sdf < 0
    bone_surface_mask = (np.abs(atlas_sdf) < 0.1)
    
    # Statistics within bone volume
    density_in_bone = density_map[bone_mask]
    density_on_surface = density_map[bone_surface_mask]
    
    # Find high-density fracture zones
    if len(density_on_surface) > 0:
        threshold = np.percentile(density_on_surface, percentile_threshold)
        high_density_mask = (density_map > threshold) & bone_surface_mask
    else:
        threshold = 0
        high_density_mask = np.zeros_like(density_map, dtype=bool)
    
    # Compute zone volume
    resolution = density_map.shape[0]
    voxel_vol = (2.0 / resolution) ** 3  # Volume per voxel in normalized space
    high_density_volume = high_density_mask.sum() * voxel_vol
    total_surface_volume = bone_surface_mask.sum() * voxel_vol
    
    results = {
        'mean_density_bone': float(density_in_bone.mean()) if len(density_in_bone) > 0 else 0,
        'max_density': float(density_map.max()),
        'mean_density_surface': float(density_on_surface.mean()) if len(density_on_surface) > 0 else 0,
        'std_density_surface': float(density_on_surface.std()) if len(density_on_surface) > 0 else 0,
        'high_density_threshold': float(threshold),
        'high_density_voxels': int(high_density_mask.sum()),
        'high_density_volume_ratio': float(high_density_volume / (total_surface_volume + 1e-8)),
    }
    
    # Print report
    print("\n" + "=" * 50)
    print("  Fracture Pattern Analysis Report")
    print("=" * 50)
    print(f"  Mean fracture density (bone volume):  {results['mean_density_bone']:.4f}")
    print(f"  Mean fracture density (surface):      {results['mean_density_surface']:.4f}")
    print(f"  Max fracture density:                 {results['max_density']:.4f}")
    print(f"  High-density threshold (p{percentile_threshold}):     {results['high_density_threshold']:.4f}")
    print(f"  High-density zone volume ratio:       {results['high_density_volume_ratio']:.4f}")
    print("=" * 50)
    
    # Save high-density zone mask
    np.save(os.path.join(output_dir, 'high_density_zones.npy'), high_density_mask)
    
    # Extract high-density zone mesh
    if high_density_mask.any():
        try:
            # Create a "density SDF" where the zero level-set is the
            # boundary of the high-density zone
            density_field = density_map - threshold
            density_field[~bone_surface_mask] = -1.0  # Outside bone surface
            
            vertices, faces, normals = extract_mesh_from_sdf(
                -density_field, level=0.0  # Negate so inside is the zone
            )
            if len(vertices) > 0:
                mesh_path = os.path.join(output_dir, 'high_density_zone.stl')
                save_mesh_stl(vertices, faces, mesh_path)
        except Exception as e:
            print(f"Warning: Could not extract high-density zone mesh: {e}")
    
    # Save the mean atlas mesh
    try:
        vertices, faces, normals = extract_mesh_from_sdf(atlas_sdf, level=0.0)
        if len(vertices) > 0:
            mesh_path = os.path.join(output_dir, 'atlas_mean_humerus.stl')
            save_mesh_stl(vertices, faces, mesh_path)
    except Exception as e:
        print(f"Warning: Could not extract atlas mesh: {e}")
    
    return results


def export_for_cad(density_map, atlas_sdf, output_dir,
                    density_levels=[0.1, 0.3, 0.5, 0.7]):
    """
    Export fracture density iso-surfaces for CAD/3D printing.
    
    Creates STL meshes at multiple density thresholds for use
    in surgical hardware design overlay.
    
    Args:
        density_map: Fracture density volume
        atlas_sdf: Atlas SDF volume
        output_dir: Export directory
        density_levels: Iso-surface levels to extract
    """
    cad_dir = os.path.join(output_dir, 'cad_export')
    os.makedirs(cad_dir, exist_ok=True)
    
    # Export atlas mesh
    try:
        vertices, faces, _ = extract_mesh_from_sdf(atlas_sdf, level=0.0)
        if len(vertices) > 0:
            save_mesh_stl(vertices, faces,
                          os.path.join(cad_dir, 'atlas_humerus.stl'))
    except Exception as e:
        print(f"Warning: Atlas mesh export failed: {e}")
    
    # Export density iso-surfaces
    for level in density_levels:
        try:
            # Create a field where the zero level-set is at the density level
            density_field = density_map - level
            
            vertices, faces, _ = extract_mesh_from_sdf(
                -density_field, level=0.0
            )
            if len(vertices) > 0:
                filename = f'fracture_density_{int(level*100)}pct.stl'
                save_mesh_stl(vertices, faces,
                              os.path.join(cad_dir, filename))
                print(f"  Exported density level {level:.0%}: {filename}")
        except Exception:
            pass
    
    print(f"\nCAD exports saved to: {cad_dir}")
