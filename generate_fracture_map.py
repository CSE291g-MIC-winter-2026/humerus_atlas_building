"""
generate_fracture_map.py — Generate and analyze fracture density map.

Usage:
    python generate_fracture_map.py --run_dir output/bone_20260214_095808_loc --epoch 49
"""

import os
import argparse
import torch
import numpy as np
import yaml

from build_bone_atlas import BoneAtlasBuilder
from fracture_atlas.density_map import compute_fracture_density_map, save_fracture_density
from fracture_atlas.fracture_analyzer import analyze_fracture_patterns, export_for_cad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory')
    parser.add_argument('--epoch', type=int, default=49,
                        help='Checkpoint epoch to load')
    args_cli = parser.parse_args()
    
    # Load atlas config
    config_path = os.path.join(args_cli.run_dir, 'config_atlas.yaml')
    with open(config_path, 'r') as f:
        args_atlas = yaml.safe_load(f)

    # Load data config
    data_config_path = os.path.join(args_cli.run_dir, 'config_data.yaml')
    with open(data_config_path, 'r') as f:
        args_data = yaml.safe_load(f)

    # Merge configs
    args = {**args_data, **args_atlas}
    
    # Update args for loading
    args['load_model'] = {
        'path': args_cli.run_dir,
        'epoch': args_cli.epoch
    }
    
    # Initialize builder and load checkpoint
    print(f"Loading checkpoint from {args_cli.run_dir} (epoch {args_cli.epoch})...")
    builder = BoneAtlasBuilder(args, train=False)
    
    # 1. Compute Fracture Density Map
    print("\n[Step 1] Computing fracture density map...")
    density_map, atlas_sdf = compute_fracture_density_map(
        builder, split='train', resolution=128
    )
    
    # Save raw density map
    output_dir = os.path.join(args_cli.run_dir, 'fracture_atlas')
    save_fracture_density(
        density_map, atlas_sdf, np.eye(4), output_dir, epoch=args_cli.epoch
    )
    
    # 2. Analyze Patterns & High-Density Zones
    print("\n[Step 2] Analyzing fracture patterns...")
    stats = analyze_fracture_patterns(
        density_map, atlas_sdf, output_dir, percentile_threshold=90
    )
    
    # 3. Export for CAD
    print("\n[Step 3] Exporting for CAD...")
    export_for_cad(density_map, atlas_sdf, output_dir)
    
    print("\nDone! Results saved to:", output_dir)


if __name__ == '__main__':
    main()
