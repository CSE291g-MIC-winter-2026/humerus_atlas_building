"""
run.py — Entry point for CINeMA-Bone atlas training.

Usage:
    python run.py                              # Train with default config
    python run.py --prepare-data               # Pre-process STL data first
    python run.py --epochs_train 10            # Override training epochs
    python run.py --device cpu                 # Run on CPU
"""

import os
import sys
import yaml
import argparse
from datetime import datetime


def load_configs(cmd_args=None):
    """
    Load and merge YAML configs, then apply command-line overrides.
    """
    config_dir = os.path.join(os.path.dirname(__file__), 'configs')
    
    # Load atlas config
    with open(os.path.join(config_dir, 'config_atlas.yaml'), 'r') as f:
        args_atlas = yaml.safe_load(f)
    
    # Load dataset config
    with open(os.path.join(config_dir, 'config_data.yaml'), 'r') as f:
        all_data_configs = yaml.safe_load(f)
    
    config_data_name = args_atlas.get('config_data', 'humerus')
    if cmd_args and 'config_data' in cmd_args:
        config_data_name = cmd_args['config_data']
    
    args_data = {'dataset': all_data_configs[config_data_name]}
    
    # Merge configs
    args = {**args_data, **args_atlas}
    
    # Load subject IDs if specified
    subject_ids_path = args['dataset'].get('subject_ids', None)
    if subject_ids_path and os.path.exists(subject_ids_path):
        with open(subject_ids_path, 'r') as f:
            subject_ids_data = yaml.safe_load(f)
            dataset_name = args['dataset'].get('dataset_name', 'humerus_fracture')
            if dataset_name in subject_ids_data:
                args['dataset']['subject_ids'] = subject_ids_data[dataset_name]['subject_ids']
            else:
                args['dataset']['subject_ids'] = None
    else:
        args['dataset']['subject_ids'] = None
    
    # Apply command-line overrides
    if cmd_args:
        args = override_args(args, cmd_args)
    
    # Create output directory
    job_id = os.environ.get("SLURM_JOB_ID", "loc")[-3:]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bone_{timestamp}_{job_id}"
    args['output_dir'] = os.path.join(args['output_dir'], run_name)
    os.makedirs(args['output_dir'], exist_ok=True)
    
    # Save configs to output
    with open(os.path.join(args['output_dir'], 'config_atlas.yaml'), 'w') as f:
        yaml.dump(args_atlas, f, default_flow_style=False)
    with open(os.path.join(args['output_dir'], 'config_data.yaml'), 'w') as f:
        yaml.dump(args_data, f, default_flow_style=False)
    
    print(f"Output directory: {args['output_dir']}")
    print(f"INR Decoder config: {args['inr_decoder']}")
    
    return args


def override_args(config_args, cmd_args):
    """Apply command-line overrides using __ separator for nested keys."""
    for key, value in cmd_args.items():
        if value is None:
            continue
        
        if '__' in key:
            key1, key2 = key.split('__', 1)
            if key1 in config_args and isinstance(config_args[key1], dict):
                config_args[key1][key2] = value
        else:
            config_args[key] = value
    
    return config_args


def parse_cmd_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CINeMA-Bone Atlas Builder")
    
    # Data preparation
    parser.add_argument('--prepare-data', action='store_true',
                        help='Run data preprocessing before training')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--voxel_resolution', type=int, default=None,
                        help='Voxel grid resolution for preprocessing')
    
    # Training config
    parser.add_argument('--config_data', type=str, default=None,
                        help='Dataset configuration name')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--epochs_train', type=int, default=None,
                        dest='epochs__train')
    parser.add_argument('--epochs_val', type=int, default=None,
                        dest='epochs__val')
    parser.add_argument('--n_subjects_train', type=int, default=None,
                        dest='n_subjects__train')
    parser.add_argument('--n_subjects_val', type=int, default=None,
                        dest='n_subjects__val')
    parser.add_argument('--validate_every', type=int, default=None,
                        help='Validate and save checkpoint every N epochs')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    
    # INR config
    parser.add_argument('--inr_decoder__hidden_size', type=int, default=None)
    parser.add_argument('--inr_decoder__num_hidden_layers', type=int, default=None)
    parser.add_argument('--inr_decoder__latent_dim', type=int, nargs='+', default=None)
    parser.add_argument('--inr_decoder__tf_dim', type=int, default=None)
    
    # Atlas generation
    parser.add_argument('--atlas_gen__resolution', type=int, default=None)
    
    args = parser.parse_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    return cmd_args


def main():
    """Main entry point."""
    cmd_args = parse_cmd_args()
    
    # Optional: run data preparation first
    if cmd_args.pop('prepare_data', False):
        print("=" * 60)
        print("Phase 1: Data Preparation")
        print("=" * 60)
        from data_loading.prepare_data import main as prepare_main
        sys.argv = ['prepare_data.py',
                     '--data_dir', cmd_args.get('data_dir', './data'),
                     '--resolution', str(cmd_args.get('voxel_resolution', 128))]
        prepare_main()
        print()
    
    # Load configs and start training
    print("=" * 60)
    print("Phase 2: Atlas Training")
    print("=" * 60)
    args = load_configs(cmd_args)
    
    # Import and run atlas builder
    from build_bone_atlas import BoneAtlasBuilder
    atlas_builder = BoneAtlasBuilder(args)
    
    print("\n✓ Atlas training complete!")
    print(f"  Output: {args['output_dir']}")


if __name__ == '__main__':
    main()
