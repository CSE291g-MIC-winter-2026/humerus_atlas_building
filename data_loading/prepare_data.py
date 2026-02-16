"""
prepare_data.py — Batch preprocessing: discover STL files, voxelize to SDF,
generate metadata TSV and subject_ids YAML.

Usage:
    python data_loading/prepare_data.py --data_dir ./data --output_dir ./data/processed --resolution 128
"""

import os
import re
import glob
import argparse
import numpy as np
import yaml
from tqdm import tqdm

from data_loading.mesh_to_volume import (
    stl_to_sdf_with_metadata,
    combine_fragments,
    sample_points_for_side_check,
    infer_mirror_against_template,
)


def discover_subjects(data_dir: str) -> list:
    """
    Walk the data directory and enumerate all subjects.
    
    Detects three types of folders:
    - patient_XX: fractured bone fragments
    - GT XX: manual ground-truth reconstructions
    - template_control: healthy reference bone
    
    Returns:
        List of subject dicts with keys:
            subject_id, folder_path, stl_files, subject_type, n_fragments
    """
    subjects = []
    
    for entry in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        
        # Find all .stl files in the folder
        stl_files = sorted(glob.glob(os.path.join(folder_path, '*.stl')))
        if len(stl_files) == 0:
            continue
        
        # Determine subject type
        if entry.startswith('patient_'):
            subject_type = 'patient'
            subject_id = entry  # e.g., 'patient_00'
        elif entry.startswith('GT'):
            subject_type = 'gt'
            # Normalize ID: 'GT 4' -> 'gt_04'
            num = re.findall(r'\d+', entry)
            subject_id = f"gt_{int(num[0]):02d}" if num else entry.lower().replace(' ', '_')
        elif entry.startswith('template'):
            subject_type = 'template'
            subject_id = 'template_control'
        else:
            subject_type = 'unknown'
            subject_id = entry.lower().replace(' ', '_')
        
        subjects.append({
            'subject_id': subject_id,
            'folder_path': folder_path,
            'stl_files': stl_files,
            'subject_type': subject_type,
            'n_fragments': len(stl_files),
        })
    
    print(f"Discovered {len(subjects)} subjects:")
    for s in subjects:
        print(f"  {s['subject_id']}: {s['n_fragments']} fragments ({s['subject_type']})")
    
    return subjects


def process_subjects(
    subjects: list,
    output_dir: str,
    resolution: int = 128,
    canonicalize_side: bool = False,
    mirror_gain_threshold: float = 0.97,
):
    """
    Process all subjects: combine fragments, voxelize to SDF, save as .npy.
    
    Args:
        subjects: List of subject dicts from discover_subjects()
        output_dir: Directory to save processed .npy files
        resolution: Voxel grid resolution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = []

    template_points = None
    template_subject_id = None
    if canonicalize_side:
        template_subjects = [s for s in subjects if s['subject_type'] == 'template']
        if len(template_subjects) == 0:
            raise ValueError("Side canonicalization requested but no template subject found.")

        template_subject = template_subjects[0]
        template_subject_id = template_subject['subject_id']
        template_mesh = combine_fragments(template_subject['stl_files'])
        template_points = sample_points_for_side_check(template_mesh, n_points=5000)

        print(
            f"Side canonicalization enabled: template={template_subject_id}, "
            f"mirror_gain_threshold={mirror_gain_threshold:.3f}"
        )
    
    for subj in tqdm(subjects, desc="Voxelizing subjects"):
        subject_id = subj['subject_id']
        stl_files = subj['stl_files']
        
        print(f"\nProcessing {subject_id} ({len(stl_files)} fragments)...")
        
        try:
            side_info = {
                'mirror_left': False,
                'orig_cost': np.nan,
                'mirrored_cost': np.nan,
                'ratio': np.nan,
            }

            if canonicalize_side and subj['subject_type'] != 'template':
                mesh_for_side = combine_fragments(stl_files)
                side_info = infer_mirror_against_template(
                    mesh_for_side,
                    template_points=template_points,
                    n_points=3500,
                    mirror_gain_threshold=mirror_gain_threshold,
                )
                print(
                    f"  Side check: orig={side_info['orig_cost']:.6f}, "
                    f"mirror={side_info['mirrored_cost']:.6f}, "
                    f"ratio={side_info['ratio']:.3f}, "
                    f"mirror={side_info['mirror_left']}"
                )

            result = stl_to_sdf_with_metadata(
                stl_files,
                resolution=resolution,
                mirror_left=side_info['mirror_left']
            )
            
            sdf = result['sdf']
            
            # Save SDF volume
            sdf_path = os.path.join(output_dir, f"{subject_id}_sdf.npy")
            np.save(sdf_path, sdf)
            
            # Compute some statistics
            n_inside = (sdf < 0).sum()
            fill_ratio = n_inside / sdf.size
            
            metadata.append({
                'subject_id': subject_id,
                'sdf_path': sdf_path,
                'subject_type': subj['subject_type'],
                'n_fragments': subj['n_fragments'],
                'fill_ratio': float(fill_ratio),
                'center_x': float(result['center_of_mass'][0]),
                'center_y': float(result['center_of_mass'][1]),
                'center_z': float(result['center_of_mass'][2]),
                'scale': float(result['scale']),
                'mirrored': result['mirrored'],
                'canonicalized_to_template': bool(canonicalize_side),
                'template_subject_id': template_subject_id if canonicalize_side else '',
                'side_orig_cost': float(side_info['orig_cost']),
                'side_mirrored_cost': float(side_info['mirrored_cost']),
                'side_ratio_mirror_over_orig': float(side_info['ratio']),
            })
            
            print(f"  Saved: {sdf_path} (shape={sdf.shape}, fill={fill_ratio:.3f})")
            
        except Exception as e:
            print(f"  ERROR processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
    
    return metadata


def save_tsv(metadata: list, output_path: str):
    """Save metadata as TSV file compatible with CINeMA's data loading."""
    import pandas as pd
    df = pd.DataFrame(metadata)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved metadata TSV: {output_path} ({len(df)} subjects)")


def save_subject_ids(metadata: list, output_path: str,
                     train_ratio: float = 0.8, seed: int = 42):
    """
    Generate train/val split and save as subject_ids YAML.
    
    Template subjects always go to training.
    GT and patient subjects are split randomly.
    """
    np.random.seed(seed)
    
    # Separate template vs. trainable subjects
    template_ids = [m['subject_id'] for m in metadata if m['subject_type'] == 'template']
    other_ids = [m['subject_id'] for m in metadata if m['subject_type'] != 'template']
    
    # Shuffle and split
    np.random.shuffle(other_ids)
    n_train = max(1, int(len(other_ids) * train_ratio))
    
    train_ids = template_ids + other_ids[:n_train]
    val_ids = other_ids[n_train:] if n_train < len(other_ids) else other_ids[-1:]
    
    subject_ids = {
        'humerus_fracture': {
            'subject_ids': {
                'train': train_ids,
                'val': val_ids,
            }
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(subject_ids, f, default_flow_style=False)
    
    print(f"Saved subject IDs: {output_path}")
    print(f"  Train: {len(train_ids)} subjects")
    print(f"  Val: {len(val_ids)} subjects")


def main():
    parser = argparse.ArgumentParser(description="Prepare bone SDF data from STL meshes")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root data directory containing patient/GT/template folders')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for processed .npy SDF files')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Voxel grid resolution per axis')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of subjects for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/val split')
    parser.add_argument('--canonicalize_side', action='store_true',
                        help='Infer side against template and mirror to a common side')
    parser.add_argument('--mirror_gain_threshold', type=float, default=0.97,
                        help='Mirror only if mirror/original ICP cost ratio is below this threshold')
    args = parser.parse_args()
    
    # Step 1: Discover all subjects
    subjects = discover_subjects(args.data_dir)
    
    # Step 2: Process (voxelize) all subjects
    metadata = process_subjects(
        subjects,
        args.output_dir,
        args.resolution,
        canonicalize_side=args.canonicalize_side,
        mirror_gain_threshold=args.mirror_gain_threshold,
    )
    
    # Step 3: Save metadata TSV
    tsv_path = os.path.join(args.output_dir, 'subjects.tsv')
    save_tsv(metadata, tsv_path)
    
    # Step 4: Generate train/val split
    ids_path = os.path.join(os.path.dirname(args.output_dir), 'configs', 'subject_ids.yaml')
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    save_subject_ids(metadata, ids_path, args.train_ratio, args.seed)
    
    print("\n✓ Data preparation complete!")
    print(f"  SDF volumes: {args.output_dir}")
    print(f"  Metadata: {tsv_path}")
    print(f"  Subject IDs: {ids_path}")


if __name__ == '__main__':
    main()
