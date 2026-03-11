"""
Evaluate atlas quality with plain mesh distances.

This script computes direct Chamfer and HD95 distances between atlas and each
subject point cloud. No bridge composition or ICP registration is used.

Usage example:
python evals/evaluate_atlas_bridge.py \
    --atlas-mesh output/bone_20260215_142952_loc/atlas/atlas_mesh_ep2999.stl \
    --data-dir data --subject-types gt --canonicalize-side
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import trimesh


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_loading.mesh_to_volume import (
    combine_fragments,
    center_and_normalize,
    infer_mirror_against_template,
    sample_points_for_side_check,
)
from evals.atlas_metrics import (
    chamfer_distance_symmetric,
    hausdorff95_distance,
    summarize,
)
from evals.report_transform_alignment import run_report
from models.inr_decoder import embed2affine


@dataclass
class SubjectMesh:
    subject_id: str
    subject_type: str
    mesh: trimesh.Trimesh
    sampled_points: np.ndarray


def _subject_ids_from_processed(processed_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
    return [os.path.splitext(os.path.basename(f))[0].replace("_sdf", "") for f in files]


def _build_learned_transform_map(checkpoint_path: str, processed_dir: str | None) -> tuple[dict[str, np.ndarray], str]:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transforms = checkpoint["transformations"].detach().float()

    if processed_dir is None:
        processed_dir = checkpoint["args"]["dataset"]["processed_dir"]
    assert processed_dir is not None

    subject_ids = _subject_ids_from_processed(processed_dir)
    if len(subject_ids) != transforms.shape[0]:
        raise ValueError(
            f"Subject/transform mismatch for learned transforms: {len(subject_ids)} ids vs {transforms.shape[0]} transforms"
        )

    R, t = embed2affine(transforms[:, :6])
    R_np = R.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()

    transform_map = {}
    for i, sid in enumerate(subject_ids):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R_np[i]
        mat[:3, 3] = t_np[i]
        transform_map[sid] = mat

    return transform_map, processed_dir


def _normalize_subject_id(folder_name: str) -> tuple[str, str]:
    if folder_name.startswith("patient_"):
        return folder_name, "patient"

    if folder_name.startswith("GT"):
        numbers = re.findall(r"\d+", folder_name)
        idx = int(numbers[0]) if numbers else -1
        return (f"gt_{idx:02d}" if idx >= 0 else folder_name.lower().replace(" ", "_"), "gt")

    if folder_name.startswith("template"):
        return "template_control", "template"

    return folder_name.lower().replace(" ", "_"), "unknown"


def discover_subject_folders(data_dir: str, accepted_types: set[str]) -> list[dict]:
    subjects: list[dict] = []

    for entry in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, entry)
        if not os.path.isdir(folder_path):
            continue

        stl_files = sorted(glob.glob(os.path.join(folder_path, "*.stl")))
        if not stl_files:
            continue

        subject_id, subject_type = _normalize_subject_id(entry)
        if "all" not in accepted_types and subject_type not in accepted_types:
            continue

        subjects.append(
            {
                "subject_id": subject_id,
                "subject_type": subject_type,
                "folder_path": folder_path,
                "stl_files": stl_files,
            }
        )

    return subjects


def load_subject_meshes(
    subject_specs: list[dict],
    n_points: int,
    canonicalize_side: bool,
    mirror_gain_threshold: float,
    learned_transform_map: dict[str, np.ndarray] | None,
    normalize_meshes: bool,
) -> list[SubjectMesh]:
    subject_meshes: list[SubjectMesh] = []

    template_points = None
    if canonicalize_side:
        templates = [s for s in subject_specs if s["subject_type"] == "template"]
        if not templates:
            raise ValueError(
                "--canonicalize-side requires a template subject folder (e.g. data/template_control)."
            )
        template_mesh = combine_fragments(templates[0]["stl_files"])
        template_points = np.asarray(
            sample_points_for_side_check(template_mesh, n_points=5000),
            dtype=np.float64,
        )

    for spec in subject_specs:
        mesh = combine_fragments(spec["stl_files"])

        if normalize_meshes:
            mesh, _, _ = center_and_normalize(mesh)

        if canonicalize_side and spec["subject_type"] != "template":
            assert template_points is not None
            side_info = infer_mirror_against_template(
                mesh,
                template_points=template_points,
                n_points=min(3500, n_points),
                mirror_gain_threshold=mirror_gain_threshold,
            )
            if side_info["mirror_left"]:
                mesh = mesh.copy()
                mesh.vertices[:, 0] *= -1.0
                mesh.faces = mesh.faces[:, ::-1]

        sampled_points = np.asarray(mesh.sample(n_points), dtype=np.float64)

        if learned_transform_map is not None and spec["subject_id"] in learned_transform_map:
            sampled_points = np.asarray(
                trimesh.transform_points(sampled_points, learned_transform_map[spec["subject_id"]]),
                dtype=np.float64,
            )

        subject_meshes.append(
            SubjectMesh(
                subject_id=spec["subject_id"],
                subject_type=spec["subject_type"],
                mesh=mesh,
                sampled_points=sampled_points,
            )
        )

    return subject_meshes


def evaluate(
    atlas_mesh_path: str,
    data_dir: str,
    subject_types: set[str],
    n_surface_points: int,
    seed: int,
    canonicalize_side: bool,
    mirror_gain_threshold: float,
    checkpoint_path: str | None,
    processed_dir: str | None,
    apply_learned_transforms: bool,
    normalize_meshes: bool,
) -> dict:
    np.random.seed(seed)

    atlas_mesh = trimesh.load(atlas_mesh_path, force="mesh")
    if isinstance(atlas_mesh, trimesh.Scene):
        atlas_mesh = trimesh.util.concatenate(
            [g for g in atlas_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if not isinstance(atlas_mesh, trimesh.Trimesh):
        raise ValueError(f"Could not parse atlas mesh at {atlas_mesh_path}")

    if normalize_meshes:
        atlas_mesh, _, _ = center_and_normalize(atlas_mesh)

    atlas_points = np.asarray(atlas_mesh.sample(n_surface_points), dtype=np.float64)

    learned_transform_map = None
    learned_transform_info = {
        "enabled": bool(apply_learned_transforms),
        "checkpoint": checkpoint_path,
        "processed_dir": processed_dir,
        "applied_subjects": 0,
        "missing_subjects": [],
    }
    if apply_learned_transforms:
        if checkpoint_path is None:
            raise ValueError("--apply-learned-transforms requires --checkpoint")
        learned_transform_map, resolved_processed = _build_learned_transform_map(
            checkpoint_path=checkpoint_path,
            processed_dir=processed_dir,
        )
        learned_transform_info["processed_dir"] = resolved_processed

    subject_specs = discover_subject_folders(data_dir, subject_types)
    subjects = load_subject_meshes(
        subject_specs,
        n_points=n_surface_points,
        canonicalize_side=canonicalize_side,
        mirror_gain_threshold=mirror_gain_threshold,
        learned_transform_map=learned_transform_map,
        normalize_meshes=normalize_meshes,
    )

    if apply_learned_transforms and learned_transform_map is not None:
        ids = [s.subject_id for s in subjects]
        learned_transform_info["applied_subjects"] = int(sum(sid in learned_transform_map for sid in ids))
        learned_transform_info["missing_subjects"] = [sid for sid in ids if sid not in learned_transform_map]

    if len(subjects) < 1:
        raise ValueError("Need at least one subject for distance evaluation.")

    atlas_to_subject_chamfer = []
    atlas_to_subject_hd95 = []
    ids = [s.subject_id for s in subjects]

    for subj in subjects:
        atlas_to_subject_chamfer.append(
            chamfer_distance_symmetric(atlas_points, subj.sampled_points)
        )
        atlas_to_subject_hd95.append(
            hausdorff95_distance(atlas_points, subj.sampled_points)
        )

    results = {
        "method": "plain_atlas_subject_mesh_distance",
        "inputs": {
            "atlas_mesh": atlas_mesh_path,
            "data_dir": data_dir,
            "subject_types": sorted(subject_types),
            "num_subjects": len(subjects),
            "subject_ids": ids,
            "n_surface_points": n_surface_points,
            "seed": seed,
            "canonicalize_side": canonicalize_side,
            "mirror_gain_threshold": mirror_gain_threshold,
            "learned_transform_correction": learned_transform_info,
            "normalize_meshes": normalize_meshes,
        },
        "metrics": {
            "atlas_to_subject": {
                "chamfer": summarize(atlas_to_subject_chamfer),
                "hd95": summarize(atlas_to_subject_hd95),
            },
        },
    }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate atlas quality with plain Chamfer/HD95 metrics")
    parser.add_argument(
        "--atlas-mesh",
        type=str,
        default="output/bone_20260215_142952_loc/atlas/atlas_mesh_ep2999.stl",
        help="Path to atlas STL mesh",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing raw subject folders (GT xx, patient_xx, template_control)",
    )
    parser.add_argument(
        "--subject-types",
        type=str,
        nargs="+",
        default=["gt"],
        help="Subset of subject types: gt patient template unknown all",
    )
    parser.add_argument(
        "--n-surface-points",
        type=int,
        default=4000,
        help="Number of surface points sampled per mesh for metric computation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--canonicalize-side",
        action="store_true",
        help="Mirror subjects to template side before evaluation",
    )
    parser.add_argument(
        "--mirror-gain-threshold",
        type=float,
        default=0.97,
        help="Threshold for side mirroring decision (mirrored_cost/original_cost)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="evals/results",
        help="Directory for JSON results",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="atlas_eval_ep2999_plain_distance.json",
        help="Output JSON filename",
    )
    parser.add_argument(
        "--no-normalize-meshes",
        action="store_true",
        help="Disable canonical centering/scaling prior to point-cloud sampling",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint containing learned per-subject transforms",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Processed SDF directory used to map checkpoint transforms to subject IDs",
    )
    parser.add_argument(
        "--apply-learned-transforms",
        action="store_true",
        help="Apply learned checkpoint transforms to subject points before atlas evaluation",
    )
    parser.add_argument(
        "--save-transform-report",
        action="store_true",
        help="Also generate per-subject transform report JSON/CSV in the same run",
    )
    parser.add_argument(
        "--transform-report-prefix",
        type=str,
        default="transform_alignment_from_eval",
        help="Prefix for optional transform report outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subject_types = set([s.lower() for s in args.subject_types])
    valid = {"gt", "patient", "template", "unknown", "all"}
    invalid = sorted(subject_types - valid)
    if invalid:
        raise ValueError(f"Invalid subject type(s): {invalid}. Valid options: {sorted(valid)}")

    results = evaluate(
        atlas_mesh_path=args.atlas_mesh,
        data_dir=args.data_dir,
        subject_types=subject_types,
        n_surface_points=args.n_surface_points,
        seed=args.seed,
        canonicalize_side=args.canonicalize_side,
        mirror_gain_threshold=args.mirror_gain_threshold,
        checkpoint_path=args.checkpoint,
        processed_dir=args.processed_dir,
        apply_learned_transforms=args.apply_learned_transforms,
        normalize_meshes=not args.no_normalize_meshes,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 72)
    print("Atlas Evaluation (Plain Mesh Distances)")
    print("=" * 72)
    print(f"Atlas mesh: {results['inputs']['atlas_mesh']}")
    print(f"Subjects:   {results['inputs']['num_subjects']} ({', '.join(results['inputs']['subject_ids'])})")
    print("-")

    atlas_to_subject_chamfer = results["metrics"]["atlas_to_subject"]["chamfer"]["mean"]
    atlas_to_subject_hd95 = results["metrics"]["atlas_to_subject"]["hd95"]["mean"]

    print(f"Atlas-to-subject Chamfer:     {atlas_to_subject_chamfer:.6f}")
    print(f"Atlas-to-subject HD95:        {atlas_to_subject_hd95:.6f}")
    if args.apply_learned_transforms:
        applied = results["inputs"]["learned_transform_correction"]["applied_subjects"]
        print(f"Learned-transform correction: enabled (applied to {applied} subjects)")
    print(f"Saved detailed report to: {save_path}")

    if args.save_transform_report:
        if args.checkpoint is None:
            raise ValueError("--save-transform-report requires --checkpoint")
        transform_report = run_report(
            checkpoint_path=args.checkpoint,
            atlas_mesh_path=args.atlas_mesh,
            processed_dir=args.processed_dir,
            n_surface_points=args.n_surface_points,
            surface_band=0.03,
            rotation_outlier_deg=40.0,
        )
        report_json = os.path.join(args.save_dir, f"{args.transform_report_prefix}.json")
        report_csv = os.path.join(args.save_dir, f"{args.transform_report_prefix}.csv")
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(transform_report, f, indent=2)
        import csv

        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(transform_report["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(transform_report["rows"])
        print(f"Saved transform report JSON: {report_json}")
        print(f"Saved transform report CSV:  {report_csv}")


if __name__ == "__main__":
    main()
