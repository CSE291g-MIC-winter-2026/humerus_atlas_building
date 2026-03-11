"""Create side-by-side point-cloud overlays for GT meshes vs atlas mesh."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_loading.mesh_to_volume import combine_fragments
from data_loading.mesh_to_volume import center_and_normalize
from models.inr_decoder import embed2affine
from evals.atlas_metrics import chamfer_distance_symmetric


def _normalize_subject_id(folder_name: str) -> tuple[str, str]:
    if folder_name.startswith("GT"):
        numbers = re.findall(r"\d+", folder_name)
        idx = int(numbers[0]) if numbers else -1
        return (f"gt_{idx:02d}" if idx >= 0 else folder_name.lower().replace(" ", "_"), "gt")
    if folder_name.startswith("patient_"):
        return folder_name, "patient"
    if folder_name.startswith("template"):
        return "template_control", "template"
    return folder_name.lower().replace(" ", "_"), "unknown"


def discover_gt_subjects(data_dir: str) -> list[dict]:
    subjects = []
    for entry in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        subject_id, subject_type = _normalize_subject_id(entry)
        if subject_type != "gt":
            continue
        stl_files = sorted(glob.glob(os.path.join(folder_path, "*.stl")))
        if not stl_files:
            continue
        subjects.append({
            "subject_id": subject_id,
            "folder_path": folder_path,
            "stl_files": stl_files,
        })
    return subjects


def _subject_ids_from_processed(processed_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
    return [os.path.splitext(os.path.basename(f))[0].replace("_sdf", "") for f in files]


def _build_learned_transform_map(
    checkpoint_path: str,
    processed_dir: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    transforms = ckpt["transformations"].detach().float()

    if processed_dir is None:
        processed_dir = ckpt["args"]["dataset"]["processed_dir"]
    assert processed_dir is not None

    subject_ids = _subject_ids_from_processed(processed_dir)
    if len(subject_ids) != transforms.shape[0]:
        raise ValueError(
            f"Subject/transform mismatch: {len(subject_ids)} ids vs {transforms.shape[0]} transforms"
        )

    R, t = embed2affine(transforms[:, :6])
    R_np = R.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()

    transform_map = {}
    transform_inv_map = {}
    for i, sid in enumerate(subject_ids):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R_np[i]
        mat[:3, 3] = t_np[i]
        transform_map[sid] = mat

        mat_inv = np.eye(4, dtype=np.float64)
        mat_inv[:3, :3] = R_np[i].T
        mat_inv[:3, 3] = -(R_np[i].T @ t_np[i])
        transform_inv_map[sid] = mat_inv

    return transform_map, transform_inv_map


def set_axes_equal(ax: Any, points_all: np.ndarray) -> None:
    mins = points_all.min(axis=0)
    maxs = points_all.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = 0.5 * float(np.max(maxs - mins) + 1e-8)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 10 GT point clouds (blue) vs atlas (red)")
    parser.add_argument(
        "--atlas-mesh",
        type=str,
        default="output/bone_20260215_142952_loc/atlas/atlas_mesh_ep2999.stl",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-subjects", type=int, default=10)
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="evals/results/gt_vs_atlas_pointclouds")
    parser.add_argument(
        "--no-normalize-meshes",
        action="store_true",
        help="Disable canonical centering/scaling before sampling",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file for learned subject transforms",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Processed SDF directory used to map transforms to subject IDs",
    )
    parser.add_argument(
        "--apply-learned-transforms",
        action="store_true",
        help="Apply learned transforms to GT point clouds before plotting",
    )
    parser.add_argument(
        "--transform-mode",
        type=str,
        default="forward",
        choices=["forward", "inverse", "best"],
        help="How to use learned transform when --apply-learned-transforms is enabled",
    )
    parser.add_argument(
        "--icp-refine",
        action="store_true",
        help="Apply a final rigid ICP refinement to GT points for visualization",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    transform_map = None
    transform_inv_map = None
    if args.apply_learned_transforms:
        if args.checkpoint is None:
            raise ValueError("--apply-learned-transforms requires --checkpoint")
        transform_map, transform_inv_map = _build_learned_transform_map(
            args.checkpoint,
            args.processed_dir,
        )

    atlas_mesh = trimesh.load(args.atlas_mesh, force="mesh")
    if isinstance(atlas_mesh, trimesh.Scene):
        atlas_mesh = trimesh.util.concatenate(
            [g for g in atlas_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if not isinstance(atlas_mesh, trimesh.Trimesh):
        raise ValueError(f"Could not parse atlas mesh: {args.atlas_mesh}")
    if not args.no_normalize_meshes:
        atlas_mesh, _, _ = center_and_normalize(atlas_mesh)
    atlas_pts = np.asarray(atlas_mesh.sample(args.n_points), dtype=np.float64)

    gt_subjects = discover_gt_subjects(args.data_dir)[: args.n_subjects]
    if len(gt_subjects) == 0:
        raise ValueError("No GT subjects found under data-dir")

    for i, subject in enumerate(gt_subjects, start=1):
        gt_mesh = combine_fragments(subject["stl_files"])
        if not args.no_normalize_meshes:
            gt_mesh, _, _ = center_and_normalize(gt_mesh)
        gt_pts = np.asarray(gt_mesh.sample(args.n_points), dtype=np.float64)
        mode_used = "none"
        if transform_map is not None and transform_inv_map is not None and subject["subject_id"] in transform_map:
            sid = subject["subject_id"]
            pts_fwd = np.asarray(trimesh.transform_points(gt_pts, transform_map[sid]), dtype=np.float64)
            pts_inv = np.asarray(trimesh.transform_points(gt_pts, transform_inv_map[sid]), dtype=np.float64)

            if args.transform_mode == "forward":
                gt_pts = pts_fwd
                mode_used = "forward"
            elif args.transform_mode == "inverse":
                gt_pts = pts_inv
                mode_used = "inverse"
            else:
                cd_raw = chamfer_distance_symmetric(gt_pts, atlas_pts)
                cd_fwd = chamfer_distance_symmetric(pts_fwd, atlas_pts)
                cd_inv = chamfer_distance_symmetric(pts_inv, atlas_pts)
                if cd_fwd <= cd_raw and cd_fwd <= cd_inv:
                    gt_pts = pts_fwd
                    mode_used = "best=forward"
                elif cd_inv <= cd_raw and cd_inv <= cd_fwd:
                    gt_pts = pts_inv
                    mode_used = "best=inverse"
                else:
                    mode_used = "best=raw"

        if args.icp_refine:
            m_icp, pts_icp, _ = trimesh.registration.icp(
                gt_pts,
                atlas_pts,
                reflection=False,
                translation=True,
                scale=False,
                return_cost=True,
            )
            _ = m_icp
            gt_pts = np.asarray(pts_icp, dtype=np.float64)
            mode_used = f"{mode_used}+icp" if mode_used != "none" else "icp"

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax_any: Any = ax

        ax_any.scatter(
            gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
            c="blue", s=1, alpha=0.35, label=f"{subject['subject_id']} (GT)"
        )
        ax_any.scatter(
            atlas_pts[:, 0], atlas_pts[:, 1], atlas_pts[:, 2],
            c="red", s=1, alpha=0.35, label="atlas"
        )

        all_pts = np.vstack([gt_pts, atlas_pts])
        set_axes_equal(ax, all_pts)
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        title_suffix = ""
        if args.apply_learned_transforms:
            title_suffix = f" (NN {mode_used})"
        elif args.icp_refine:
            title_suffix = " (icp)"
        ax.set_title(f"GT vs Atlas: {subject['subject_id']}{title_suffix}")
        ax.legend(loc="upper right")

        out_path = os.path.join(args.out_dir, f"{i:02d}_{subject['subject_id']}_vs_atlas.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"Done. Wrote {len(gt_subjects)} images to {args.out_dir}")


if __name__ == "__main__":
    main()
