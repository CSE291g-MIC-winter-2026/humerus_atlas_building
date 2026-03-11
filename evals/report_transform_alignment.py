"""Generate per-subject before/after alignment report using learned transforms."""

from __future__ import annotations

import argparse
import csv
import json
import os
import glob
import sys
from typing import Any

import numpy as np
import torch
import trimesh


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.inr_decoder import embed2affine
from evals.atlas_metrics import chamfer_distance_symmetric, hausdorff95_distance


def _subject_ids_from_processed(processed_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
    ids = [os.path.splitext(os.path.basename(f))[0].replace("_sdf", "") for f in files]
    return ids


def _sample_surface_points_from_sdf(
    sdf_path: str,
    n_points: int,
    surface_band: float,
) -> np.ndarray:
    sdf = np.load(sdf_path)
    res = sdf.shape[0]

    near = np.abs(sdf) <= surface_band
    idx = np.argwhere(near)

    if idx.shape[0] < 128:
        # Fallback: choose the voxels with smallest |SDF| values
        flat = np.abs(sdf).reshape(-1)
        k = min(max(n_points * 3, 1024), flat.shape[0])
        choice = np.argpartition(flat, k - 1)[:k]
        idx = np.column_stack(np.unravel_index(choice, sdf.shape))

    replace = idx.shape[0] < n_points
    pick = np.random.choice(idx.shape[0], size=n_points, replace=replace)
    vox = idx[pick].astype(np.float64)

    # Convert voxel indices to normalized coordinate space used by the INR.
    coords = -1.0 + 2.0 * (vox / float(res - 1))
    return coords


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis


def _axis_angle_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    dot = float(np.clip(np.abs(np.dot(axis_a, axis_b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _load_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def run_report(
    checkpoint_path: str,
    atlas_mesh_path: str,
    processed_dir: str | None,
    n_surface_points: int,
    surface_band: float,
    rotation_outlier_deg: float,
) -> dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint_path)
    transforms = ckpt["transformations"].detach().float()

    if processed_dir is None:
        processed_dir = ckpt["args"]["dataset"]["processed_dir"]

    subject_ids = _subject_ids_from_processed(processed_dir)
    if len(subject_ids) != transforms.shape[0]:
        raise ValueError(
            f"Subject count mismatch: {len(subject_ids)} ids vs {transforms.shape[0]} transforms"
        )

    atlas_mesh = trimesh.load(atlas_mesh_path, force="mesh")
    if isinstance(atlas_mesh, trimesh.Scene):
        atlas_mesh = trimesh.util.concatenate(
            [g for g in atlas_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    atlas_points = np.asarray(atlas_mesh.sample(n_surface_points), dtype=np.float64)
    atlas_axis = _principal_axis(atlas_points)

    R, t = embed2affine(transforms[:, :6])
    R_np = R.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()

    rows = []
    for i, subject_id in enumerate(subject_ids):
        sdf_path = os.path.join(processed_dir, f"{subject_id}_sdf.npy")
        src_points = _sample_surface_points_from_sdf(
            sdf_path=sdf_path,
            n_points=n_surface_points,
            surface_band=surface_band,
        )

        rot_vec = transforms[i, :3]
        tr_vec = transforms[i, 3:6]
        rot_deg = float(torch.linalg.norm(rot_vec) * 180.0 / np.pi)
        tr_norm = float(torch.linalg.norm(tr_vec))

        src_axis_before = _principal_axis(src_points)
        ori_before_deg = _axis_angle_deg(src_axis_before, atlas_axis)

        src_points_after = (R_np[i] @ src_points.T).T + t_np[i][None, :]
        src_axis_after = _principal_axis(src_points_after)
        ori_after_deg = _axis_angle_deg(src_axis_after, atlas_axis)

        chamfer_before = chamfer_distance_symmetric(src_points, atlas_points)
        chamfer_after = chamfer_distance_symmetric(src_points_after, atlas_points)
        hd95_before = hausdorff95_distance(src_points, atlas_points)
        hd95_after = hausdorff95_distance(src_points_after, atlas_points)

        row = {
            "index": i,
            "subject_id": subject_id,
            "rotation_deg": rot_deg,
            "translation_norm": tr_norm,
            "orientation_before_deg": ori_before_deg,
            "orientation_after_deg": ori_after_deg,
            "orientation_delta_deg": ori_after_deg - ori_before_deg,
            "chamfer_before": chamfer_before,
            "chamfer_after": chamfer_after,
            "chamfer_delta": chamfer_after - chamfer_before,
            "hd95_before": hd95_before,
            "hd95_after": hd95_after,
            "hd95_delta": hd95_after - hd95_before,
            "rotation_outlier": rot_deg >= rotation_outlier_deg,
        }
        rows.append(row)

    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "checkpoint": checkpoint_path,
        "atlas_mesh": atlas_mesh_path,
        "processed_dir": processed_dir,
        "n_subjects": len(rows),
        "n_surface_points": n_surface_points,
        "surface_band": surface_band,
        "rotation_outlier_deg": rotation_outlier_deg,
        "means": {
            "rotation_deg": _mean("rotation_deg"),
            "translation_norm": _mean("translation_norm"),
            "orientation_before_deg": _mean("orientation_before_deg"),
            "orientation_after_deg": _mean("orientation_after_deg"),
            "chamfer_before": _mean("chamfer_before"),
            "chamfer_after": _mean("chamfer_after"),
            "hd95_before": _mean("hd95_before"),
            "hd95_after": _mean("hd95_after"),
        },
        "counts": {
            "orientation_improved": int(sum(r["orientation_delta_deg"] < 0 for r in rows)),
            "chamfer_improved": int(sum(r["chamfer_delta"] < 0 for r in rows)),
            "hd95_improved": int(sum(r["hd95_delta"] < 0 for r in rows)),
            "rotation_outliers": int(sum(r["rotation_outlier"] for r in rows)),
        },
        "top_rotation_outliers": [
            {
                "subject_id": r["subject_id"],
                "rotation_deg": r["rotation_deg"],
                "chamfer_delta": r["chamfer_delta"],
                "orientation_delta_deg": r["orientation_delta_deg"],
            }
            for r in sorted(rows, key=lambda x: x["rotation_deg"], reverse=True)[:10]
        ],
    }

    return {"summary": summary, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report per-subject before/after orientation alignment from learned transforms"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/bone_20260215_142952_loc/checkpoint_epoch_2999.pth",
    )
    parser.add_argument(
        "--atlas-mesh",
        type=str,
        default="output/bone_20260215_142952_loc/atlas/atlas_mesh_ep2999.stl",
    )
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--n-surface-points", type=int, default=2500)
    parser.add_argument("--surface-band", type=float, default=0.03)
    parser.add_argument("--rotation-outlier-deg", type=float, default=40.0)
    parser.add_argument("--save-dir", type=str, default="evals/results")
    parser.add_argument("--save-prefix", type=str, default="transform_alignment_ep2999")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(42)

    report = run_report(
        checkpoint_path=args.checkpoint,
        atlas_mesh_path=args.atlas_mesh,
        processed_dir=args.processed_dir,
        n_surface_points=args.n_surface_points,
        surface_band=args.surface_band,
        rotation_outlier_deg=args.rotation_outlier_deg,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    json_path = os.path.join(args.save_dir, f"{args.save_prefix}.json")
    csv_path = os.path.join(args.save_dir, f"{args.save_prefix}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(report["rows"])

    s = report["summary"]
    print("=" * 72)
    print("Learned Transform Alignment Report")
    print("=" * 72)
    print(f"Checkpoint: {s['checkpoint']}")
    print(f"Atlas mesh:  {s['atlas_mesh']}")
    print(f"Subjects:    {s['n_subjects']}")
    print("-")
    print(
        f"Mean orientation angle: {s['means']['orientation_before_deg']:.3f} -> "
        f"{s['means']['orientation_after_deg']:.3f} deg"
    )
    print(
        f"Mean Chamfer:           {s['means']['chamfer_before']:.3f} -> "
        f"{s['means']['chamfer_after']:.3f}"
    )
    print(
        f"Mean HD95:              {s['means']['hd95_before']:.3f} -> "
        f"{s['means']['hd95_after']:.3f}"
    )
    print(
        f"Improved subjects (orientation/chamfer/hd95): "
        f"{s['counts']['orientation_improved']}/{s['counts']['chamfer_improved']}/{s['counts']['hd95_improved']}"
    )
    print(f"Rotation outliers (>= {s['rotation_outlier_deg']} deg): {s['counts']['rotation_outliers']}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()
