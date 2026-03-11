"""Mesh-based metrics used by atlas evaluation scripts."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def chamfer_distance_symmetric(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute symmetric Chamfer distance between two point clouds."""
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    d_ab, _ = tree_b.query(points_a, k=1)
    d_ba, _ = tree_a.query(points_b, k=1)

    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def hausdorff95_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute robust Hausdorff distance (95th percentile) between point clouds."""
    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    d_ab, _ = tree_b.query(points_a, k=1)
    d_ba, _ = tree_a.query(points_b, k=1)

    distances = np.concatenate([d_ab, d_ba], axis=0)
    return float(np.percentile(distances, 95.0))


def summarize(values: list[float]) -> dict:
    """Return mean/std/min/max summary for a list of scalar metrics."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
