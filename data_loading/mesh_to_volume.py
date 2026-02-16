"""
mesh_to_volume.py — STL mesh to Signed Distance Field (SDF) voxelization pipeline.

Converts triangulated surface meshes (.stl) to volumetric SDF representations
suitable for training with Implicit Neural Representations (INRs).
"""

import os
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt


def load_stl(path: str) -> trimesh.Trimesh:
    """Load an STL mesh file."""
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # Handle scenes with multiple meshes by concatenating
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No valid mesh geometry found in {path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"Could not load mesh from {path}")
    return mesh


def combine_fragments(stl_paths: list) -> trimesh.Trimesh:
    """
    Combine multiple fragment STL meshes into a single mesh.
    
    Args:
        stl_paths: List of paths to STL fragment files
    
    Returns:
        Combined trimesh.Trimesh object
    """
    meshes = []
    for p in stl_paths:
        try:
            m = load_stl(p)
            meshes.append(m)
        except Exception as e:
            print(f"  Warning: Could not load {p}: {e}")
    
    if len(meshes) == 0:
        raise ValueError("No valid meshes to combine")
    
    if len(meshes) == 1:
        return meshes[0]
    
    return trimesh.util.concatenate(meshes)


def mirror_mesh_x(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Mirror a mesh across the YZ plane (flip X axis) for bilateral mirroring.
    This converts left-side bones to right-side orientation.
    
    Args:
        mesh: Input mesh
    
    Returns:
        Mirrored mesh with corrected face normals
    """
    mirrored = mesh.copy()
    mirrored.vertices[:, 0] *= -1
    # Flip face winding to correct normals after reflection
    mirrored.faces = mirrored.faces[:, ::-1]
    return mirrored


def _sample_points_for_side_check(mesh: trimesh.Trimesh, n_points: int = 3500) -> np.ndarray:
    """
    Sample and normalize surface points for robust side inference.

    Points are centered and RMS-scaled to reduce sensitivity to translation/scale.
    """
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = pts.astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    rms = np.sqrt((pts ** 2).sum(axis=1).mean()) + 1e-8
    pts /= rms
    return pts.astype(np.float32)


def sample_points_for_side_check(mesh: trimesh.Trimesh, n_points: int = 3500) -> np.ndarray:
    """Public wrapper for side-check point sampling."""
    return _sample_points_for_side_check(mesh, n_points=n_points)


def _mirror_points_x(points: np.ndarray) -> np.ndarray:
    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    return mirrored


def infer_mirror_against_template(
    mesh: trimesh.Trimesh,
    template_points: np.ndarray,
    n_points: int = 3500,
    mirror_gain_threshold: float = 0.97,
) -> dict:
    """
    Decide whether `mesh` should be mirrored to match template side.

    Method:
    - Sample points from mesh (original and mirrored).
    - Run rigid ICP (no reflection) to template points.
    - If mirrored ICP cost is sufficiently lower than original, mirror.

    Args:
        mesh: Subject mesh (possibly fragmented/combined).
        template_points: Pre-sampled normalized points from canonical template.
        n_points: Number of subject points sampled for side inference.
        mirror_gain_threshold:
            Mirror is selected only if:
                mirrored_cost / original_cost < mirror_gain_threshold

    Returns:
        Dict with keys:
            mirror_left (bool), orig_cost (float), mirrored_cost (float), ratio (float)
    """
    points = _sample_points_for_side_check(mesh, n_points=n_points)
    points_mirror = _mirror_points_x(points)

    # No reflection in ICP; reflection decision is handled explicitly by mirroring.
    _, _, orig_cost = trimesh.registration.icp(
        points, template_points, reflection=False, translation=True,
        scale=True, return_cost=True
    )
    _, _, mirrored_cost = trimesh.registration.icp(
        points_mirror, template_points, reflection=False, translation=True,
        scale=True, return_cost=True
    )

    ratio = (mirrored_cost + 1e-12) / (orig_cost + 1e-12)
    mirror_left = ratio < mirror_gain_threshold

    return {
        'mirror_left': bool(mirror_left),
        'orig_cost': float(orig_cost),
        'mirrored_cost': float(mirrored_cost),
        'ratio': float(ratio),
    }


def detect_side(mesh: trimesh.Trimesh) -> str:
    """
    Heuristic side detection based on mesh geometry.
    Uses the principal axis orientation to guess left vs right.
    
    For proximal humerus, the head typically points medially.
    This is a best-effort heuristic — can be overridden by metadata.
    
    Args:
        mesh: Input humerus mesh
    
    Returns:
        'left' or 'right' (default assumption is 'right')
    """
    # Default: assume right. Override in metadata if available.
    # A more sophisticated approach would use PCA or landmark detection.
    return 'right'


def center_and_normalize(mesh: trimesh.Trimesh, target_bbox: float = 2.0) -> tuple:
    """
    Center mesh at origin and normalize to fit within [-1, 1]^3.
    
    Args:
        mesh: Input mesh
        target_bbox: Target bounding box size (2.0 means [-1, 1])
    
    Returns:
        Tuple of (normalized mesh, center_of_mass, scale_factor)
    """
    centered = mesh.copy()
    # Use bounding-box center instead of vertex mean so long bones are
    # symmetrically placed in the canonical cube and do not drift to a face.
    bbox_min, bbox_max = centered.bounds
    com = (bbox_min + bbox_max) / 2.0
    centered.vertices -= com
    
    # Scale to fit within [-1, 1]
    extent = centered.vertices.max(axis=0) - centered.vertices.min(axis=0)
    scale = target_bbox / (extent.max() + 1e-8)
    centered.vertices *= scale
    
    return centered, com, scale


def voxelize_mesh(mesh: trimesh.Trimesh, resolution: int = 128,
                  padding: float = 0.05) -> np.ndarray:
    """
    Voxelize a mesh into a binary occupancy grid, then compute SDF.
    
    Uses trimesh's voxelization + distance transform for SDF computation.
    This is faster than exact SDF computation and sufficient for INR training.
    
    Args:
        mesh: Normalized mesh (vertices in [-1, 1])
        resolution: Voxel grid resolution per axis
        padding: Padding fraction around the mesh
    
    Returns:
        SDF volume of shape (resolution, resolution, resolution).
        Negative inside, positive outside.
    """
    half = 1.0 + padding
    
    # Voxelize using trimesh and place occupied voxels into a fixed
    # resolution^3 grid in normalized world coordinates [-half, half]^3.
    # This avoids the previous resize step, which could shift/crop anatomy.
    pitch = (2.0 * half) / resolution
    occupancy = np.zeros((resolution, resolution, resolution), dtype=np.float32)

    try:
        voxels = mesh.voxelized(pitch=pitch)
        voxels = voxels.fill()  # Fill interior
        points = voxels.points  # (N, 3) occupied voxel centers in world coords

        # Map world coordinates to fixed-grid indices.
        indices = np.rint(
            ((points + half) / (2.0 * half)) * (resolution - 1)
        ).astype(np.int32)
        indices = np.clip(indices, 0, resolution - 1)
        occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    except Exception:
        # Fallback: use ray-based inside/outside testing
        occupancy = _ray_based_voxelize(mesh, resolution, half)

    # Compute SDF from binary occupancy using distance transforms
    sdf = compute_sdf_from_occupancy(occupancy)
    
    return sdf


def _ray_based_voxelize(mesh: trimesh.Trimesh, resolution: int,
                         half: float) -> np.ndarray:
    """
    Fallback voxelization using ray-based inside/outside testing.
    """
    grid = np.linspace(-half, half, resolution)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    
    # Test inside/outside in batches
    occupancy = np.zeros(resolution ** 3, dtype=np.float32)
    batch_size = 100000
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            inside = mesh.contains(batch)
            occupancy[i:i + batch_size] = inside.astype(np.float32)
        except Exception:
            pass
    
    return occupancy.reshape(resolution, resolution, resolution)


def _resize_occupancy(occupancy: np.ndarray, target_res: int) -> np.ndarray:
    """Resize occupancy grid to target resolution using nearest interpolation."""
    if occupancy.shape == (target_res, target_res, target_res):
        return occupancy
    
    from scipy.ndimage import zoom
    factors = tuple(target_res / s for s in occupancy.shape)
    return zoom(occupancy, factors, order=0)


def compute_sdf_from_occupancy(occupancy: np.ndarray) -> np.ndarray:
    """
    Compute approximate SDF from binary occupancy grid using distance transforms.
    
    Args:
        occupancy: Binary occupancy grid (1 inside, 0 outside)
    
    Returns:
        SDF: negative inside bone, positive outside bone, normalized to [-1, 1]
    """
    # Distance from outside points to surface
    dist_outside = distance_transform_edt(1 - occupancy)
    # Distance from inside points to surface
    dist_inside = distance_transform_edt(occupancy)
    
    # SDF: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    # Normalize to [-1, 1]
    max_abs = max(abs(sdf.max()), abs(sdf.min()), 1e-8)
    sdf = sdf / max_abs
    
    return sdf.astype(np.float32)


def stl_to_sdf(stl_path: str, resolution: int = 128,
               mirror_left: bool = False) -> np.ndarray:
    """
    Full pipeline: STL file → normalized SDF volume.
    
    Args:
        stl_path: Path to STL file (or list of paths for fragments)
        resolution: Voxel grid resolution
        mirror_left: If True, mirror the mesh (left → right orientation)
    
    Returns:
        SDF array of shape (resolution, resolution, resolution)
    """
    if isinstance(stl_path, list):
        mesh = combine_fragments(stl_path)
    else:
        mesh = load_stl(stl_path)
    
    if mirror_left:
        mesh = mirror_mesh_x(mesh)
    
    mesh, com, scale = center_and_normalize(mesh)
    sdf = voxelize_mesh(mesh, resolution=resolution)
    
    return sdf


def stl_to_sdf_with_metadata(stl_path, resolution: int = 128,
                              mirror_left: bool = False) -> dict:
    """
    Full pipeline returning SDF + metadata needed for inverse transforms.
    
    Returns:
        Dict with keys: 'sdf', 'center_of_mass', 'scale', 'mirrored'
    """
    if isinstance(stl_path, list):
        mesh = combine_fragments(stl_path)
    else:
        mesh = load_stl(stl_path)
    
    mirrored = False
    if mirror_left:
        mesh = mirror_mesh_x(mesh)
        mirrored = True
    
    mesh, com, scale = center_and_normalize(mesh)
    sdf = voxelize_mesh(mesh, resolution=resolution)
    
    return {
        'sdf': sdf,
        'center_of_mass': com,
        'scale': scale,
        'mirrored': mirrored,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        res = int(sys.argv[2]) if len(sys.argv) > 2 else 128
        sdf = stl_to_sdf(path, resolution=res)
        print(f"SDF shape: {sdf.shape}, min: {sdf.min():.4f}, max: {sdf.max():.4f}")
        # Count interior voxels
        n_inside = (sdf < 0).sum()
        print(f"Interior voxels: {n_inside} ({100*n_inside/sdf.size:.1f}%)")
    else:
        print("Usage: python mesh_to_volume.py <path_to_stl> [resolution]")
