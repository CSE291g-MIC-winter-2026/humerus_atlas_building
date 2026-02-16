"""
test_pipeline.py — End-to-end tests for the CINeMA-Bone pipeline.

Tests:
    1. Mesh loading and bilateral mirroring (data_loading/mesh_to_volume.py)
    2. SDF voxelization from synthetic mesh (data_loading/mesh_to_volume.py)
    3. BoneDataset coordinate sampling (data_loading/dataset.py)
    4. SIREN forward pass shapes (models/siren.py)
    5. BoneINRDecoder forward pass (models/inr_decoder.py)
    6. Loss function computation (utils.py)
    7. Grid generation and marching cubes (utils.py)
"""

import os
import sys
import tempfile
import numpy as np
import torch
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Test 1: Mesh Operations
# ===================================================================

class TestMeshToVolume:
    """Test STL mesh loading and SDF conversion."""
    
    def _create_synthetic_stl(self, tmpdir):
        """Create a synthetic sphere STL for testing."""
        import trimesh
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        path = os.path.join(tmpdir, 'test_sphere.stl')
        sphere.export(path)
        return path, sphere
    
    def test_load_stl(self, tmp_path):
        from data_loading.mesh_to_volume import load_stl
        path, _ = self._create_synthetic_stl(str(tmp_path))
        mesh = load_stl(path)
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    
    def test_mirror_mesh(self, tmp_path):
        from data_loading.mesh_to_volume import load_stl, mirror_mesh_x
        path, _ = self._create_synthetic_stl(str(tmp_path))
        mesh = load_stl(path)
        
        mirrored = mirror_mesh_x(mesh)
        
        # X coordinates should be negated
        np.testing.assert_allclose(
            mirrored.vertices[:, 0], -mesh.vertices[:, 0], atol=1e-6
        )
        # Y, Z should be unchanged
        np.testing.assert_allclose(
            mirrored.vertices[:, 1:], mesh.vertices[:, 1:], atol=1e-6
        )
    
    def test_center_and_normalize(self, tmp_path):
        from data_loading.mesh_to_volume import load_stl, center_and_normalize
        path, _ = self._create_synthetic_stl(str(tmp_path))
        mesh = load_stl(path)
        
        normalized, com, scale = center_and_normalize(mesh)
        
        # Should be roughly centered at origin
        center = normalized.vertices.mean(axis=0)
        assert np.abs(center).max() < 0.1
        
        # Should fit within [-1, 1]
        assert normalized.vertices.max() <= 1.05
        assert normalized.vertices.min() >= -1.05
    
    def test_sdf_computation(self):
        from data_loading.mesh_to_volume import compute_sdf_from_occupancy
        
        # Create a synthetic occupancy (solid cube in center)
        occ = np.zeros((32, 32, 32), dtype=np.float32)
        occ[8:24, 8:24, 8:24] = 1.0
        
        sdf = compute_sdf_from_occupancy(occ)
        
        assert sdf.shape == (32, 32, 32)
        assert sdf.min() < 0  # Interior is negative
        assert sdf.max() > 0  # Exterior is positive
        # Center should be most negative (deepest inside)
        assert sdf[16, 16, 16] < sdf[0, 0, 0]
    
    def test_stl_to_sdf(self, tmp_path):
        from data_loading.mesh_to_volume import stl_to_sdf
        path, _ = self._create_synthetic_stl(str(tmp_path))
        
        sdf = stl_to_sdf(path, resolution=32)
        
        assert sdf.shape == (32, 32, 32)
        assert sdf.dtype == np.float32
        # Should have both inside and outside voxels
        assert (sdf < 0).sum() > 0
        assert (sdf > 0).sum() > 0
    
    def test_combine_fragments(self, tmp_path):
        from data_loading.mesh_to_volume import combine_fragments
        import trimesh
        
        # Create two small sphere "fragments" at different positions
        frag1 = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
        frag1.vertices += [0.3, 0, 0]  # Offset fragment 1
        frag2 = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
        frag2.vertices -= [0.3, 0, 0]  # Offset fragment 2
        
        p1 = os.path.join(str(tmp_path), 'frag1.stl')
        p2 = os.path.join(str(tmp_path), 'frag2.stl')
        frag1.export(p1)
        frag2.export(p2)
        
        combined = combine_fragments([p1, p2])
        assert len(combined.vertices) > 0
        # Combined should have vertices from both fragments
        assert len(combined.vertices) == len(frag1.vertices) + len(frag2.vertices)


# ===================================================================
# Test 2: Dataset
# ===================================================================

class TestBoneDataset:
    """Test the PyTorch BoneDataset."""
    
    def _create_test_data(self, tmpdir, n_subjects=3, resolution=16):
        """Create synthetic SDF data for testing."""
        processed_dir = os.path.join(tmpdir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        for i in range(n_subjects):
            occ = np.zeros((resolution, resolution, resolution), dtype=np.float32)
            occ[4:12, 4:12, 4:12] = 1.0
            from data_loading.mesh_to_volume import compute_sdf_from_occupancy
            sdf = compute_sdf_from_occupancy(occ)
            np.save(os.path.join(processed_dir, f'subject_{i:02d}_sdf.npy'), sdf)
        
        return processed_dir
    
    def test_dataset_initialization(self, tmp_path):
        from data_loading.dataset import BoneDataset
        
        processed_dir = self._create_test_data(str(tmp_path))
        
        args = {
            'dataset': {
                'data_dir': str(tmp_path),
                'processed_dir': processed_dir,
                'world_bbox': [100, 100, 100],
                'surface_sample_ratio': 0.5,
                'surface_threshold': 0.05,
                'conditions': {},
            },
            'n_coords_per_subject': 1000,
        }
        
        ds = BoneDataset(args, split='train')
        assert len(ds) == 3
    
    def test_dataset_getitem(self, tmp_path):
        from data_loading.dataset import BoneDataset
        
        processed_dir = self._create_test_data(str(tmp_path))
        
        args = {
            'dataset': {
                'data_dir': str(tmp_path),
                'processed_dir': processed_dir,
                'world_bbox': [100, 100, 100],
                'surface_sample_ratio': 0.5,
                'surface_threshold': 0.05,
                'conditions': {},
            },
            'n_coords_per_subject': 1000,
        }
        
        ds = BoneDataset(args, split='train')
        coords, values, conditions, idx_df = ds[0]
        
        assert coords.shape == (1000, 3)
        assert values.shape == (1000, 1)
        assert coords.dtype == torch.float32
        assert values.dtype == torch.float32
        # Coordinates should be in [-1, 1]
        assert coords.min() >= -1.0
        assert coords.max() <= 1.0


# ===================================================================
# Test 3: SIREN Network
# ===================================================================

class TestSiren:
    """Test SIREN network forward pass."""
    
    def test_sine_layer_forward(self):
        from models.siren import SineLayer
        
        layer = SineLayer(3, 0, 64, is_first=True, omega=30)
        x = torch.randn(100, 3)
        lat = torch.zeros(100, 0)
        
        out, _ = layer((x, lat))
        assert out.shape == (100, 64)
    
    def test_sine_layer_modulated(self):
        from models.siren import SineLayer
        
        layer = SineLayer(3, 32, 64, is_first=True, omega=30)
        x = torch.randn(100, 3)
        lat = torch.randn(100, 32)
        
        out, _ = layer((x, lat))
        assert out.shape == (100, 64)
    
    def test_siren_forward(self):
        from models.siren import Siren
        
        net = Siren(
            in_size=3, lat_size=32, out_size=1,
            hidden_size=64, num_layers=3,
            f_om=30, h_om=30, outermost_linear=True,
            modulated_layers=[1, 3]
        )
        
        coords = torch.randn(100, 3)
        lat = torch.randn(100, 32)
        
        out = net((coords, lat))
        assert out.shape == (100, 1)
    
    def test_siren_no_modulation(self):
        from models.siren import Siren
        
        net = Siren(
            in_size=3, lat_size=32, out_size=1,
            hidden_size=64, num_layers=3,
            f_om=30, h_om=30, outermost_linear=True,
            modulated_layers=[]
        )
        
        coords = torch.randn(50, 3)
        lat = torch.randn(50, 32)
        
        out = net((coords, lat))
        assert out.shape == (50, 1)


# ===================================================================
# Test 4: INR Decoder
# ===================================================================

class TestBoneINRDecoder:
    """Test the bone INR decoder."""
    
    def _make_args(self):
        return {
            'device': 'cpu',
            'inr_decoder': {
                'in_dim': 3,
                'out_dim': [1],
                'hidden_size': 64,
                'num_hidden_layers': 2,
                'latent_dim': [32, 3, 3, 3],
                'omega': [30, 30],
                'modulated_layers': [1],
                'cnn_kernel_size': 0,
                'cond_dims': 0,
                'tf_dim': 6,
            }
        }
    
    def test_decoder_forward(self):
        from models.inr_decoder import BoneINRDecoder
        
        args = self._make_args()
        decoder = BoneINRDecoder(args, 'cpu')
        
        N = 100
        B = 2
        coords = torch.randn(N, 3)
        latents = torch.randn(B, 32, 3, 3, 3)
        tfs = torch.zeros(N, 6)
        idx_df = torch.randint(0, B, (N,))
        
        out = decoder(coords, latents, tfs=tfs, idcs_df=idx_df)
        assert out.shape == (N, 1)
    
    def test_embed2affine(self):
        from models.inr_decoder import embed2affine
        
        params = torch.zeros(5, 6)  # Identity transform
        R, t = embed2affine(params)
        
        assert R.shape == (5, 3, 3)
        assert t.shape == (5, 3)
        # Zero params should give near-identity rotation
        # (due to Rodrigues formula normalization, it depends on limit)


# ===================================================================
# Test 5: Loss Functions
# ===================================================================

class TestBoneLoss:
    """Test the bone loss function."""
    
    def test_sdf_loss_l1(self):
        from utils import BoneLoss
        
        args = {
            'optimizer': {
                'loss_metric': 'l1',
                'latent_reg_weight': 0.0,
                'tf_weight': 0.0,
            }
        }
        
        loss_fn = BoneLoss(args)
        pred = torch.randn(100, 1)
        target = torch.randn(100, 1)
        
        losses = loss_fn(pred, target)
        assert 'total' in losses
        assert 'sdf' in losses
        assert losses['total'].item() > 0
    
    def test_sdf_loss_with_regularization(self):
        from utils import BoneLoss
        
        args = {
            'optimizer': {
                'loss_metric': 'l1',
                'latent_reg_weight': 0.01,
                'tf_weight': 0.01,
            }
        }
        
        loss_fn = BoneLoss(args)
        pred = torch.randn(100, 1)
        target = torch.randn(100, 1)
        latents = torch.randn(4, 32, 3, 3, 3)
        tfs = torch.randn(100, 6)
        
        losses = loss_fn(pred, target, tfs=tfs, latents=latents)
        assert losses['latent_reg'].item() > 0
        assert losses['tf_reg'].item() > 0
        # Total should be sum of all components
        expected = (losses['sdf'].item() +
                    0.01 * losses['latent_reg'].item() +
                    0.01 * losses['tf_reg'].item())
        assert abs(losses['total'].item() - expected) < 1e-4


# ===================================================================
# Test 6: Utilities
# ===================================================================

class TestUtils:
    """Test utility functions."""
    
    def test_generate_world_grid(self):
        from utils import generate_world_grid
        
        args = {
            'atlas_gen': {
                'resolution': 16,
                'spacing': [1.0, 1.0, 1.0],
            }
        }
        
        coords, shape, affine = generate_world_grid(args)
        
        assert coords.shape == (16 * 16 * 16, 3)
        assert shape == [16, 16, 16]
        assert affine.shape == (4, 4)
        # Coordinates should span [-1, 1]
        assert coords.min().item() == pytest.approx(-1.0, abs=1e-5)
        assert coords.max().item() == pytest.approx(1.0, abs=1e-5)
    
    def test_extract_mesh_from_sdf(self):
        from utils import extract_mesh_from_sdf
        
        # Create a synthetic SDF (sphere)
        resolution = 32
        grid = np.linspace(-1, 1, resolution)
        xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
        sdf = np.sqrt(xx**2 + yy**2 + zz**2) - 0.5  # Sphere of radius 0.5
        
        vertices, faces, normals = extract_mesh_from_sdf(sdf, level=0.0)
        
        assert len(vertices) > 0
        assert len(faces) > 0
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
