"""
Unit tests for eigenvalue solvers and projection matrix builders.
"""

import pytest
import torch
import numpy as np
from nspo2.eigensolvers import EigSolver, ProjectionHead


class TestEigSolver:
    """Test EigSolver functionality."""
    
    def test_initialization(self):
        """Test EigSolver initialization."""
        solver = EigSolver(dim=10, use_randomized=False)
        
        assert solver.dim == 10
        assert solver.use_randomized == False
        assert solver.eigenvalues is None
        assert solver.eigenvectors is None
    
    def test_exact_eigendecomposition(self):
        """Test exact eigendecomposition with known matrix."""
        dim = 5
        solver = EigSolver(dim=dim, use_randomized=False)
        
        # Create a diagonal matrix with known eigenvalues
        true_eigenvalues = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        cov = torch.diag(true_eigenvalues)
        
        # Add some rotation to make it non-diagonal
        torch.manual_seed(42)
        Q = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(Q)
        cov = Q @ cov @ Q.T
        
        # Solve for top-3 eigenvalues
        k = 3
        eigenvalues, eigenvectors = solver.solve(cov, k)
        
        assert eigenvalues.shape == (k,)
        assert eigenvectors.shape == (dim, k)
        
        # Check eigenvalues are sorted descending
        assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        
        # Check eigenvalues are approximately correct
        assert torch.allclose(eigenvalues, true_eigenvalues[:k], atol=1e-5)
        
        # Check eigenvectors are orthonormal
        gram = eigenvectors.T @ eigenvectors
        assert torch.allclose(gram, torch.eye(k), atol=1e-5)
    
    def test_randomized_eigendecomposition(self):
        """Test randomized power iteration method."""
        dim = 50
        solver = EigSolver(dim=dim, use_randomized=True, n_power_iterations=3)
        
        # Create low-rank covariance
        torch.manual_seed(42)
        rank = 5
        U = torch.randn(dim, rank)
        U, _ = torch.linalg.qr(U)
        S = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        cov = U @ torch.diag(S) @ U.T
        
        # Add small noise for numerical stability
        cov = cov + torch.eye(dim) * 0.1
        
        # Solve for top eigenvalues
        k = 3
        eigenvalues, eigenvectors = solver.solve(cov, k)
        
        assert eigenvalues.shape == (k,)
        assert eigenvectors.shape == (dim, k)
        
        # Check that top eigenvalues are approximately correct
        assert eigenvalues[0] > 9.0  # Should be close to 10.1
        assert eigenvalues[1] > 7.0  # Should be close to 8.1
        assert eigenvalues[2] > 5.0  # Should be close to 6.1
    
    def test_condition_number(self):
        """Test condition number computation."""
        dim = 5
        solver = EigSolver(dim=dim)
        
        # Create ill-conditioned matrix
        eigenvalues = torch.tensor([100.0, 10.0, 1.0, 0.1, 0.01])
        cov = torch.diag(eigenvalues)
        
        solver.solve(cov, 5)
        cond_number = solver.get_condition_number()
        
        # Condition number should be ratio of largest to smallest
        expected_cond = 100.0 / 0.01
        assert abs(cond_number - expected_cond) / expected_cond < 0.1
    
    def test_invalid_k(self):
        """Test error handling for invalid k."""
        solver = EigSolver(dim=10)
        cov = torch.eye(10)
        
        with pytest.raises(ValueError, match="k must be in"):
            solver.solve(cov, 0)
        
        with pytest.raises(ValueError, match="k must be in"):
            solver.solve(cov, 11)


class TestProjectionHead:
    """Test ProjectionHead functionality."""
    
    def test_initialization(self):
        """Test ProjectionHead initialization."""
        proj = ProjectionHead(dim=10, epsilon=1e-6)
        
        assert proj.dim == 10
        assert proj.epsilon == 1e-6
        assert proj.K is None
        assert proj.rank == 0
    
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthonormalization."""
        dim = 5
        proj = ProjectionHead(dim=dim)
        
        # Create non-orthogonal vectors
        vectors = torch.randn(dim, 3)
        
        # Orthonormalize
        Q = proj._gram_schmidt(vectors)
        
        # Check orthonormality
        gram = Q.T @ Q
        assert torch.allclose(gram, torch.eye(3), atol=1e-5)
        
        # Check normalization
        norms = torch.norm(Q, dim=0)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)
    
    def test_projection_construction(self):
        """Test projection matrix construction."""
        dim = 10
        proj = ProjectionHead(dim=dim)
        
        # Create orthonormal eigenvectors
        k = 3
        torch.manual_seed(42)
        eigenvectors = torch.randn(dim, k)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)
        
        # Build projection
        proj.build_projection(eigenvectors)
        
        assert proj.K is not None
        assert proj.K.shape == (dim, k)
        assert proj.rank == k
        
        # Get projection matrix
        P = proj.get_projection_matrix()
        
        # Check that P is symmetric
        assert torch.allclose(P, P.T, atol=1e-5)
        
        # Check that P is idempotent (P^2 = P)
        P2 = P @ P
        assert torch.allclose(P, P2, atol=1e-5)
    
    def test_projection_removes_components(self):
        """Test that projection removes components in specified directions."""
        dim = 10
        proj = ProjectionHead(dim=dim)
        
        # Create two orthonormal directions to project out
        torch.manual_seed(42)
        directions = torch.randn(dim, 2)
        directions, _ = torch.linalg.qr(directions)
        
        proj.build_projection(directions)
        
        # Create a gradient with components in these directions
        grad = torch.randn(dim)
        
        # Project
        projected = proj.project(grad)
        
        # Check that projected gradient is orthogonal to the directions
        residual = directions.T @ projected
        assert torch.allclose(residual, torch.zeros(2), atol=1e-5)
    
    def test_efficient_vs_direct_projection(self):
        """Test that efficient and direct projection give same results."""
        dim = 20
        
        # Efficient projection
        proj_efficient = ProjectionHead(dim=dim, use_efficient=True)
        
        # Direct projection
        proj_direct = ProjectionHead(dim=dim, use_efficient=False)
        
        # Same eigenvectors
        torch.manual_seed(42)
        eigenvectors = torch.randn(dim, 5)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)
        
        proj_efficient.build_projection(eigenvectors)
        proj_direct.build_projection(eigenvectors)
        
        # Test gradient
        grad = torch.randn(dim)
        
        projected_efficient = proj_efficient.project(grad)
        projected_direct = proj_direct.project(grad)
        
        assert torch.allclose(projected_efficient, projected_direct, atol=1e-5)
    
    def test_verify_projection(self):
        """Test projection verification metrics."""
        dim = 10
        proj = ProjectionHead(dim=dim)
        
        # Build projection
        torch.manual_seed(42)
        eigenvectors = torch.randn(dim, 3)
        proj.build_projection(eigenvectors)
        
        # Test gradient
        grad = torch.randn(dim)
        
        # Verify
        metrics = proj.verify_projection(grad)
        
        assert 'max_residual' in metrics
        assert 'variance_reduction' in metrics
        assert 'norm_ratio' in metrics
        assert 'orthogonal' in metrics
        assert metrics['null_space_dim'] == 3
        assert metrics['orthogonal'] == True  # Should be orthogonal
    
    def test_projection_preserves_orthogonal_components(self):
        """Test that projection preserves components orthogonal to null space."""
        dim = 10
        proj = ProjectionHead(dim=dim)
        
        # Create null space (2 dimensions)
        torch.manual_seed(42)
        null_space = torch.randn(dim, 2)
        null_space, _ = torch.linalg.qr(null_space)
        
        proj.build_projection(null_space)
        
        # Create a vector orthogonal to null space
        orthogonal_vec = torch.randn(dim)
        # Remove components in null space
        for i in range(2):
            orthogonal_vec = orthogonal_vec - (null_space[:, i] @ orthogonal_vec) * null_space[:, i]
        
        # Project (should not change since it's already orthogonal)
        projected = proj.project(orthogonal_vec)
        
        assert torch.allclose(projected, orthogonal_vec, atol=1e-5)


class TestIntegrationEigSolverProjection:
    """Test integration between EigSolver and ProjectionHead."""
    
    def test_full_pipeline(self):
        """Test complete eigendecomposition and projection pipeline."""
        dim = 20
        
        # Create covariance with known structure
        torch.manual_seed(42)
        # High variance directions
        high_var_dirs = torch.randn(dim, 3)
        high_var_dirs, _ = torch.linalg.qr(high_var_dirs)
        
        # Create covariance
        cov = torch.zeros((dim, dim))
        for i, var in enumerate([10.0, 8.0, 6.0]):
            cov += var * torch.outer(high_var_dirs[:, i], high_var_dirs[:, i])
        cov += torch.eye(dim) * 0.5  # Add small variance in all directions
        
        # Solve eigendecomposition
        solver = EigSolver(dim=dim)
        k = 3
        eigenvalues, eigenvectors = solver.solve(cov, k)
        
        # Build projection
        proj = ProjectionHead(dim=dim)
        proj.build_projection(eigenvectors)
        
        # Test gradient with high variance in top directions
        grad = 5.0 * high_var_dirs[:, 0] + 3.0 * high_var_dirs[:, 1] + torch.randn(dim) * 0.1
        
        # Project
        projected = proj.project(grad)
        
        # Verify variance reduction
        orig_var = torch.var(grad).item()
        proj_var = torch.var(projected).item()
        
        assert proj_var < orig_var  # Variance should be reduced
        
        # Verify that high-variance components are removed
        for i in range(3):
            component = high_var_dirs[:, i] @ projected
            assert abs(component) < 0.5  # Should be mostly removed


class TestIntegrationWithNSPO2:
    """Test integration with main NSPO2 class."""
    
    def test_nspo2_projection(self):
        """Test that NSPO2 properly performs projection."""
        from nspo2 import NSPO2
        from nspo2.config import get_default_config
        
        config = get_default_config(dim=10)
        config['update_freq'] = 5
        config['projection_rank'] = 3
        nspo2 = NSPO2(config)
        
        # Process enough gradients to trigger eigendecomposition
        torch.manual_seed(42)
        for i in range(10):
            grad = torch.randn(10)
            # Add structure: high variance in first dimension
            grad[0] += 5.0
            
            projected = nspo2.hook(grad)
            
            if i >= config['update_freq']:
                # After update_freq steps, projection should be active
                assert nspo2.eigenvectors is not None
                assert nspo2.eigenvalues is not None
                
                # Check variance reduction
                if len(nspo2.history['variance_reduction']) > 0:
                    recent_reduction = nspo2.history['variance_reduction'][-1]
                    if i > config['update_freq']:  # Give it time to converge
                        assert recent_reduction > 0  # Should have some reduction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])