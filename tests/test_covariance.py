"""
Unit tests for covariance estimation module.
"""

import pytest
import torch
import numpy as np
from nspo2.covariance import CovEstimator


class TestCovEstimator:
    """Test CovEstimator functionality."""
    
    def test_initialization(self):
        """Test CovEstimator initialization."""
        estimator = CovEstimator(dim=10, decay=0.9, shrinkage=0.1)
        
        assert estimator.dim == 10
        assert estimator.decay == 0.9
        assert estimator.shrinkage == 0.1
        assert estimator.n_updates == 0
        assert estimator.mean.shape == (10,)
    
    def test_invalid_parameters(self):
        """Test invalid parameter validation."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            CovEstimator(dim=0)
        
        with pytest.raises(ValueError, match="Decay must be in"):
            CovEstimator(dim=10, decay=1.5)
        
        with pytest.raises(ValueError, match="Shrinkage must be in"):
            CovEstimator(dim=10, shrinkage=1.0)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            CovEstimator(dim=10, epsilon=0)
    
    def test_full_covariance_update(self):
        """Test full covariance matrix updates."""
        dim = 5
        estimator = CovEstimator(dim=dim, decay=0.9, shrinkage=0.0, epsilon=1e-8)
        
        # Create gradients with known covariance
        torch.manual_seed(42)
        true_mean = torch.zeros(dim)
        true_cov = torch.eye(dim) * 2.0  # Diagonal covariance
        
        # Generate samples
        n_samples = 1000
        for _ in range(n_samples):
            grad = torch.randn(dim) * torch.sqrt(torch.diag(true_cov))
            estimator.update(grad)
        
        # Check mean estimation (with EMA, mean won't be exactly zero)
        estimated_mean = estimator.mean
        assert torch.allclose(estimated_mean, true_mean, atol=0.5)
        
        # Check covariance estimation (diagonal should be approximately 2.0)
        estimated_cov = estimator.get_covariance()
        diag_estimated = torch.diag(estimated_cov)
        diag_true = torch.diag(true_cov)
        # Check that diagonal elements are in the right range
        assert torch.all(diag_estimated > 0.5)
        assert torch.all(diag_estimated < 3.5)
    
    def test_diagonal_covariance(self):
        """Test diagonal covariance estimation."""
        dim = 10
        estimator = CovEstimator(dim=dim, estimation_type='diagonal')
        
        # Update with random gradients
        for _ in range(100):
            grad = torch.randn(dim)
            estimator.update(grad)
        
        cov = estimator.get_covariance()
        
        # Check that covariance is diagonal
        assert torch.allclose(cov, torch.diag(torch.diag(cov)))
    
    def test_low_rank_covariance(self):
        """Test low-rank covariance approximation."""
        dim = 20
        rank = 5
        estimator = CovEstimator(dim=dim, estimation_type='low_rank', rank=rank)
        
        # Update with gradients that have low-rank structure
        torch.manual_seed(42)
        U_true = torch.randn(dim, rank)
        U_true, _ = torch.linalg.qr(U_true)  # Orthonormalize
        
        for _ in range(100):
            coeffs = torch.randn(rank)
            grad = U_true @ coeffs
            estimator.update(grad)
        
        cov = estimator.get_covariance()
        
        # Check that covariance has approximately correct rank
        eigenvalues, _ = torch.linalg.eigh(cov)
        significant_eigenvalues = eigenvalues[eigenvalues > 1e-3]
        assert len(significant_eigenvalues) <= rank + 2  # Allow some tolerance
    
    def test_shrinkage_regularization(self):
        """Test shrinkage regularization effect."""
        dim = 10
        
        # Without shrinkage
        estimator_no_shrink = CovEstimator(dim=dim, shrinkage=0.0)
        
        # With shrinkage
        estimator_shrink = CovEstimator(dim=dim, shrinkage=0.5)
        
        # Update with same gradients
        torch.manual_seed(42)
        for _ in range(50):
            grad = torch.randn(dim)
            estimator_no_shrink.update(grad)
            estimator_shrink.update(grad)
        
        cov_no_shrink = estimator_no_shrink.get_covariance()
        cov_shrink = estimator_shrink.get_covariance()
        
        # Shrinkage should reduce off-diagonal elements
        off_diag_no_shrink = cov_no_shrink - torch.diag(torch.diag(cov_no_shrink))
        off_diag_shrink = cov_shrink - torch.diag(torch.diag(cov_shrink))
        
        assert torch.norm(off_diag_shrink) < torch.norm(off_diag_no_shrink)
    
    def test_eigendecomposition(self):
        """Test eigendecomposition computation."""
        dim = 5
        estimator = CovEstimator(dim=dim)
        
        # Create covariance with known eigenstructure
        torch.manual_seed(42)
        eigenvalues_true = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5])
        Q = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(Q)
        
        # Generate gradients from this distribution
        for _ in range(500):
            z = torch.randn(dim) * torch.sqrt(eigenvalues_true)
            grad = Q @ z
            estimator.update(grad)
        
        eigenvalues, eigenvectors = estimator.get_eigendecomposition()
        
        # Check eigenvalues are sorted in descending order
        assert torch.all(eigenvalues[:-1] >= eigenvalues[1:])
        
        # Check eigenvalues are approximately correct (order of magnitude)
        assert eigenvalues[0] > 3.0  # Largest eigenvalue
        assert eigenvalues[-1] < 2.0  # Smallest eigenvalue
    
    def test_condition_number(self):
        """Test condition number computation."""
        dim = 5
        estimator = CovEstimator(dim=dim, epsilon=1e-6)
        
        # Initially should be 1 (no updates)
        assert estimator.get_condition_number() == 1.0
        
        # Create ill-conditioned covariance
        for i in range(100):
            grad = torch.randn(dim)
            grad[0] *= 10  # Make first dimension have much higher variance
            estimator.update(grad)
        
        cond_number = estimator.get_condition_number()
        assert cond_number > 1.0
        assert cond_number < float('inf')
    
    def test_positive_definite_check(self):
        """Test positive definite check."""
        dim = 5
        estimator = CovEstimator(dim=dim, epsilon=1e-6)
        
        # Update with gradients
        for _ in range(10):
            estimator.update(torch.randn(dim))
        
        # Covariance should be positive definite
        assert estimator.is_positive_definite()
    
    def test_trace_computation(self):
        """Test trace computation."""
        dim = 5
        estimator = CovEstimator(dim=dim, epsilon=1e-6)
        
        # Update with unit variance gradients
        for _ in range(100):
            estimator.update(torch.randn(dim))
        
        trace = estimator.get_trace()
        
        # Trace should be approximately dim (unit variance in each dimension)
        assert abs(trace - dim) < 2.0
    
    def test_reset(self):
        """Test reset functionality."""
        dim = 5
        estimator = CovEstimator(dim=dim)
        
        # Update estimator
        for _ in range(10):
            estimator.update(torch.randn(dim))
        
        assert estimator.n_updates == 10
        assert torch.any(estimator.mean != 0)
        
        # Reset
        estimator.reset()
        
        assert estimator.n_updates == 0
        assert torch.all(estimator.mean == 0)
        assert estimator.effective_samples == 0.0
    
    def test_gradient_dimension_mismatch(self):
        """Test error on gradient dimension mismatch."""
        estimator = CovEstimator(dim=10)
        
        with pytest.raises(ValueError, match="Gradient dimension"):
            estimator.update(torch.randn(5))
    
    def test_bias_correction(self):
        """Test bias correction in covariance estimation."""
        dim = 5
        estimator = CovEstimator(dim=dim, decay=0.95, epsilon=1e-8)
        
        # Small number of updates
        for _ in range(3):
            estimator.update(torch.randn(dim))
        
        # Get covariance (should apply bias correction)
        cov = estimator.get_covariance()
        
        # Covariance should be finite and reasonable
        assert torch.all(torch.isfinite(cov))
        # Check only diagonal elements are non-negative (off-diagonal can be negative)
        assert torch.all(torch.diag(cov) >= 0)
    
    def test_streaming_update_efficiency(self):
        """Test that streaming updates are memory efficient."""
        dim = 100
        estimator = CovEstimator(dim=dim, estimation_type='full')
        
        # Perform many updates
        for _ in range(1000):
            grad = torch.randn(dim)
            estimator.update(grad)
        
        # Check that internal state size is constant
        if estimator.estimation_type == 'full':
            assert estimator.cov.shape == (dim, dim)
        elif estimator.estimation_type == 'diagonal':
            assert estimator.cov.shape == (dim,)
        
        assert estimator.mean.shape == (dim,)


class TestIntegrationWithNSPO2:
    """Test integration of CovEstimator with NSPO2."""
    
    def test_nspo2_covariance_update(self):
        """Test that NSPO2 properly updates covariance."""
        from nspo2 import NSPO2
        from nspo2.config import get_default_config
        
        config = get_default_config(dim=10)
        nspo2 = NSPO2(config)
        
        # Process some gradients
        for _ in range(10):
            grad = torch.randn(10)
            nspo2.hook(grad)
        
        # Check that covariance has been updated
        assert nspo2.cov_estimator.n_updates == 10
        
        # Get covariance and check it's valid
        cov = nspo2.cov_estimator.get_covariance()
        assert cov.shape == (10, 10)
        assert nspo2.cov_estimator.is_positive_definite()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])