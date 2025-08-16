"""
Covariance estimation module for NSPO2.
"""

import torch
from typing import Optional, Literal
import warnings


class CovEstimator:
    """Covariance estimator with streaming updates and shrinkage regularization.
    
    This class maintains a running estimate of the covariance matrix of gradients
    using exponential moving average (EMA) or exponentially weighted moving average (EWMA).
    It supports shrinkage regularization for numerical stability.
    """
    
    def __init__(
        self,
        dim: int,
        decay: float = 0.99,
        shrinkage: float = 0.01,
        epsilon: float = 1e-6,
        estimation_type: Literal['full', 'diagonal', 'low_rank'] = 'full',
        rank: Optional[int] = None
    ):
        """Initialize the covariance estimator.
        
        Args:
            dim: Dimension of the gradient vectors
            decay: Decay factor for EMA updates (0 < decay < 1)
            shrinkage: Shrinkage parameter for regularization (0 <= shrinkage < 1)
            epsilon: Small constant for numerical stability
            estimation_type: Type of covariance estimation ('full', 'diagonal', 'low_rank')
            rank: Rank for low-rank approximation (only used if estimation_type='low_rank')
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if not 0 < decay < 1:
            raise ValueError(f"Decay must be in (0, 1), got {decay}")
        if not 0 <= shrinkage < 1:
            raise ValueError(f"Shrinkage must be in [0, 1), got {shrinkage}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        self.dim = dim
        self.decay = decay
        self.shrinkage = shrinkage
        self.epsilon = epsilon
        self.estimation_type = estimation_type
        
        # Initialize mean and covariance
        self.mean = torch.zeros(dim)
        
        if estimation_type == 'full':
            self.cov = torch.zeros((dim, dim))
        elif estimation_type == 'diagonal':
            self.cov = torch.zeros(dim)  # Only store diagonal
        elif estimation_type == 'low_rank':
            if rank is None:
                rank = min(dim // 4, 32)  # Default rank
            self.rank = min(rank, dim)
            # Low-rank approximation: cov ≈ U @ S @ U.T
            self.U = torch.zeros((dim, self.rank))  # Left singular vectors
            self.S = torch.zeros(self.rank)  # Singular values
        else:
            raise ValueError(f"Unknown estimation_type: {estimation_type}")
        
        self.n_updates = 0
        self.effective_samples = 0.0  # Effective sample size with decay
    
    def update(self, grad: torch.Tensor) -> None:
        """Update covariance estimate with new gradient.
        
        Args:
            grad: Gradient vector (1D tensor)
        """
        if grad.dim() != 1:
            grad = grad.flatten()
        
        if grad.shape[0] != self.dim:
            raise ValueError(f"Gradient dimension {grad.shape[0]} doesn't match expected {self.dim}")
        
        # Detach and ensure no gradient tracking
        grad = grad.detach()
        
        # Update effective sample size
        self.effective_samples = self.decay * self.effective_samples + 1.0
        
        # Update mean using EMA
        if self.n_updates == 0:
            self.mean = grad.clone()
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * grad
        
        # Center the gradient
        centered_grad = grad - self.mean
        
        # Update covariance based on estimation type
        if self.estimation_type == 'full':
            self._update_full_covariance(centered_grad)
        elif self.estimation_type == 'diagonal':
            self._update_diagonal_covariance(centered_grad)
        elif self.estimation_type == 'low_rank':
            self._update_low_rank_covariance(centered_grad)
        
        self.n_updates += 1
    
    def _update_full_covariance(self, centered_grad: torch.Tensor) -> None:
        """Update full covariance matrix."""
        outer_product = torch.outer(centered_grad, centered_grad)
        
        if self.n_updates == 0:
            self.cov = outer_product
        else:
            self.cov = self.decay * self.cov + (1 - self.decay) * outer_product
    
    def _update_diagonal_covariance(self, centered_grad: torch.Tensor) -> None:
        """Update diagonal covariance (variance) only."""
        variance = centered_grad ** 2
        
        if self.n_updates == 0:
            self.cov = variance
        else:
            self.cov = self.decay * self.cov + (1 - self.decay) * variance
    
    def _update_low_rank_covariance(self, centered_grad: torch.Tensor) -> None:
        """Update low-rank approximation of covariance."""
        if self.n_updates == 0:
            # Initialize with the first gradient
            self.U[:, 0] = centered_grad / (torch.norm(centered_grad) + self.epsilon)
            self.S[0] = torch.norm(centered_grad) ** 2
        else:
            # Incremental SVD update (simplified version)
            # Project gradient onto current subspace
            coeffs = self.U.T @ centered_grad
            residual = centered_grad - self.U @ coeffs
            residual_norm = torch.norm(residual)
            
            # Update singular values
            self.S = self.decay * self.S + (1 - self.decay) * coeffs ** 2
            
            # Add new direction if residual is significant
            if residual_norm > self.epsilon and self.n_updates < self.rank:
                idx = min(self.n_updates, self.rank - 1)
                self.U[:, idx] = residual / residual_norm
                self.S[idx] = (1 - self.decay) * residual_norm ** 2
    
    def get_covariance(self) -> torch.Tensor:
        """Return current covariance matrix estimate with shrinkage regularization.
        
        Returns:
            Regularized covariance matrix
        """
        if self.n_updates == 0:
            # Return identity matrix if no updates yet
            return torch.eye(self.dim) * self.epsilon
        
        # Apply bias correction for finite samples
        if self.effective_samples > 1:
            bias_correction = self.effective_samples / (self.effective_samples - 1)
        else:
            bias_correction = 1.0
        
        if self.estimation_type == 'full':
            # Apply shrinkage: Σ_λ = (1-λ)Σ̂ + λ·diag(Σ̂)
            cov_corrected = self.cov * bias_correction
            
            if self.shrinkage > 0:
                diag_cov = torch.diag(torch.diag(cov_corrected))
                cov_shrunk = (1 - self.shrinkage) * cov_corrected + self.shrinkage * diag_cov
            else:
                cov_shrunk = cov_corrected
            
            # Add epsilon to diagonal for numerical stability
            cov_shrunk = cov_shrunk + torch.eye(self.dim) * self.epsilon
            
            return cov_shrunk
        
        elif self.estimation_type == 'diagonal':
            # Return diagonal matrix
            var_corrected = self.cov * bias_correction
            
            if self.shrinkage > 0:
                var_mean = torch.mean(var_corrected)
                var_shrunk = (1 - self.shrinkage) * var_corrected + self.shrinkage * var_mean
            else:
                var_shrunk = var_corrected
            
            return torch.diag(var_shrunk + self.epsilon)
        
        elif self.estimation_type == 'low_rank':
            # Reconstruct covariance from low-rank factors
            S_corrected = self.S * bias_correction
            
            # Apply shrinkage to singular values
            if self.shrinkage > 0:
                S_mean = torch.mean(S_corrected)
                S_shrunk = (1 - self.shrinkage) * S_corrected + self.shrinkage * S_mean
            else:
                S_shrunk = S_corrected
            
            # Reconstruct: Σ = U @ diag(S) @ U.T + εI
            active_rank = min(self.n_updates, self.rank)
            U_active = self.U[:, :active_rank]
            S_active = S_shrunk[:active_rank]
            
            cov = U_active @ torch.diag(S_active) @ U_active.T
            cov = cov + torch.eye(self.dim) * self.epsilon
            
            return cov
    
    def get_eigendecomposition(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get eigendecomposition of the covariance matrix.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in descending order
        """
        cov = self.get_covariance()
        
        # Ensure matrix is symmetric (numerical errors can break symmetry)
        cov = (cov + cov.T) / 2
        
        # Compute eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except RuntimeError as e:
            warnings.warn(f"Eigendecomposition failed, returning approximate values: {e}")
            # Fallback to SVD which is more stable
            U, S, _ = torch.linalg.svd(cov)
            eigenvalues = S
            eigenvectors = U
        
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def reset(self) -> None:
        """Reset the estimator state."""
        self.mean.zero_()
        
        if self.estimation_type == 'full':
            self.cov.zero_()
        elif self.estimation_type == 'diagonal':
            self.cov.zero_()
        elif self.estimation_type == 'low_rank':
            self.U.zero_()
            self.S.zero_()
        
        self.n_updates = 0
        self.effective_samples = 0.0
    
    def get_condition_number(self) -> float:
        """Get condition number of the covariance matrix.
        
        Returns:
            Condition number (ratio of largest to smallest eigenvalue)
        """
        if self.n_updates == 0:
            return 1.0
        
        eigenvalues, _ = self.get_eigendecomposition()
        
        # Filter out very small eigenvalues
        eigenvalues = eigenvalues[eigenvalues > self.epsilon]
        
        if len(eigenvalues) == 0:
            return float('inf')
        
        return (eigenvalues[0] / eigenvalues[-1]).item()
    
    def get_trace(self) -> float:
        """Get trace (sum of diagonal elements) of covariance matrix.
        
        Returns:
            Trace of the covariance matrix
        """
        if self.n_updates == 0:
            return self.dim * self.epsilon
        
        if self.estimation_type == 'diagonal':
            return (torch.sum(self.cov) + self.dim * self.epsilon).item()
        else:
            cov = self.get_covariance()
            return torch.trace(cov).item()
    
    def is_positive_definite(self) -> bool:
        """Check if the covariance matrix is positive definite.
        
        Returns:
            True if positive definite, False otherwise
        """
        eigenvalues, _ = self.get_eigendecomposition()
        return torch.all(eigenvalues > 0).item()