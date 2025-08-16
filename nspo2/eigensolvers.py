"""
Eigenvalue solvers and projection matrix builders for NSPO2.
"""

import torch
from typing import Tuple, Optional
import warnings
import numpy as np


class EigSolver:
    """Eigenvalue solver for covariance matrices.
    
    Supports both exact eigendecomposition for small dimensions and
    randomized power iteration for large dimensions.
    """
    
    def __init__(
        self,
        dim: int,
        use_randomized: bool = False,
        n_power_iterations: int = 2,
        oversampling: int = 10,
        randomized_threshold: int = 100,
        epsilon: float = 1e-6,
        seed: Optional[int] = None
    ):
        """Initialize the eigenvalue solver.
        
        Args:
            dim: Dimension of the matrix
            use_randomized: Force use of randomized method
            n_power_iterations: Number of power iterations for randomized method
            oversampling: Extra dimensions for randomized method
            randomized_threshold: Dimension threshold for automatic randomized method
            epsilon: Small constant for numerical stability
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.use_randomized = use_randomized or dim > randomized_threshold
        self.n_power_iterations = n_power_iterations
        self.oversampling = oversampling
        self.epsilon = epsilon
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.eigenvalues = None
        self.eigenvectors = None
        self.last_k = None
    
    def solve(self, cov: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k eigenvalues and eigenvectors.
        
        Args:
            cov: Covariance matrix (symmetric)
            k: Number of top eigenvalues/vectors to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted in descending order
        """
        if k <= 0 or k > self.dim:
            raise ValueError(f"k must be in (0, {self.dim}], got {k}")
        
        # Ensure matrix is symmetric
        cov = (cov + cov.T) / 2
        
        if self.use_randomized and k < self.dim // 2:
            eigenvalues, eigenvectors = self._randomized_eig(cov, k)
        else:
            eigenvalues, eigenvectors = self._exact_eig(cov, k)
        
        # Store results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.last_k = k
        
        return eigenvalues, eigenvectors
    
    def _exact_eig(self, cov: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact eigendecomposition.
        
        Args:
            cov: Covariance matrix
            k: Number of top eigenvalues/vectors
            
        Returns:
            Top-k eigenvalues and eigenvectors
        """
        try:
            # Full eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except RuntimeError as e:
            warnings.warn(f"Eigendecomposition failed, using SVD: {e}")
            # Fallback to SVD
            U, S, _ = torch.linalg.svd(cov)
            eigenvalues = S
            eigenvectors = U
        
        # Sort in descending order and select top-k
        idx = torch.argsort(eigenvalues, descending=True)[:k]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _randomized_eig(self, cov: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute approximate eigendecomposition using randomized power iteration.
        
        Based on Halko, Martinsson, and Tropp (2011) randomized SVD algorithm.
        
        Args:
            cov: Covariance matrix
            k: Number of top eigenvalues/vectors
            
        Returns:
            Approximate top-k eigenvalues and eigenvectors
        """
        n = cov.shape[0]
        l = min(k + self.oversampling, n)  # Oversampled rank
        
        # Step 1: Generate random test matrix
        omega = torch.randn(n, l)
        
        # Step 2: Power iteration to improve approximation
        Y = cov @ omega
        for _ in range(self.n_power_iterations):
            Y = cov @ (cov @ Y)
        
        # Step 3: Orthonormalize Y
        Q, _ = torch.linalg.qr(Y, mode='reduced')
        
        # Step 4: Form the small matrix B = Q^T @ A @ Q
        B = Q.T @ cov @ Q
        
        # Step 5: Compute eigendecomposition of small matrix
        s, Uhat = torch.linalg.eigh(B)
        
        # Step 6: Recover eigenvectors of original matrix
        U = Q @ Uhat
        
        # Sort in descending order and select top-k
        idx = torch.argsort(s, descending=True)[:k]
        eigenvalues = s[idx]
        eigenvectors = U[:, idx]
        
        # Ensure eigenvectors are normalized
        eigenvectors = eigenvectors / torch.norm(eigenvectors, dim=0, keepdim=True)
        
        return eigenvalues, eigenvectors
    
    def update(self, cov: torch.Tensor, k: int) -> None:
        """Update internal state with new eigendecomposition.
        
        Args:
            cov: New covariance matrix
            k: Number of top eigenvalues/vectors to compute
        """
        self.solve(cov, k)
    
    def get_condition_number(self) -> float:
        """Get condition number of the last decomposition.
        
        Returns:
            Condition number (ratio of largest to smallest eigenvalue)
        """
        if self.eigenvalues is None:
            return 1.0
        
        # Filter very small eigenvalues
        valid_eigenvalues = self.eigenvalues[self.eigenvalues > self.epsilon]
        
        if len(valid_eigenvalues) == 0:
            return float('inf')
        
        return (valid_eigenvalues[0] / valid_eigenvalues[-1]).item()


class ProjectionHead:
    """Projection matrix builder and applicator.
    
    Constructs and applies the projection matrix P = I - K(K^T K)^(-1)K^T
    where K is the matrix of eigenvectors to project out.
    """
    
    def __init__(self, dim: int, epsilon: float = 1e-6, use_efficient: bool = True):
        """Initialize the projection head.
        
        Args:
            dim: Dimension of the space
            epsilon: Small constant for numerical stability
            use_efficient: Use efficient matrix-vector product (avoid materializing P)
        """
        self.dim = dim
        self.epsilon = epsilon
        self.use_efficient = use_efficient
        
        self.K = None  # Eigenvectors to project out (orthonormalized)
        self.K_inv_gram = None  # (K^T K)^(-1) for efficient computation
        self.projection_matrix = None  # Full projection matrix (if not efficient)
        self.rank = 0
    
    def build_projection(self, eigenvectors: torch.Tensor, regularize: bool = True) -> None:
        """Construct projection matrix from eigenvectors.
        
        Args:
            eigenvectors: Matrix of eigenvectors to project out (columns)
            regularize: Apply Gram-Schmidt orthonormalization
        """
        if eigenvectors.dim() == 1:
            eigenvectors = eigenvectors.unsqueeze(1)
        
        if eigenvectors.shape[0] != self.dim:
            raise ValueError(f"Eigenvector dimension {eigenvectors.shape[0]} doesn't match {self.dim}")
        
        self.rank = eigenvectors.shape[1]
        
        # Orthonormalize eigenvectors using Gram-Schmidt if requested
        if regularize:
            self.K = self._gram_schmidt(eigenvectors)
        else:
            self.K = eigenvectors
        
        # Compute (K^T K)^(-1) for efficient projection
        gram_matrix = self.K.T @ self.K
        
        # Add regularization for numerical stability
        gram_matrix = gram_matrix + torch.eye(self.rank) * self.epsilon
        
        try:
            # Try Cholesky decomposition (more stable for positive definite)
            L = torch.linalg.cholesky(gram_matrix)
            self.K_inv_gram = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback to standard inverse
            self.K_inv_gram = torch.linalg.inv(gram_matrix)
        
        # Optionally materialize full projection matrix
        if not self.use_efficient and self.dim <= 1000:
            # P = I - K @ (K^T K)^(-1) @ K^T
            self.projection_matrix = torch.eye(self.dim) - self.K @ self.K_inv_gram @ self.K.T
    
    def _gram_schmidt(self, vectors: torch.Tensor) -> torch.Tensor:
        """Orthonormalize vectors using modified Gram-Schmidt.
        
        Args:
            vectors: Matrix of vectors (columns)
            
        Returns:
            Orthonormalized vectors
        """
        n, k = vectors.shape
        Q = torch.zeros_like(vectors)
        
        for i in range(k):
            # Start with the i-th vector
            q = vectors[:, i].clone()
            
            # Subtract projections onto previous vectors
            for j in range(i):
                q = q - (Q[:, j] @ q) * Q[:, j]
            
            # Normalize
            q_norm = torch.norm(q)
            if q_norm > self.epsilon:
                Q[:, i] = q / q_norm
            else:
                # Vector is linearly dependent, use random direction
                Q[:, i] = torch.randn(n)
                Q[:, i] = Q[:, i] / torch.norm(Q[:, i])
        
        return Q
    
    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply projection to gradient.
        
        Computes Pg = g - K(K^T K)^(-1)K^T g
        
        Args:
            grad: Gradient vector to project
            
        Returns:
            Projected gradient
        """
        if self.K is None:
            # No projection matrix built yet, return original
            return grad
        
        original_shape = grad.shape
        grad_flat = grad.flatten()
        
        if self.use_efficient:
            # Efficient computation: Pg = g - K @ (K^T K)^(-1) @ K^T @ g
            # Step 1: Compute K^T @ g
            coeffs = self.K.T @ grad_flat
            
            # Step 2: Compute (K^T K)^(-1) @ coeffs
            inv_coeffs = self.K_inv_gram @ coeffs
            
            # Step 3: Compute K @ inv_coeffs
            projection_component = self.K @ inv_coeffs
            
            # Step 4: Compute g - projection_component
            projected = grad_flat - projection_component
        else:
            if self.projection_matrix is None:
                # Build projection matrix if not already done
                self.build_projection(self.K, regularize=False)
            
            # Direct matrix multiplication
            projected = self.projection_matrix @ grad_flat
        
        return projected.reshape(original_shape)
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Return current projection matrix.
        
        Returns:
            Full projection matrix P
        """
        if self.projection_matrix is not None:
            return self.projection_matrix
        
        if self.K is None:
            return torch.eye(self.dim)
        
        # Materialize projection matrix
        P = torch.eye(self.dim) - self.K @ self.K_inv_gram @ self.K.T
        return P
    
    def get_null_space_dimension(self) -> int:
        """Get dimension of the null space being projected out.
        
        Returns:
            Rank of the projection (dimension of null space)
        """
        return self.rank
    
    def verify_projection(self, grad: torch.Tensor) -> dict:
        """Verify projection properties.
        
        Args:
            grad: Test gradient
            
        Returns:
            Dictionary with verification metrics
        """
        if self.K is None:
            return {"status": "No projection matrix built"}
        
        grad_flat = grad.flatten()
        projected = self.project(grad)
        projected_flat = projected.flatten()
        
        # Check that projected gradient is orthogonal to null space
        residual = self.K.T @ projected_flat
        max_residual = torch.max(torch.abs(residual)).item()
        
        # Compute variance reduction
        orig_var = torch.var(grad_flat).item()
        proj_var = torch.var(projected_flat).item()
        var_reduction = (orig_var - proj_var) / (orig_var + self.epsilon)
        
        # Compute projection magnitude
        proj_norm = torch.norm(projected_flat).item()
        orig_norm = torch.norm(grad_flat).item()
        norm_ratio = proj_norm / (orig_norm + self.epsilon)
        
        return {
            "max_residual": max_residual,
            "variance_reduction": var_reduction,
            "norm_ratio": norm_ratio,
            "null_space_dim": self.rank,
            "orthogonal": max_residual < 1e-5
        }