"""
Strategy selection module for NSPO2.

Implements different projection strategies for eigenvector selection/masking.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Literal


class Strategy(ABC):
    """Abstract base class for projection strategies."""
    
    @abstractmethod
    def apply(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        rank: int,
        reference_direction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply strategy to select/mask eigenvectors.
        
        Args:
            eigenvectors: Matrix of eigenvectors (columns)
            eigenvalues: Vector of eigenvalues (sorted descending)
            rank: Number of eigenvectors to select
            reference_direction: Optional reference direction to preserve
            
        Returns:
            Selected eigenvectors based on strategy
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class NoiseAwareStrategy(Strategy):
    """Noise-aware strategy: removes top-r eigenvectors (highest variance directions).
    
    This strategy maximizes variance reduction by projecting out the directions
    with the highest variance (top eigenvalues).
    """
    
    def apply(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        rank: int,
        reference_direction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Select top-r eigenvectors to project out.
        
        Args:
            eigenvectors: Matrix of eigenvectors (columns)
            eigenvalues: Vector of eigenvalues (sorted descending)
            rank: Number of eigenvectors to select
            reference_direction: Not used in this strategy
            
        Returns:
            Top-r eigenvectors (highest variance directions)
        """
        if rank <= 0:
            return torch.empty(eigenvectors.shape[0], 0)
        
        if rank >= eigenvectors.shape[1]:
            return eigenvectors
        
        # Return top-r eigenvectors (already sorted by eigenvalue)
        return eigenvectors[:, :rank]
    
    def get_name(self) -> str:
        return "noise"


class KnowledgePreservingStrategy(Strategy):
    """Knowledge-preserving strategy: preserves important directions while removing noise.
    
    This strategy tries to preserve knowledge-bearing directions (reference direction)
    while still reducing variance. It selects eigenvectors that are most orthogonal
    to the reference direction.
    """
    
    def apply(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        rank: int,
        reference_direction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Select eigenvectors most orthogonal to reference direction.
        
        Args:
            eigenvectors: Matrix of eigenvectors (columns)
            eigenvalues: Vector of eigenvalues (sorted descending)
            rank: Number of eigenvectors to select
            reference_direction: Direction to preserve (e.g., average gradient)
            
        Returns:
            Selected eigenvectors that minimize interference with reference
        """
        if rank <= 0:
            return torch.empty(eigenvectors.shape[0], 0)
        
        if rank >= eigenvectors.shape[1]:
            return eigenvectors
        
        if reference_direction is None:
            # No reference direction, fall back to noise-aware strategy
            return eigenvectors[:, :rank]
        
        # Normalize reference direction
        ref_norm = torch.norm(reference_direction)
        if ref_norm < 1e-8:
            # Reference direction is nearly zero, fall back to noise-aware
            return eigenvectors[:, :rank]
        
        ref_normalized = reference_direction / ref_norm
        
        # Compute alignment (absolute cosine similarity) with each eigenvector
        alignments = torch.abs(eigenvectors.T @ ref_normalized)
        
        # Create score combining eigenvalue importance and orthogonality
        # Higher eigenvalue + lower alignment = better candidate for removal
        # Score = eigenvalue * (1 - alignment)
        scores = eigenvalues * (1 - alignments)
        
        # Select top-r eigenvectors by score
        _, indices = torch.sort(scores, descending=True)
        selected_indices = indices[:rank]
        
        # Sort selected indices to maintain order
        selected_indices, _ = torch.sort(selected_indices)
        
        return eigenvectors[:, selected_indices]
    
    def get_name(self) -> str:
        return "keep"


class HybridStrategy(Strategy):
    """Hybrid strategy: balances between noise removal and knowledge preservation.
    
    This strategy splits the rank between noise-aware and knowledge-preserving
    approaches, removing some high-variance directions while preserving others
    that might contain important information.
    """
    
    def __init__(self, noise_fraction: float = 0.5):
        """Initialize hybrid strategy.
        
        Args:
            noise_fraction: Fraction of rank to use for noise removal (0 to 1)
        """
        self.noise_fraction = max(0.0, min(1.0, noise_fraction))
    
    def apply(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        rank: int,
        reference_direction: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply hybrid selection of eigenvectors.
        
        Args:
            eigenvectors: Matrix of eigenvectors (columns)
            eigenvalues: Vector of eigenvalues (sorted descending)
            rank: Number of eigenvectors to select
            reference_direction: Optional direction to partially preserve
            
        Returns:
            Selected eigenvectors using hybrid approach
        """
        if rank <= 0:
            return torch.empty(eigenvectors.shape[0], 0)
        
        if rank >= eigenvectors.shape[1]:
            return eigenvectors
        
        # Split rank between noise removal and knowledge preservation
        noise_rank = int(rank * self.noise_fraction)
        preserve_rank = rank - noise_rank
        
        if noise_rank == rank:
            # All for noise removal
            return eigenvectors[:, :rank]
        
        if preserve_rank == rank:
            # All for knowledge preservation
            strategy = KnowledgePreservingStrategy()
            return strategy.apply(eigenvectors, eigenvalues, rank, reference_direction)
        
        # Combine both approaches
        selected_indices = []
        
        # Add top eigenvectors for noise removal
        selected_indices.extend(range(noise_rank))
        
        # For knowledge preservation, select from remaining eigenvectors
        if reference_direction is not None and preserve_rank > 0:
            ref_norm = torch.norm(reference_direction)
            if ref_norm > 1e-8:
                ref_normalized = reference_direction / ref_norm
                
                # Compute alignment with remaining eigenvectors
                remaining_eigenvectors = eigenvectors[:, noise_rank:]
                remaining_eigenvalues = eigenvalues[noise_rank:]
                
                alignments = torch.abs(remaining_eigenvectors.T @ ref_normalized)
                scores = remaining_eigenvalues * (1 - alignments)
                
                # Select top preserve_rank by score
                _, relative_indices = torch.sort(scores, descending=True)
                for idx in relative_indices[:preserve_rank]:
                    selected_indices.append(noise_rank + idx.item())
            else:
                # Reference is zero, just take next eigenvectors
                selected_indices.extend(range(noise_rank, rank))
        else:
            # No reference direction, just take next eigenvectors
            selected_indices.extend(range(noise_rank, rank))
        
        # Sort indices and return selected eigenvectors
        selected_indices = sorted(selected_indices[:rank])
        return eigenvectors[:, selected_indices]
    
    def get_name(self) -> str:
        return "hybrid"


class StrategyFactory:
    """Factory for creating strategy instances."""
    
    @staticmethod
    def create(
        strategy_name: Literal['noise', 'keep', 'hybrid'],
        **kwargs
    ) -> Strategy:
        """Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy ('noise', 'keep', 'hybrid')
            **kwargs: Additional arguments for strategy initialization
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy_name is unknown
        """
        if strategy_name == 'noise':
            return NoiseAwareStrategy()
        elif strategy_name == 'keep':
            return KnowledgePreservingStrategy()
        elif strategy_name == 'hybrid':
            noise_fraction = kwargs.get('noise_fraction', 0.5)
            return HybridStrategy(noise_fraction)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    @staticmethod
    def get_available_strategies() -> list[str]:
        """Get list of available strategy names.
        
        Returns:
            List of strategy names
        """
        return ['noise', 'keep', 'hybrid']