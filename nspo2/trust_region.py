"""
Trust region module for NSPO2.

Ensures projected gradients don't deviate too much from original gradients.
"""

import torch
from typing import Literal, Optional


class TrustRegion:
    """Trust region manager for constraining projection deviations.
    
    Implements mixing between original and projected gradients when
    the distance between them exceeds the trust radius.
    """
    
    def __init__(
        self,
        initial_alpha: float = 0.5,
        min_alpha: float = 0.1,
        max_alpha: float = 1.0,
        distance_metric: Literal['l2', 'cosine'] = 'l2',
        adaptive: bool = True,
        adaptation_rate: float = 0.1
    ):
        """Initialize trust region manager.
        
        Args:
            initial_alpha: Initial mixing coefficient
            min_alpha: Minimum mixing coefficient (more original gradient)
            max_alpha: Maximum mixing coefficient (more projected gradient)
            distance_metric: Distance metric to use ('l2' or 'cosine')
            adaptive: Whether to adapt trust radius based on history
            adaptation_rate: Rate of trust radius adaptation
        """
        self.initial_alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.distance_metric = distance_metric
        self.adaptive = adaptive
        self.adaptation_rate = adaptation_rate
        
        # State tracking
        self.last_distance = 0.0
        self.last_alpha = 1.0
        self.last_violation = False
        
        # History for adaptive trust radius
        self.distance_history = []
        self.violation_history = []
        self.adaptive_trust_radius_multiplier = 1.0
    
    def apply(
        self,
        original_grad: torch.Tensor,
        projected_grad: torch.Tensor,
        trust_radius: float
    ) -> torch.Tensor:
        """Apply trust region constraint to projected gradient.
        
        Args:
            original_grad: Original gradient
            projected_grad: Projected gradient
            trust_radius: Trust radius threshold
            
        Returns:
            Mixed gradient respecting trust region
        """
        # Calculate distance
        distance = self._calculate_distance(original_grad, projected_grad)
        self.last_distance = distance
        
        # Apply adaptive trust radius if enabled
        if self.adaptive:
            effective_trust_radius = trust_radius * self.adaptive_trust_radius_multiplier
        else:
            effective_trust_radius = trust_radius
        
        # Check if distance exceeds trust radius
        if distance > effective_trust_radius:
            # Compute mixing coefficient
            alpha = self.get_mixing_coefficient(distance, effective_trust_radius)
            self.last_alpha = alpha
            self.last_violation = True
            
            # Apply mixed update: α·Pg + (1-α)·g
            mixed_grad = alpha * projected_grad + (1 - alpha) * original_grad
        else:
            self.last_alpha = 1.0
            self.last_violation = False
            mixed_grad = projected_grad
        
        # Update history
        self.distance_history.append(distance)
        self.violation_history.append(self.last_violation)
        
        # Adapt trust radius if enabled
        if self.adaptive:
            self._adapt_trust_radius()
        
        return mixed_grad
    
    def _calculate_distance(
        self,
        original_grad: torch.Tensor,
        projected_grad: torch.Tensor
    ) -> float:
        """Calculate distance between gradients.
        
        Args:
            original_grad: Original gradient
            projected_grad: Projected gradient
            
        Returns:
            Distance value
        """
        if self.distance_metric == 'l2':
            # L2 norm of difference
            distance = torch.norm(projected_grad - original_grad).item()
        elif self.distance_metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            cosine_sim = torch.nn.functional.cosine_similarity(
                original_grad.flatten().unsqueeze(0),
                projected_grad.flatten().unsqueeze(0)
            ).item()
            distance = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distance
    
    def get_mixing_coefficient(self, distance: float, trust_radius: float) -> float:
        """Compute mixing coefficient based on distance and trust radius.
        
        Uses smooth interpolation to avoid abrupt changes.
        
        Args:
            distance: Distance between gradients
            trust_radius: Trust radius threshold
            
        Returns:
            Mixing coefficient α in [min_alpha, max_alpha]
        """
        if distance <= trust_radius:
            return self.max_alpha
        
        # Smooth transition using sigmoid-like function
        # When distance = trust_radius, α = initial_alpha
        # As distance increases, α decreases toward min_alpha
        
        ratio = trust_radius / max(distance, 1e-10)
        
        # Use smooth interpolation
        # α = min_alpha + (max_alpha - min_alpha) * sigmoid(k * (ratio - 0.5))
        # Simplified version:
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * ratio
        
        # Ensure within bounds
        alpha = min(max(alpha, self.min_alpha), self.max_alpha)
        
        return alpha
    
    def _adapt_trust_radius(self):
        """Adapt trust radius multiplier based on recent history."""
        if len(self.violation_history) < 10:
            return
        
        # Calculate recent violation rate
        recent_violations = self.violation_history[-20:]
        violation_rate = sum(recent_violations) / len(recent_violations)
        
        # Adapt multiplier
        if violation_rate > 0.5:
            # Too many violations, increase trust radius
            self.adaptive_trust_radius_multiplier *= (1 + self.adaptation_rate)
        elif violation_rate < 0.2:
            # Few violations, can decrease trust radius
            self.adaptive_trust_radius_multiplier *= (1 - self.adaptation_rate * 0.5)
        
        # Keep multiplier in reasonable range
        self.adaptive_trust_radius_multiplier = min(
            max(self.adaptive_trust_radius_multiplier, 0.5), 2.0
        )
    
    def get_last_distance(self) -> float:
        """Get last calculated distance.
        
        Returns:
            Last distance value
        """
        return self.last_distance
    
    def get_last_alpha(self) -> float:
        """Get last used mixing coefficient.
        
        Returns:
            Last alpha value
        """
        return self.last_alpha
    
    def get_violation_rate(self, window: int = 100) -> float:
        """Get recent trust region violation rate.
        
        Args:
            window: Number of recent steps to consider
            
        Returns:
            Violation rate (0 to 1)
        """
        if not self.violation_history:
            return 0.0
        
        recent = self.violation_history[-window:]
        return sum(recent) / len(recent)
    
    def reset(self):
        """Reset trust region state."""
        self.last_distance = 0.0
        self.last_alpha = 1.0
        self.last_violation = False
        self.distance_history = []
        self.violation_history = []
        self.adaptive_trust_radius_multiplier = 1.0