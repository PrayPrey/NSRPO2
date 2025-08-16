"""
Core NSPO2 implementation.
"""

import torch
from typing import Dict, Any, Optional
import numpy as np
from .config import NSPO2Config, validate_config
from .covariance import CovEstimator
from .eigensolvers import EigSolver, ProjectionHead
from .strategies import StrategyFactory
from .trust_region import TrustRegion


class NSPO2:
    """Null Space Preference Optimization v2 main class.
    
    This class implements the core NSPO2 functionality for variance reduction
    in policy gradient and preference optimization through null space projection.
    """
    
    def __init__(self, cfg: NSPO2Config):
        """Initialize NSPO2 with configuration.
        
        Args:
            cfg: NSPO2 configuration dictionary
        """
        # Validate configuration
        validate_config(cfg)
        
        # Store configuration
        self.cfg = cfg
        self.dim = cfg['dim']
        self.strategy = cfg['strategy']
        self.projection_rank = cfg['projection_rank']
        self.update_freq = cfg['update_freq']
        self.trust_radius = cfg['trust_radius']
        self.target_bias = cfg['target_bias']
        self.max_rank = cfg['max_rank']
        self.min_rank = cfg['min_rank']
        self.epsilon = cfg['epsilon']
        
        # Optional parameters with defaults
        self.cov_momentum = cfg.get('cov_momentum', 0.9)
        self.shrinkage = cfg.get('shrinkage', 0.1)
        self.use_randomized_svd = cfg.get('use_randomized_svd', self.dim > 100)
        self.svd_oversampling = cfg.get('svd_oversampling', 10)
        self.bias_gamma = cfg.get('bias_gamma', 1.0)
        self.auto_strategy = cfg.get('auto_strategy', False)
        self.verbose = cfg.get('verbose', False)
        
        # Set random seed for reproducibility
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        
        # Initialize internal state
        self.step_count = 0
        self.is_active = True
        
        # Initialize components
        estimation_type = 'full' if self.dim <= 1000 else 'low_rank'
        self.cov_estimator = CovEstimator(
            dim=self.dim,
            decay=self.cov_momentum,
            shrinkage=self.shrinkage,
            epsilon=self.epsilon,
            estimation_type=estimation_type,
            rank=min(self.max_rank, self.dim // 4) if estimation_type == 'low_rank' else None
        )
        
        self.eig_solver = EigSolver(
            dim=self.dim,
            use_randomized=self.use_randomized_svd,
            n_power_iterations=2,
            oversampling=self.svd_oversampling,
            randomized_threshold=100,
            epsilon=self.epsilon,
            seed=cfg['seed']
        )
        
        self.projection_head = ProjectionHead(
            dim=self.dim,
            epsilon=self.epsilon,
            use_efficient=True
        )
        
        # Initialize strategy
        self.strategy_obj = StrategyFactory.create(self.strategy)
        self.reference_direction = None  # Will track average gradient or other reference
        
        # Initialize trust region
        self.trust_region = TrustRegion(
            initial_alpha=0.5,
            min_alpha=0.1,
            max_alpha=1.0,
            distance_metric='l2',
            adaptive=True,
            adaptation_rate=0.1
        )
        
        self.eigenvectors = None
        self.eigenvalues = None
        self.selected_eigenvectors = None  # Eigenvectors selected by strategy
        self.last_update_step = 0
        
        # Metrics tracking
        self.history = {
            'variance_original': [],
            'variance_projected': [],
            'variance_reduction': [],
            'bias_estimate': [],
            'trust_region_violations': [],
            'rank_used': [],
            'strategy_used': []
        }
        
        if self.verbose:
            print(f"NSPO2 initialized with dim={self.dim}, strategy={self.strategy}, "
                  f"projection_rank={self.projection_rank}")
    
    def hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Transform gradient g to projected gradient Pg.
        
        This is the main interface for gradient transformation. It applies
        null space projection to reduce variance while managing bias.
        
        Args:
            grad: Input gradient tensor
            
        Returns:
            Projected gradient tensor
        """
        # Increment step counter
        self.step_count += 1
        
        # Check if NSPO is active
        if not self.is_active:
            return grad
        
        # Store original shape and flatten
        original_shape = grad.shape
        grad_flat = grad.flatten()
        
        # Update covariance estimate
        self.cov_estimator.update(grad_flat)
        
        # Check if we should update eigenvectors
        should_update = (self.step_count - self.last_update_step) >= self.update_freq
        
        if should_update and self.cov_estimator.n_updates > 0:
            # Get current covariance matrix
            cov = self.cov_estimator.get_covariance()
            
            # Compute eigendecomposition (get more eigenvectors than needed for strategy selection)
            k_compute = min(self.projection_rank * 2, self.dim)  # Get extra for strategy flexibility
            self.eigenvalues, self.eigenvectors = self.eig_solver.solve(cov, k_compute)
            
            # Update reference direction (exponential moving average of gradients)
            if self.reference_direction is None:
                self.reference_direction = grad_flat.clone()
            else:
                self.reference_direction = 0.9 * self.reference_direction + 0.1 * grad_flat
            
            # Apply strategy to select eigenvectors
            self.selected_eigenvectors = self.strategy_obj.apply(
                self.eigenvectors,
                self.eigenvalues,
                self.projection_rank,
                self.reference_direction
            )
            
            # Build projection matrix with selected eigenvectors
            self.projection_head.build_projection(self.selected_eigenvectors)
            
            self.last_update_step = self.step_count
        
        # Apply projection if we have eigenvectors
        if self.eigenvectors is not None:
            projected_flat = self.projection_head.project(grad_flat)
            
            # Apply trust region constraint
            mixed_flat = self.trust_region.apply(grad_flat, projected_flat, self.trust_radius)
            projected_grad = mixed_flat.reshape(original_shape)
        else:
            # No projection yet (not enough updates)
            projected_grad = grad.clone()
        
        # Record metrics
        if grad.numel() > 0:
            orig_var = grad.var().item() if grad.numel() > 1 else 0.0
            proj_var = projected_grad.var().item() if projected_grad.numel() > 1 else 0.0
            var_reduction = (orig_var - proj_var) / (orig_var + self.epsilon) if orig_var > 0 else 0.0
            
            # Estimate bias (cosine similarity between original and projected)
            if self.eigenvectors is not None:
                cosine_sim = torch.nn.functional.cosine_similarity(
                    grad_flat.unsqueeze(0),
                    projected_grad.flatten().unsqueeze(0)
                ).item()
                bias_estimate = 1.0 - cosine_sim
            else:
                bias_estimate = 0.0
            
            self.history['variance_original'].append(orig_var)
            self.history['variance_projected'].append(proj_var)
            self.history['variance_reduction'].append(var_reduction)
            self.history['bias_estimate'].append(bias_estimate)
            self.history['trust_region_violations'].append(1 if self.trust_region.last_violation else 0)
            self.history['rank_used'].append(self.projection_rank)
            self.history['strategy_used'].append(self.strategy)
        
        if self.verbose and self.step_count % 100 == 0:
            cond_number = self.cov_estimator.get_condition_number()
            recent_var_red = np.mean(self.history['variance_reduction'][-10:]) if len(self.history['variance_reduction']) >= 10 else 0
            print(f"NSPO2 step {self.step_count}: shape={grad.shape}, "
                  f"cov_cond={cond_number:.2e}, var_reduction={recent_var_red:.2%}")
        
        return projected_grad
    
    def step_end(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Update history and internal state at the end of optimization step.
        
        This method should be called after each optimization step to update
        internal metrics and potentially adjust NSPO2 parameters.
        
        Args:
            metrics: Optional dictionary of external metrics (e.g., loss, reward)
        """
        # Store external metrics if provided
        if metrics is not None:
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
        
        # Placeholder for future implementations:
        # - Update covariance estimates
        # - Check for strategy switching
        # - Update adaptive rank
        # - Check bias sentinel conditions
        
        if self.verbose and self.step_count % 100 == 0:
            recent_var_reduction = np.mean(self.history['variance_reduction'][-10:]) if self.history['variance_reduction'] else 0
            print(f"Recent variance reduction: {recent_var_reduction:.2%}")
    
    def state_dict(self) -> Dict[str, Any]:
        """Save state for checkpointing.
        
        Returns:
            Dictionary containing all state information
        """
        state = {
            'config': self.cfg,
            'step_count': self.step_count,
            'is_active': self.is_active,
            'history': self.history,
            'cov_estimator_state': {
                'mean': self.cov_estimator.mean,
                'cov': self.cov_estimator.cov,
                'n_updates': self.cov_estimator.n_updates,
                'effective_samples': self.cov_estimator.effective_samples,
                'estimation_type': self.cov_estimator.estimation_type
            },
            'eigenvectors': self.eigenvectors,
            'eigenvalues': self.eigenvalues,
            'last_update_step': self.last_update_step,
            'cov_momentum': self.cov_momentum,
            'shrinkage': self.shrinkage,
            'use_randomized_svd': self.use_randomized_svd,
            'svd_oversampling': self.svd_oversampling,
            'bias_gamma': self.bias_gamma,
            'auto_strategy': self.auto_strategy,
            'verbose': self.verbose
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.
        
        Args:
            state_dict: Dictionary containing state information
        """
        # Restore configuration
        self.cfg = state_dict['config']
        validate_config(self.cfg)
        
        # Restore basic parameters from config
        self.dim = self.cfg['dim']
        self.strategy = self.cfg['strategy']
        self.projection_rank = self.cfg['projection_rank']
        self.update_freq = self.cfg['update_freq']
        self.trust_radius = self.cfg['trust_radius']
        self.target_bias = self.cfg['target_bias']
        self.max_rank = self.cfg['max_rank']
        self.min_rank = self.cfg['min_rank']
        self.epsilon = self.cfg['epsilon']
        
        # Restore state
        self.step_count = state_dict['step_count']
        self.is_active = state_dict['is_active']
        self.history = state_dict['history']
        
        # Restore covariance estimator state
        if 'cov_estimator_state' in state_dict:
            cov_state = state_dict['cov_estimator_state']
            self.cov_estimator.mean = cov_state['mean']
            self.cov_estimator.cov = cov_state['cov']
            self.cov_estimator.n_updates = cov_state['n_updates']
            self.cov_estimator.effective_samples = cov_state['effective_samples']
        
        self.eigenvectors = state_dict.get('eigenvectors')
        self.eigenvalues = state_dict.get('eigenvalues')
        self.last_update_step = state_dict.get('last_update_step', 0)
        
        # Rebuild projection if we have eigenvectors
        if self.eigenvectors is not None:
            self.projection_head.build_projection(self.eigenvectors)
        
        # Restore optional parameters
        self.cov_momentum = state_dict.get('cov_momentum', 0.9)
        self.shrinkage = state_dict.get('shrinkage', 0.1)
        self.use_randomized_svd = state_dict.get('use_randomized_svd', self.dim > 100)
        self.svd_oversampling = state_dict.get('svd_oversampling', 10)
        self.bias_gamma = state_dict.get('bias_gamma', 1.0)
        self.auto_strategy = state_dict.get('auto_strategy', False)
        self.verbose = state_dict.get('verbose', False)
        
        if self.verbose:
            print(f"NSPO2 state loaded: step_count={self.step_count}, is_active={self.is_active}")
    
    def reset(self) -> None:
        """Reset NSPO2 state to initial values."""
        self.step_count = 0
        self.is_active = True
        self.cov_estimator.reset()
        self.eigenvectors = None
        self.eigenvalues = None
        self.last_update_step = 0
        self.projection_head = ProjectionHead(
            dim=self.dim,
            epsilon=self.epsilon,
            use_efficient=True
        )
        self.history = {
            'variance_original': [],
            'variance_projected': [],
            'variance_reduction': [],
            'bias_estimate': [],
            'trust_region_violations': [],
            'rank_used': [],
            'strategy_used': []
        }
        
        if self.verbose:
            print("NSPO2 state reset")
    
    def set_active(self, active: bool) -> None:
        """Enable or disable NSPO2.
        
        Args:
            active: Whether to activate NSPO2
        """
        self.is_active = active
        if self.verbose:
            print(f"NSPO2 {'activated' if active else 'deactivated'}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics and statistics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'step_count': self.step_count,
            'is_active': self.is_active,
            'current_rank': self.projection_rank,
            'current_strategy': self.strategy,
        }
        
        # Add recent statistics if available
        if self.history['variance_reduction']:
            metrics['recent_variance_reduction'] = np.mean(self.history['variance_reduction'][-10:])
        if self.history['bias_estimate']:
            metrics['recent_bias'] = np.mean(self.history['bias_estimate'][-10:])
        if self.history['trust_region_violations']:
            metrics['total_violations'] = sum(self.history['trust_region_violations'])
        
        return metrics