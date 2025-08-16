"""
Optimizer wrapper for NSPO2 integration.

Allows NSPO2 to be used as a drop-in replacement with existing optimizers.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Dict, Any
from .core import NSPO2
from .config import NSPO2Config


class NSPO2OptimizerWrapper:
    """Wrapper that applies NSPO2 projection to gradients before optimization.
    
    This wrapper intercepts gradients, applies NSPO2 projection, and then
    forwards them to the wrapped optimizer.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        nspo2_config: NSPO2Config,
        enabled: bool = True
    ):
        """Initialize optimizer wrapper.
        
        Args:
            optimizer: PyTorch optimizer to wrap
            nspo2_config: NSPO2 configuration
            enabled: Whether NSPO2 is initially enabled
        """
        self.optimizer = optimizer
        self.nspo2 = NSPO2(nspo2_config)
        self.enabled = enabled
        self.step_count = 0
        
        # Store original step method
        self._original_step = optimizer.step
        
        # Replace optimizer's step method
        optimizer.step = self.step
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step with NSPO2 projection.
        
        Args:
            closure: Optional closure for recomputing loss
            
        Returns:
            Loss value if closure is provided
        """
        if not self.enabled:
            # NSPO2 disabled, use original optimizer
            return self._original_step(closure)
        
        # Apply NSPO2 projection to all gradients
        with torch.no_grad():
            # Collect all gradients
            all_grads = []
            grad_shapes = []
            params_with_grad = []
            
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        all_grads.append(param.grad.view(-1))
                        grad_shapes.append(param.grad.shape)
                        params_with_grad.append(param)
            
            if all_grads:
                # Concatenate all gradients
                concatenated_grad = torch.cat(all_grads)
                
                # Apply NSPO2 projection
                projected_grad = self.nspo2.hook(concatenated_grad)
                
                # Split back and update parameters
                start_idx = 0
                for param, shape in zip(params_with_grad, grad_shapes):
                    numel = param.grad.numel()
                    param.grad.copy_(projected_grad[start_idx:start_idx + numel].view(shape))
                    start_idx += numel
        
        # Call original optimizer step
        loss = self._original_step(closure)
        
        # Update NSPO2 state
        self.nspo2.step_end()
        self.step_count += 1
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients.
        
        Args:
            set_to_none: Whether to set gradients to None instead of zero
        """
        self.optimizer.zero_grad(set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary.
        
        Returns:
            State dictionary containing optimizer and NSPO2 states
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'nspo2': self.nspo2.state_dict(),
            'step_count': self.step_count,
            'enabled': self.enabled
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.nspo2.load_state_dict(state_dict['nspo2'])
        self.step_count = state_dict.get('step_count', 0)
        self.enabled = state_dict.get('enabled', True)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable NSPO2.
        
        Args:
            enabled: Whether to enable NSPO2
        """
        self.enabled = enabled
        self.nspo2.set_active(enabled)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get NSPO2 metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.nspo2.get_metrics()
    
    @property
    def param_groups(self):
        """Access to optimizer's param groups."""
        return self.optimizer.param_groups


def create_nspo2_optimizer(
    model_parameters,
    optimizer_class: type,
    optimizer_kwargs: Dict[str, Any],
    nspo2_config: Optional[NSPO2Config] = None,
    auto_dim: bool = True
) -> NSPO2OptimizerWrapper:
    """Convenience function to create NSPO2-wrapped optimizer.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        optimizer_kwargs: Keyword arguments for optimizer
        nspo2_config: NSPO2 configuration (if None and auto_dim=True, will auto-calculate dimension)
        auto_dim: If True and nspo2_config is None, automatically calculate dimension
        
    Returns:
        NSPO2OptimizerWrapper instance
    
    Example:
        >>> import torch.optim as optim
        >>> from nspo2 import create_nspo2_optimizer, get_default_config
        >>> 
        >>> model = MyModel()
        >>> # Option 1: Auto-calculate dimension
        >>> optimizer = create_nspo2_optimizer(
        ...     model.parameters(),
        ...     optim.Adam,
        ...     {'lr': 1e-3}
        ... )
        >>> 
        >>> # Option 2: Manual configuration
        >>> config = get_default_config(dim=model.num_parameters())
        >>> optimizer = create_nspo2_optimizer(
        ...     model.parameters(),
        ...     optim.Adam,
        ...     {'lr': 1e-3},
        ...     config
        ... )
    """
    # Convert parameters to list if it's a generator
    if not isinstance(model_parameters, list):
        model_parameters = list(model_parameters)
    
    # Auto-calculate dimension if needed
    if nspo2_config is None and auto_dim:
        from .config import get_default_config
        total_dim = sum(p.numel() for p in model_parameters if p.requires_grad)
        nspo2_config = get_default_config(dim=total_dim)
    elif nspo2_config is None:
        raise ValueError("nspo2_config must be provided if auto_dim=False")
    
    base_optimizer = optimizer_class(model_parameters, **optimizer_kwargs)
    return NSPO2OptimizerWrapper(base_optimizer, nspo2_config)