"""
Configuration classes for NSPO2 module.
"""

from typing import TypedDict, Literal, Optional


class NSPO2Config(TypedDict):
    """Configuration for NSPO2 module.
    
    Attributes:
        dim: Feature/gradient dimension
        strategy: Projection strategy ('noise', 'hybrid', 'keep')
        projection_rank: Number of top eigenvectors to project (default 16)
        update_freq: Update eigenvectors every k steps
        trust_radius: Upper bound for ‖Pg-g‖
        target_bias: Upper bound for bias estimation
        max_rank: Maximum projection rank (default 32)
        min_rank: Minimum projection rank (default 0)
        epsilon: Numerical stability term
        seed: Random seed for reproducibility
        cov_momentum: Momentum for covariance EMA (default 0.9)
        shrinkage: Shrinkage parameter for covariance regularization (default 0.1)
        use_randomized_svd: Use randomized SVD for large dimensions (default True)
        svd_oversampling: Oversampling parameter for randomized SVD (default 10)
        bias_gamma: Weight for bias term in adaptive rank objective (default 1.0)
        auto_strategy: Enable automatic strategy switching (default False)
        verbose: Enable verbose logging (default False)
    """
    dim: int
    strategy: Literal['noise', 'hybrid', 'keep']
    projection_rank: int
    update_freq: int
    trust_radius: float
    target_bias: float
    max_rank: int
    min_rank: int
    epsilon: float
    seed: int
    
    # Optional parameters with defaults
    cov_momentum: Optional[float]
    shrinkage: Optional[float]
    use_randomized_svd: Optional[bool]
    svd_oversampling: Optional[int]
    bias_gamma: Optional[float]
    auto_strategy: Optional[bool]
    verbose: Optional[bool]


def validate_config(config: NSPO2Config) -> None:
    """Validate NSPO2 configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    # Required parameters validation
    if config['dim'] <= 0:
        raise ValueError(f"dim must be positive, got {config['dim']}")
    
    if config['strategy'] not in ['noise', 'hybrid', 'keep']:
        raise ValueError(f"strategy must be 'noise', 'hybrid', or 'keep', got {config['strategy']}")
    
    if config['projection_rank'] <= 0:
        raise ValueError(f"projection_rank must be positive, got {config['projection_rank']}")
    
    if config['projection_rank'] > config['dim']:
        raise ValueError(f"projection_rank ({config['projection_rank']}) cannot exceed dim ({config['dim']})")
    
    if config['update_freq'] <= 0:
        raise ValueError(f"update_freq must be positive, got {config['update_freq']}")
    
    if config['trust_radius'] <= 0:
        raise ValueError(f"trust_radius must be positive, got {config['trust_radius']}")
    
    if config['target_bias'] < 0:
        raise ValueError(f"target_bias must be non-negative, got {config['target_bias']}")
    
    if config['max_rank'] <= 0:
        raise ValueError(f"max_rank must be positive, got {config['max_rank']}")
    
    if config['min_rank'] < 0:
        raise ValueError(f"min_rank must be non-negative, got {config['min_rank']}")
    
    if config['min_rank'] > config['max_rank']:
        raise ValueError(f"min_rank ({config['min_rank']}) cannot exceed max_rank ({config['max_rank']})")
    
    if config['max_rank'] > config['dim']:
        raise ValueError(f"max_rank ({config['max_rank']}) cannot exceed dim ({config['dim']})")
    
    if config['epsilon'] <= 0:
        raise ValueError(f"epsilon must be positive, got {config['epsilon']}")
    
    # Optional parameters validation
    if 'cov_momentum' in config and config['cov_momentum'] is not None:
        if not 0 < config['cov_momentum'] <= 1:
            raise ValueError(f"cov_momentum must be in (0, 1], got {config['cov_momentum']}")
    
    if 'shrinkage' in config and config['shrinkage'] is not None:
        if not 0 <= config['shrinkage'] < 1:
            raise ValueError(f"shrinkage must be in [0, 1), got {config['shrinkage']}")
    
    if 'svd_oversampling' in config and config['svd_oversampling'] is not None:
        if config['svd_oversampling'] < 0:
            raise ValueError(f"svd_oversampling must be non-negative, got {config['svd_oversampling']}")
    
    if 'bias_gamma' in config and config['bias_gamma'] is not None:
        if config['bias_gamma'] < 0:
            raise ValueError(f"bias_gamma must be non-negative, got {config['bias_gamma']}")


def get_default_config(dim: int, seed: int = 42) -> NSPO2Config:
    """Get default NSPO2 configuration.
    
    Args:
        dim: Feature/gradient dimension
        seed: Random seed
        
    Returns:
        Default configuration dictionary
    """
    return NSPO2Config(
        dim=dim,
        strategy='hybrid',
        projection_rank=min(16, dim // 2),
        update_freq=10,
        trust_radius=0.1,
        target_bias=0.05,
        max_rank=min(32, dim // 2),
        min_rank=0,
        epsilon=1e-6,
        seed=seed,
        cov_momentum=0.9,
        shrinkage=0.1,
        use_randomized_svd=dim > 100,
        svd_oversampling=10,
        bias_gamma=1.0,
        auto_strategy=False,
        verbose=False
    )