"""
Unit tests for NSPO2 configuration.
"""

import pytest
import torch
from nspo2 import NSPO2Config, NSPO2
from nspo2.config import validate_config, get_default_config


class TestNSPO2Config:
    """Test NSPO2Config validation and functionality."""
    
    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = NSPO2Config(
            dim=100,
            strategy='noise',
            projection_rank=16,
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=32,
            min_rank=0,
            epsilon=1e-6,
            seed=42
        )
        # Should not raise
        validate_config(config)
    
    def test_invalid_dim(self):
        """Test invalid dimension raises error."""
        config = NSPO2Config(
            dim=0,  # Invalid
            strategy='noise',
            projection_rank=16,
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=32,
            min_rank=0,
            epsilon=1e-6,
            seed=42
        )
        with pytest.raises(ValueError, match="dim must be positive"):
            validate_config(config)
    
    def test_invalid_strategy(self):
        """Test invalid strategy raises error."""
        config = NSPO2Config(
            dim=100,
            strategy='invalid',  # Invalid
            projection_rank=16,
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=32,
            min_rank=0,
            epsilon=1e-6,
            seed=42
        )
        with pytest.raises(ValueError, match="strategy must be"):
            validate_config(config)
    
    def test_projection_rank_exceeds_dim(self):
        """Test projection_rank > dim raises error."""
        config = NSPO2Config(
            dim=10,
            strategy='noise',
            projection_rank=20,  # Exceeds dim
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=32,
            min_rank=0,
            epsilon=1e-6,
            seed=42
        )
        with pytest.raises(ValueError, match="projection_rank .* cannot exceed dim"):
            validate_config(config)
    
    def test_min_rank_exceeds_max_rank(self):
        """Test min_rank > max_rank raises error."""
        config = NSPO2Config(
            dim=100,
            strategy='noise',
            projection_rank=16,
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=10,
            min_rank=20,  # Exceeds max_rank
            epsilon=1e-6,
            seed=42
        )
        with pytest.raises(ValueError, match="min_rank .* cannot exceed max_rank"):
            validate_config(config)
    
    def test_optional_parameters(self):
        """Test optional parameters validation."""
        config = NSPO2Config(
            dim=100,
            strategy='noise',
            projection_rank=16,
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=32,
            min_rank=0,
            epsilon=1e-6,
            seed=42,
            cov_momentum=0.95,
            shrinkage=0.2,
            use_randomized_svd=True,
            svd_oversampling=5,
            bias_gamma=2.0,
            auto_strategy=True,
            verbose=True
        )
        validate_config(config)
    
    def test_invalid_cov_momentum(self):
        """Test invalid cov_momentum raises error."""
        config = NSPO2Config(
            dim=100,
            strategy='noise',
            projection_rank=16,
            update_freq=10,
            trust_radius=0.1,
            target_bias=0.05,
            max_rank=32,
            min_rank=0,
            epsilon=1e-6,
            seed=42,
            cov_momentum=1.5  # Invalid
        )
        with pytest.raises(ValueError, match="cov_momentum must be in"):
            validate_config(config)
    
    def test_get_default_config(self):
        """Test default configuration generation."""
        config = get_default_config(dim=50, seed=123)
        assert config['dim'] == 50
        assert config['seed'] == 123
        assert config['strategy'] == 'hybrid'
        assert config['projection_rank'] == 16  # min(16, 50//2)
        assert config['max_rank'] == 25  # min(32, 50//2)
        validate_config(config)


class TestNSPO2Core:
    """Test NSPO2 core functionality."""
    
    def test_initialization(self):
        """Test NSPO2 initialization with valid config."""
        config = get_default_config(dim=100, seed=42)
        nspo2 = NSPO2(config)
        
        assert nspo2.dim == 100
        assert nspo2.strategy == 'hybrid'
        assert nspo2.step_count == 0
        assert nspo2.is_active == True
    
    def test_hook_identity_transformation(self):
        """Test hook performs identity transformation initially."""
        config = get_default_config(dim=10, seed=42)
        nspo2 = NSPO2(config)
        
        grad = torch.randn(10)
        projected_grad = nspo2.hook(grad)
        
        # Should be identity transformation for now
        assert torch.allclose(grad, projected_grad)
        assert nspo2.step_count == 1
    
    def test_hook_inactive(self):
        """Test hook returns original gradient when inactive."""
        config = get_default_config(dim=10, seed=42)
        nspo2 = NSPO2(config)
        nspo2.set_active(False)
        
        grad = torch.randn(10)
        projected_grad = nspo2.hook(grad)
        
        assert torch.allclose(grad, projected_grad)
    
    def test_state_dict_save_load(self):
        """Test state_dict save and load functionality."""
        config = get_default_config(dim=20, seed=42)
        nspo2 = NSPO2(config)
        
        # Perform some operations
        for _ in range(5):
            grad = torch.randn(20)
            nspo2.hook(grad)
            nspo2.step_end({'loss': torch.rand(1).item()})
        
        # Save state
        state = nspo2.state_dict()
        assert state['step_count'] == 5
        assert len(state['history']['variance_original']) == 5
        
        # Create new instance and load state
        nspo2_new = NSPO2(get_default_config(dim=20, seed=99))
        nspo2_new.load_state_dict(state)
        
        assert nspo2_new.step_count == 5
        assert len(nspo2_new.history['variance_original']) == 5
        assert nspo2_new.cfg == config
    
    def test_step_end_metrics(self):
        """Test step_end stores external metrics."""
        config = get_default_config(dim=10, seed=42)
        nspo2 = NSPO2(config)
        
        metrics = {'loss': 0.5, 'reward': 10.0}
        nspo2.step_end(metrics)
        
        assert 'loss' in nspo2.history
        assert 'reward' in nspo2.history
        assert nspo2.history['loss'] == [0.5]
        assert nspo2.history['reward'] == [10.0]
    
    def test_reset(self):
        """Test reset functionality."""
        config = get_default_config(dim=10, seed=42)
        nspo2 = NSPO2(config)
        
        # Perform some operations
        for _ in range(3):
            grad = torch.randn(10)
            nspo2.hook(grad)
        
        assert nspo2.step_count == 3
        assert len(nspo2.history['variance_original']) == 3
        
        # Reset
        nspo2.reset()
        
        assert nspo2.step_count == 0
        assert len(nspo2.history['variance_original']) == 0
        assert nspo2.is_active == True
    
    def test_get_metrics(self):
        """Test get_metrics returns current statistics."""
        config = get_default_config(dim=10, seed=42)
        nspo2 = NSPO2(config)
        
        # Perform some operations
        for i in range(10):
            grad = torch.randn(10)
            nspo2.hook(grad)
            nspo2.step_end({'loss': 0.1 * i})
        
        metrics = nspo2.get_metrics()
        
        assert metrics['step_count'] == 10
        assert metrics['is_active'] == True
        assert metrics['current_rank'] == config['projection_rank']
        assert metrics['current_strategy'] == 'hybrid'
        assert 'recent_variance_reduction' in metrics
    
    def test_set_active(self):
        """Test activation/deactivation."""
        config = get_default_config(dim=10, seed=42)
        config['verbose'] = True
        nspo2 = NSPO2(config)
        
        assert nspo2.is_active == True
        
        nspo2.set_active(False)
        assert nspo2.is_active == False
        
        nspo2.set_active(True)
        assert nspo2.is_active == True
    
    def test_different_strategies(self):
        """Test initialization with different strategies."""
        for strategy in ['noise', 'hybrid', 'keep']:
            config = get_default_config(dim=10, seed=42)
            config['strategy'] = strategy
            nspo2 = NSPO2(config)
            assert nspo2.strategy == strategy
    
    def test_verbose_mode(self):
        """Test verbose mode prints information."""
        config = get_default_config(dim=10, seed=42)
        config['verbose'] = True
        
        # Should print during initialization
        nspo2 = NSPO2(config)
        
        # Should print during hook after 100 steps
        for i in range(101):
            grad = torch.randn(10)
            nspo2.hook(grad)
        
        # Should print during step_end after 100 steps
        nspo2.step_end()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])