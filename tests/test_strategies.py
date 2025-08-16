"""
Unit tests for strategy selection module.
"""

import pytest
import torch
import numpy as np
from nspo2.strategies import (
    Strategy,
    NoiseAwareStrategy,
    KnowledgePreservingStrategy,
    HybridStrategy,
    StrategyFactory
)


class TestNoiseAwareStrategy:
    """Test NoiseAwareStrategy functionality."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = NoiseAwareStrategy()
        assert strategy.get_name() == "noise"
    
    def test_select_top_eigenvectors(self):
        """Test that strategy selects top-r eigenvectors."""
        strategy = NoiseAwareStrategy()
        
        # Create test eigenvectors and eigenvalues
        dim = 10
        n_eig = 8
        eigenvectors = torch.eye(dim)[:, :n_eig]
        eigenvalues = torch.arange(n_eig, 0, -1).float()  # [8, 7, 6, ..., 1]
        
        # Select top 3
        rank = 3
        selected = strategy.apply(eigenvectors, eigenvalues, rank)
        
        assert selected.shape == (dim, rank)
        # Should select first 3 columns (highest eigenvalues)
        assert torch.allclose(selected, eigenvectors[:, :rank])
    
    def test_edge_cases(self):
        """Test edge cases."""
        strategy = NoiseAwareStrategy()
        
        eigenvectors = torch.eye(5)
        eigenvalues = torch.ones(5)
        
        # rank = 0
        selected = strategy.apply(eigenvectors, eigenvalues, 0)
        assert selected.shape == (5, 0)
        
        # rank >= n_eigenvectors
        selected = strategy.apply(eigenvectors, eigenvalues, 10)
        assert torch.allclose(selected, eigenvectors)
    
    def test_reference_direction_ignored(self):
        """Test that reference direction is ignored."""
        strategy = NoiseAwareStrategy()
        
        eigenvectors = torch.eye(5)
        eigenvalues = torch.ones(5)
        reference = torch.randn(5)
        
        selected_no_ref = strategy.apply(eigenvectors, eigenvalues, 3)
        selected_with_ref = strategy.apply(eigenvectors, eigenvalues, 3, reference)
        
        assert torch.allclose(selected_no_ref, selected_with_ref)


class TestKnowledgePreservingStrategy:
    """Test KnowledgePreservingStrategy functionality."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = KnowledgePreservingStrategy()
        assert strategy.get_name() == "keep"
    
    def test_preserve_aligned_direction(self):
        """Test that strategy preserves directions aligned with reference."""
        strategy = KnowledgePreservingStrategy()
        
        dim = 5
        # Create eigenvectors where first is aligned with reference
        eigenvectors = torch.eye(dim)
        eigenvalues = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Reference direction aligned with first eigenvector
        reference = eigenvectors[:, 0]
        
        # Select 2 eigenvectors
        rank = 2
        selected = strategy.apply(eigenvectors, eigenvalues, rank, reference)
        
        # Should not select the first eigenvector (aligned with reference)
        # Should select eigenvectors with high eigenvalue but low alignment
        assert selected.shape == (dim, rank)
        
        # Check that first eigenvector is not in selection
        alignment_with_first = torch.abs(selected.T @ eigenvectors[:, 0])
        assert torch.all(alignment_with_first < 0.1)
    
    def test_orthogonal_reference(self):
        """Test behavior with orthogonal reference direction."""
        strategy = KnowledgePreservingStrategy()
        
        dim = 5
        eigenvectors = torch.eye(dim)
        eigenvalues = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        
        # Reference direction orthogonal to all eigenvectors
        reference = torch.ones(dim) / torch.sqrt(torch.tensor(dim).float())
        
        rank = 2
        selected = strategy.apply(eigenvectors, eigenvalues, rank, reference)
        
        # When reference is orthogonal to all, should behave like noise-aware
        # (select top eigenvalues)
        assert selected.shape == (dim, rank)
    
    def test_no_reference_fallback(self):
        """Test fallback when no reference direction provided."""
        strategy = KnowledgePreservingStrategy()
        
        eigenvectors = torch.eye(5)
        eigenvalues = torch.ones(5)
        
        selected = strategy.apply(eigenvectors, eigenvalues, 3, None)
        
        # Should fall back to selecting top eigenvectors
        assert torch.allclose(selected, eigenvectors[:, :3])
    
    def test_zero_reference_fallback(self):
        """Test fallback when reference direction is zero."""
        strategy = KnowledgePreservingStrategy()
        
        eigenvectors = torch.eye(5)
        eigenvalues = torch.ones(5)
        reference = torch.zeros(5)
        
        selected = strategy.apply(eigenvectors, eigenvalues, 3, reference)
        
        # Should fall back to selecting top eigenvectors
        assert torch.allclose(selected, eigenvectors[:, :3])


class TestHybridStrategy:
    """Test HybridStrategy functionality."""
    
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = HybridStrategy()
        assert strategy.get_name() == "hybrid"
        assert strategy.noise_fraction == 0.5
        
        # Custom noise fraction
        strategy2 = HybridStrategy(noise_fraction=0.7)
        assert strategy2.noise_fraction == 0.7
    
    def test_split_selection(self):
        """Test that strategy splits selection between approaches."""
        strategy = HybridStrategy(noise_fraction=0.5)
        
        dim = 10
        eigenvectors = torch.eye(dim)
        eigenvalues = torch.arange(10, 0, -1).float()
        
        # Reference aligned with middle eigenvector
        reference = eigenvectors[:, 5]
        
        rank = 4  # Should split: 2 for noise, 2 for preservation
        selected = strategy.apply(eigenvectors, eigenvalues, rank, reference)
        
        assert selected.shape == (dim, rank)
        
        # Should include top 2 eigenvectors (noise removal)
        selected_indices = []
        for i in range(dim):
            if torch.any(torch.abs(selected.T @ eigenvectors[:, i]) > 0.9):
                selected_indices.append(i)
        
        assert 0 in selected_indices  # Top eigenvalue
        assert 1 in selected_indices  # Second eigenvalue
    
    def test_extreme_fractions(self):
        """Test extreme noise fractions."""
        # All noise removal
        strategy_noise = HybridStrategy(noise_fraction=1.0)
        
        eigenvectors = torch.eye(5)
        eigenvalues = torch.ones(5)
        reference = torch.randn(5)
        
        selected = strategy_noise.apply(eigenvectors, eigenvalues, 3, reference)
        assert torch.allclose(selected, eigenvectors[:, :3])
        
        # All knowledge preservation
        strategy_keep = HybridStrategy(noise_fraction=0.0)
        selected2 = strategy_keep.apply(eigenvectors, eigenvalues, 3, reference)
        assert selected2.shape == (5, 3)
    
    def test_no_reference_handling(self):
        """Test handling when no reference provided."""
        strategy = HybridStrategy(noise_fraction=0.5)
        
        eigenvectors = torch.eye(5)
        eigenvalues = torch.ones(5)
        
        selected = strategy.apply(eigenvectors, eigenvalues, 3, None)
        
        # Should still work, using default selection for preserve part
        assert selected.shape == (5, 3)


class TestStrategyFactory:
    """Test StrategyFactory functionality."""
    
    def test_create_noise_aware(self):
        """Test creating NoiseAwareStrategy."""
        strategy = StrategyFactory.create('noise')
        assert isinstance(strategy, NoiseAwareStrategy)
        assert strategy.get_name() == 'noise'
    
    def test_create_knowledge_preserving(self):
        """Test creating KnowledgePreservingStrategy."""
        strategy = StrategyFactory.create('keep')
        assert isinstance(strategy, KnowledgePreservingStrategy)
        assert strategy.get_name() == 'keep'
    
    def test_create_hybrid(self):
        """Test creating HybridStrategy."""
        strategy = StrategyFactory.create('hybrid')
        assert isinstance(strategy, HybridStrategy)
        assert strategy.get_name() == 'hybrid'
        assert strategy.noise_fraction == 0.5
        
        # With custom noise fraction
        strategy2 = StrategyFactory.create('hybrid', noise_fraction=0.7)
        assert isinstance(strategy2, HybridStrategy)
        assert strategy2.noise_fraction == 0.7
    
    def test_invalid_strategy_name(self):
        """Test error on invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            StrategyFactory.create('invalid')
    
    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        strategies = StrategyFactory.get_available_strategies()
        assert 'noise' in strategies
        assert 'keep' in strategies
        assert 'hybrid' in strategies
        assert len(strategies) == 3


class TestIntegrationWithNSPO2:
    """Test integration with NSPO2 main class."""
    
    def test_strategy_application_in_nspo2(self):
        """Test that strategies are properly applied in NSPO2."""
        from nspo2 import NSPO2
        from nspo2.config import get_default_config
        
        # Test each strategy
        for strategy_name in ['noise', 'keep', 'hybrid']:
            config = get_default_config(dim=10)
            config['strategy'] = strategy_name
            config['update_freq'] = 5
            config['projection_rank'] = 3
            
            nspo2 = NSPO2(config)
            assert nspo2.strategy_obj.get_name() == strategy_name
            
            # Process gradients to trigger projection
            torch.manual_seed(42)
            for i in range(10):
                grad = torch.randn(10)
                grad[0] += 3.0  # Add structure
                
                projected = nspo2.hook(grad)
                
                if i >= config['update_freq']:
                    # Check that selected eigenvectors are set
                    assert nspo2.selected_eigenvectors is not None
                    assert nspo2.selected_eigenvectors.shape[1] <= config['projection_rank']
    
    def test_reference_direction_tracking(self):
        """Test that reference direction is properly tracked."""
        from nspo2 import NSPO2
        from nspo2.config import get_default_config
        
        config = get_default_config(dim=5)
        config['strategy'] = 'keep'
        config['update_freq'] = 3
        
        nspo2 = NSPO2(config)
        
        # Process gradients
        for i in range(5):
            grad = torch.ones(5) * (i + 1)
            nspo2.hook(grad)
            
            if i >= config['update_freq']:
                # Reference direction should be set
                assert nspo2.reference_direction is not None
                # Should be exponential moving average of gradients
                assert nspo2.reference_direction.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])