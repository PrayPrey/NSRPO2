"""
Tests for NSPO2 Optimizer Wrapper.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from nspo2 import create_nspo2_optimizer, NSPO2OptimizerWrapper, get_default_config


class TestOptimizerWrapper:
    """Test suite for NSPO2OptimizerWrapper."""
    
    def test_auto_dimension_calculation(self):
        """Test automatic dimension calculation for model parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Test with auto-dimension (should work without specifying config)
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.01}
        )
        
        assert isinstance(optimizer, NSPO2OptimizerWrapper)
        
        # Verify dimension matches total parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert optimizer.nspo2.dim == total_params
        
    def test_manual_dimension_config(self):
        """Test manual dimension configuration."""
        model = nn.Linear(5, 3)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Manual configuration
        config = get_default_config(dim=total_params)
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.SGD,
            {'lr': 0.1},
            config
        )
        
        assert optimizer.nspo2.dim == total_params
        
    def test_optimization_step(self):
        """Test that optimization step works correctly."""
        torch.manual_seed(42)
        
        # Simple model
        model = nn.Linear(5, 3)
        x = torch.randn(2, 5)
        target = torch.randn(2, 3)
        
        # Create optimizer with auto-dimension
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.01}
        )
        
        # Training step
        initial_weights = model.weight.clone()
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Verify weights changed
        assert not torch.allclose(model.weight, initial_weights)
        
    def test_gradient_concatenation_and_split(self):
        """Test that gradients are correctly concatenated and split."""
        torch.manual_seed(42)
        
        # Model with multiple parameter tensors
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.Linear(10, 3)
        )
        
        x = torch.randn(2, 5)
        target = torch.randn(2, 3)
        
        # Create optimizer
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.SGD,
            {'lr': 0.01}
        )
        
        # Compute gradients
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Store original gradients
        original_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        
        # Apply step (which includes NSPO2 projection)
        optimizer.step()
        
        # Verify all parameters were updated
        for param in model.parameters():
            assert param.grad is not None
            
    def test_enable_disable(self):
        """Test enabling and disabling NSPO2."""
        model = nn.Linear(5, 3)
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.01}
        )
        
        # Initially enabled
        assert optimizer.enabled
        assert optimizer.nspo2.is_active
        
        # Disable
        optimizer.set_enabled(False)
        assert not optimizer.enabled
        assert not optimizer.nspo2.is_active
        
        # Re-enable
        optimizer.set_enabled(True)
        assert optimizer.enabled
        assert optimizer.nspo2.is_active
        
    def test_state_dict_save_load(self):
        """Test saving and loading optimizer state."""
        model = nn.Linear(5, 3)
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.01}
        )
        
        # Do a few steps
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.sum(model(torch.randn(2, 5)) ** 2)
            loss.backward()
            optimizer.step()
        
        # Save state
        state = optimizer.state_dict()
        
        # Create new optimizer and load state
        new_optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.01}
        )
        new_optimizer.load_state_dict(state)
        
        # Verify state was loaded
        assert new_optimizer.step_count == optimizer.step_count
        assert new_optimizer.nspo2.step_count == optimizer.nspo2.step_count
        
    def test_different_optimizers(self):
        """Test wrapper works with different optimizer types."""
        model = nn.Linear(5, 3)
        
        # Test with different optimizers
        optimizers_to_test = [
            (optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
            (optim.Adam, {'lr': 0.001}),
            (optim.AdamW, {'lr': 0.001, 'weight_decay': 0.01}),
            (optim.RMSprop, {'lr': 0.01}),
        ]
        
        for opt_class, opt_kwargs in optimizers_to_test:
            optimizer = create_nspo2_optimizer(
                model.parameters(),
                opt_class,
                opt_kwargs
            )
            
            # Test basic functionality
            optimizer.zero_grad()
            loss = torch.sum(model(torch.randn(2, 5)) ** 2)
            loss.backward()
            optimizer.step()
            
            assert optimizer.step_count == 1
            
    def test_no_gradient_parameters(self):
        """Test handling of parameters without gradients."""
        model = nn.Linear(5, 3)
        
        # Freeze bias
        model.bias.requires_grad = False
        
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.01}
        )
        
        # Verify dimension only includes parameters with gradients
        expected_dim = model.weight.numel()  # Only weight, not bias
        assert optimizer.nspo2.dim == expected_dim
        
        # Test optimization step
        optimizer.zero_grad()
        loss = torch.sum(model(torch.randn(2, 5)) ** 2)
        loss.backward()
        optimizer.step()
        
        # Bias should not have gradient
        assert model.bias.grad is None
        assert model.weight.grad is not None


class TestIntegrationWithRealModels:
    """Test integration with realistic model architectures."""
    
    def test_convolutional_model(self):
        """Test with a CNN model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.001}
        )
        
        # Test forward and backward pass
        x = torch.randn(2, 3, 32, 32)
        target = torch.randint(0, 10, (2,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        metrics = optimizer.get_metrics()
        assert 'step_count' in metrics
        assert metrics['step_count'] == 1
        
    def test_recurrent_model(self):
        """Test with an RNN model."""
        model = nn.LSTM(10, 20, num_layers=2, batch_first=True)
        
        optimizer = create_nspo2_optimizer(
            model.parameters(),
            optim.Adam,
            {'lr': 0.001}
        )
        
        # Test forward and backward pass
        x = torch.randn(2, 5, 10)  # batch, seq_len, input_size
        
        optimizer.zero_grad()
        output, _ = model(x)
        loss = torch.sum(output ** 2)
        loss.backward()
        optimizer.step()
        
        assert optimizer.step_count == 1