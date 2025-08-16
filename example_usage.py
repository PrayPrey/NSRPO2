"""
Example usage of NSPO2 module.

This script demonstrates how to use NSPO2 with PyTorch optimizers
for variance reduction in gradient-based optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from nspo2 import NSPO2, NSPO2Config, get_default_config, create_nspo2_optimizer


def create_noisy_optimization_problem(dim=20, noise_scale=2.0):
    """Create a simple optimization problem with noisy gradients.
    
    This simulates a scenario where gradients have high variance,
    which is common in reinforcement learning and preference optimization.
    """
    # True optimal point
    true_optimum = torch.randn(dim) * 0.5
    
    def loss_fn(x):
        """Quadratic loss with noise."""
        base_loss = torch.sum((x - true_optimum) ** 2)
        # Add noise to simulate high-variance gradients
        noise = torch.randn_like(x) * noise_scale
        return base_loss, noise
    
    return loss_fn, true_optimum


def train_with_nspo2(dim=20, n_steps=500, use_nspo2=True):
    """Train with or without NSPO2 for comparison.
    
    Args:
        dim: Problem dimension
        n_steps: Number of optimization steps
        use_nspo2: Whether to use NSPO2
        
    Returns:
        Dictionary with training history
    """
    # Create problem
    loss_fn, true_optimum = create_noisy_optimization_problem(dim, noise_scale=2.0)
    
    # Initialize parameters
    params = torch.randn(dim, requires_grad=True)
    
    # Create optimizer
    if use_nspo2:
        # NSPO2 configuration
        config = get_default_config(dim=dim)
        config['strategy'] = 'noise'  # Use noise-aware strategy
        config['projection_rank'] = min(5, dim // 2)  # Project out top-5 high-variance directions
        config['update_freq'] = 10  # Update eigenvectors every 10 steps
        config['trust_radius'] = 0.5  # Trust region constraint
        config['verbose'] = False
        
        # Create NSPO2-wrapped optimizer
        optimizer = create_nspo2_optimizer(
            [params],
            optim.SGD,
            {'lr': 0.01},
            config
        )
    else:
        # Standard optimizer
        optimizer = optim.SGD([params], lr=0.01)
    
    # Training history
    history = {
        'losses': [],
        'distances': [],
        'grad_norms': [],
        'grad_variances': []
    }
    
    # Training loop
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Compute loss and noisy gradient
        base_loss, noise = loss_fn(params)
        
        # Compute gradient (with noise)
        true_grad = 2 * (params - true_optimum)
        noisy_grad = true_grad + noise
        
        # Set gradient manually (simulating noisy gradient scenario)
        params.grad = noisy_grad
        
        # Record metrics before step
        with torch.no_grad():
            distance = torch.norm(params - true_optimum).item()
            grad_norm = torch.norm(noisy_grad).item()
            
            history['losses'].append(base_loss.item())
            history['distances'].append(distance)
            history['grad_norms'].append(grad_norm)
        
        # Optimization step
        optimizer.step()
        
        # Print progress
        if (step + 1) % 100 == 0:
            avg_loss = np.mean(history['losses'][-50:])
            print(f"Step {step+1}: Avg Loss = {avg_loss:.4f}, Distance = {distance:.4f}")
    
    # Get final metrics if using NSPO2
    if use_nspo2:
        metrics = optimizer.get_metrics()
        print(f"\nNSPO2 Metrics:")
        print(f"  Total steps: {metrics['step_count']}")
        if 'recent_variance_reduction' in metrics:
            print(f"  Recent variance reduction: {metrics['recent_variance_reduction']:.2%}")
    
    return history


def compare_training():
    """Compare training with and without NSPO2."""
    print("=" * 60)
    print("Training WITHOUT NSPO2 (Standard SGD)")
    print("=" * 60)
    history_baseline = train_with_nspo2(dim=20, n_steps=500, use_nspo2=False)
    
    print("\n" + "=" * 60)
    print("Training WITH NSPO2 (Variance-Reduced SGD)")
    print("=" * 60)
    history_nspo2 = train_with_nspo2(dim=20, n_steps=500, use_nspo2=True)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    axes[0].plot(history_baseline['losses'], label='Standard SGD', alpha=0.7)
    axes[0].plot(history_nspo2['losses'], label='NSPO2-SGD', alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Plot distances to optimum
    axes[1].plot(history_baseline['distances'], label='Standard SGD', alpha=0.7)
    axes[1].plot(history_nspo2['distances'], label='NSPO2-SGD', alpha=0.7)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Distance to Optimum')
    axes[1].set_title('Convergence Comparison')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('nspo2_comparison.png', dpi=150)
    print("\n[PLOT] Comparison plot saved as 'nspo2_comparison.png'")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    final_loss_baseline = np.mean(history_baseline['losses'][-50:])
    final_loss_nspo2 = np.mean(history_nspo2['losses'][-50:])
    
    final_dist_baseline = np.mean(history_baseline['distances'][-50:])
    final_dist_nspo2 = np.mean(history_nspo2['distances'][-50:])
    
    print(f"Final Loss (avg last 50 steps):")
    print(f"  Standard SGD: {final_loss_baseline:.4f}")
    print(f"  NSPO2-SGD:    {final_loss_nspo2:.4f}")
    print(f"  Improvement:  {(1 - final_loss_nspo2/final_loss_baseline)*100:.1f}%")
    
    print(f"\nFinal Distance to Optimum (avg last 50 steps):")
    print(f"  Standard SGD: {final_dist_baseline:.4f}")
    print(f"  NSPO2-SGD:    {final_dist_nspo2:.4f}")
    print(f"  Improvement:  {(1 - final_dist_nspo2/final_dist_baseline)*100:.1f}%")


def demonstrate_strategies():
    """Demonstrate different NSPO2 strategies."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING DIFFERENT NSPO2 STRATEGIES")
    print("=" * 60)
    
    strategies = ['noise', 'keep', 'hybrid']
    results = {}
    
    for strategy in strategies:
        print(f"\n[TEST] Testing strategy: {strategy}")
        
        # Create problem
        dim = 20
        loss_fn, true_optimum = create_noisy_optimization_problem(dim, noise_scale=2.0)
        params = torch.randn(dim, requires_grad=True)
        
        # NSPO2 configuration
        config = get_default_config(dim=dim)
        config['strategy'] = strategy
        config['projection_rank'] = 5
        config['update_freq'] = 10
        config['verbose'] = False
        
        # Create optimizer
        optimizer = create_nspo2_optimizer(
            [params],
            optim.SGD,
            {'lr': 0.01},
            config
        )
        
        # Train for fewer steps
        losses = []
        for step in range(200):
            optimizer.zero_grad()
            base_loss, noise = loss_fn(params)
            true_grad = 2 * (params - true_optimum)
            params.grad = true_grad + noise
            losses.append(base_loss.item())
            optimizer.step()
        
        results[strategy] = losses
        final_loss = np.mean(losses[-20:])
        print(f"  Final loss: {final_loss:.4f}")
    
    # Plot strategy comparison
    plt.figure(figsize=(10, 5))
    for strategy in strategies:
        plt.plot(results[strategy], label=f'{strategy.capitalize()} Strategy', alpha=0.7)
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('NSPO2 Strategy Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('nspo2_strategies.png', dpi=150)
    print("\n[PLOT] Strategy comparison plot saved as 'nspo2_strategies.png'")


if __name__ == "__main__":
    print("\n[DEMO] NSPO2 (Null Space Preference Optimization v2) Demo\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comparison
    compare_training()
    
    # Demonstrate different strategies
    demonstrate_strategies()
    
    print("\n[SUCCESS] Demo completed successfully!")
    print("\n[USAGE] To use NSPO2 in your project:")
    print("   1. Import: from nspo2 import create_nspo2_optimizer, get_default_config")
    print("   2. Configure: config = get_default_config(dim=your_dim)")
    print("   3. Wrap optimizer: optimizer = create_nspo2_optimizer(params, OptimClass, kwargs, config)")
    print("   4. Use normally: optimizer.step()")
    print("\n[INFO] See example_usage.py for more details!")