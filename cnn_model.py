#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
cnn_model.py - 1D Convolutional Neural Network for BP-CNN Noise Estimation

This module implements the CNN component of the BP-CNN decoder for LDPC codes
over channels with correlated noise.

The CNN learns to estimate correlated noise that standard BP decoders cannot
handle effectively. The architecture follows the reference paper:
    {4; 9,3,3,15; 64,32,16,1}
    - 4 convolutional layers
    - Kernel sizes: 9, 3, 3, 15
    - Channel progression: 1 -> 64 -> 32 -> 16 -> 1
    - ReLU activation on layers 1-3, linear output on layer 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BPCNN1D(nn.Module):
    """
    1D CNN for noise estimation in BP-CNN LDPC decoding.
    
    Takes noisy signal or noise estimate as input, outputs refined noise estimate.
    Architecture from reference paper with 'same' padding to preserve signal length.
    
    Input shape:  [batch_size, 1, block_length]
    Output shape: [batch_size, 1, block_length]
    """

    def __init__(self, in_channels: int = 1):
        """
        Initialize the BP-CNN model.
        
        Args:
            in_channels: Number of input channels (default=1)
        """
        super().__init__()
        
        # Layer 1: Wide kernel (9) captures correlation patterns
        # 1 -> 64 channels, padding=4 for 'same' output size
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=9, padding=4)
        
        # Layer 2: Feature refinement, 64 -> 32 channels
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        # Layer 3: Further compression, 32 -> 16 channels
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        
        # Layer 4: Output layer, 16 -> 1 channel
        # Large kernel (15) integrates context, NO activation for noise output
        self.conv4 = nn.Conv1d(16, 1, kernel_size=15, padding=7)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform, biases to zero."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: 4 conv layers, ReLU on first 3, linear output.
        
        Args:
            x: Input [batch, 1, block_len] - noisy signal or noise estimate
        
        Returns:
            Refined noise estimate [batch, 1, block_len]
        """
        x = F.relu(self.conv1(x))  # [B, 64, L]
        x = F.relu(self.conv2(x))  # [B, 32, L]
        x = F.relu(self.conv3(x))  # [B, 16, L]
        x = self.conv4(x)          # [B, 1, L] - no activation
        return x


class BPCNN1DConfigurable(nn.Module):
    """
    Configurable BP-CNN for architecture experimentation.
    Allows custom kernel sizes and channel progressions.
    """
    
    def __init__(self, in_channels: int = 1, 
                 kernel_sizes: list = [9, 3, 3, 15],
                 channel_sizes: list = [64, 32, 16, 1]):
        """
        Args:
            in_channels: Number of input channels
            kernel_sizes: List of kernel sizes per layer
            channel_sizes: List of output channels per layer
        """
        super().__init__()
        
        assert len(kernel_sizes) == len(channel_sizes), "Kernel and channel lists must match"
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        prev_channels = in_channels
        
        for i, (k, c) in enumerate(zip(kernel_sizes, channel_sizes)):
            padding = k // 2  # 'same' padding
            self.layers.append(nn.Conv1d(prev_channels, c, kernel_size=k, padding=padding))
            prev_channels = c
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU on all except last layer
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        x = self.layers[-1](x)  # Linear output
        return x


class BaselineLoss(nn.Module):
    """
    Baseline loss: Mean Squared Error of residual noise.
    
    LOSS_A = ||r||² / N where r = n - n_tilde (prediction error)
    
    Simple and effective, but doesn't enforce Gaussian residual statistics.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, n_pred: torch.Tensor, n_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            n_pred: Predicted noise (n_tilde)
            n_true: True noise (n)
        Returns:
            MSE loss
        """
        return self.mse(n_pred, n_true)


class EnhancedLoss(nn.Module):
    """
    Enhanced loss with normality regularization.
    
    LOSS_B = ||r||² / N + λ * (S² + (1/4)*(C - 3)²)
    
    where:
        r = n - n_tilde (residual noise)
        S = skewness (should be 0 for Gaussian)
        C = kurtosis (should be 3 for Gaussian)
        λ = regularization weight
    
    Encourages residual noise to be i.i.d. Gaussian, which improves
    BP decoder performance since LLR computation assumes Gaussian noise.
    """
    
    def __init__(self, lambda_reg: float = 0.1):
        """
        Args:
            lambda_reg: Weight for normality penalty (paper suggests 0.1-10)
        """
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, n_pred: torch.Tensor, n_true: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            n_pred: Predicted noise [batch, 1, N]
            n_true: True noise [batch, 1, N]
        
        Returns:
            total_loss: Combined loss for backprop
            components: Dict with MSE, skewness, kurtosis, penalty values
        """
        # Residual noise
        r = n_true - n_pred
        
        # MSE component
        mse = torch.mean(r ** 2)
        
        # Compute moments on flattened residuals
        r_flat = r.flatten()
        r_mean = torch.mean(r_flat)
        r_std = torch.std(r_flat) + 1e-8
        r_norm = (r_flat - r_mean) / r_std
        
        # Skewness (3rd moment) - should be 0 for Gaussian
        skewness = torch.mean(r_norm ** 3)
        
        # Kurtosis (4th moment) - should be 3 for Gaussian
        kurtosis = torch.mean(r_norm ** 4)
        
        # Normality penalty
        normality_penalty = skewness ** 2 + 0.25 * (kurtosis - 3) ** 2
        
        # Total loss
        total_loss = mse + self.lambda_reg * normality_penalty
        
        components = {
            'mse': mse.item(),
            'skewness': skewness.item(),
            'kurtosis': kurtosis.item(),
            'normality_penalty': normality_penalty.item(),
            'total': total_loss.item()
        }
        
        return total_loss, components


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_shape: tuple = (1, 1, 576)):
    """Print model architecture and verify shapes."""
    print("=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("-" * 60)
    
    device = next(model.parameters()).device
    x = torch.randn(input_shape).to(device)
    y = model(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("=" * 60)


if __name__ == "__main__":
    # Test the models
    print("Testing BPCNN1D...")
    model = BPCNN1D()
    print_model_summary(model)
    
    print("\nTesting EnhancedLoss...")
    loss_fn = EnhancedLoss(lambda_reg=0.1)
    n_pred = torch.randn(32, 1, 576)
    n_true = torch.randn(32, 1, 576)
    loss, components = loss_fn(n_pred, n_true)
    print(f"Loss components: {components}")
