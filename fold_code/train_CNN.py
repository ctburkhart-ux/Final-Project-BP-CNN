#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
train_CNN.py - Training Script for BP-CNN Noise Estimator

This module trains the BPCNN1D model to predict/correct correlated noise
from noisy received signals. The trained CNN is used within the BP-CNN
decoder pipeline to improve LDPC decoding over correlated noise channels.

Training approach:
    1. Load pre-generated training data (y, n pairs)
    2. Train CNN to minimize loss between predicted and true noise
    3. Support both baseline (MSE) and enhanced (MSE + normality) losses
    4. Save best model based on validation loss

GPU optimizations:
    - Mixed precision training (FP16) for faster computation
    - cuDNN benchmark mode for optimal kernel selection
    - TF32 matmul for tensor core acceleration
    - Pinned memory for efficient CPU-GPU transfer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Enable GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True      # Auto-tune convolution algorithms
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul

from config import (
    device,
    TRAIN_DATA_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    LAMBDA_REGULARIZATION,
    BLOCK_LEN,
)
from cnn_model import BPCNN1D, BaselineLoss, EnhancedLoss, count_parameters


def load_train_dataset(path: str):
    """
    Load training dataset from saved file.
    
    Transforms data into format expected by Conv1d:
        - Input X: [num_samples, 1, block_length]
        - Target T: [num_samples, 1, block_length]
    
    Args:
        path: Path to saved .pt file
    
    Returns:
        X: Noisy signal tensor (CNN input)
        T: True noise tensor (CNN target)
        data: Full data dictionary with metadata
    """
    data = torch.load(path)
    y = data['y']  # Noisy signal [N, L]
    n = data['n']  # True noise [N, L]
    
    # Add channel dimension for Conv1d: [N, L] -> [N, 1, L]
    X = y.unsqueeze(1)
    T = n.unsqueeze(1)
    
    return X, T, data


class Trainer:
    """
    Training manager for BP-CNN models with GPU optimization.
    
    Handles:
        - Loss function selection (baseline/enhanced)
        - Mixed precision training
        - Training and validation loops
        - Best model tracking
        - History logging
    """
    
    def __init__(self, model: nn.Module, loss_type: str = 'baseline',
                 lambda_reg: float = 0.1, lr: float = 1e-3, device: torch.device = None):
        """
        Initialize trainer.
        
        Args:
            model: CNN model to train
            loss_type: 'baseline' (MSE only) or 'enhanced' (MSE + normality)
            lambda_reg: Regularization weight for enhanced loss
            lr: Learning rate for Adam optimizer
            device: PyTorch device (CPU/CUDA)
        """
        self.model = model
        self.loss_type = loss_type
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        
        # Setup loss function
        if loss_type == 'baseline':
            self.criterion = BaselineLoss()
        elif loss_type == 'enhanced':
            self.criterion = EnhancedLoss(lambda_reg=lambda_reg)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Adam optimizer - good default for CNNs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Mixed precision training (GPU only)
        # Uses FP16 for forward/backward, FP32 for weight updates
        self.use_amp = self.device.type == 'cuda' and hasattr(torch.cuda.amp, 'autocast')
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training history for plotting
        self.history = {
            'train_loss': [],
            'mse': [],
            'normality_penalty': [] if loss_type == 'enhanced' else None,
            'epoch_time': [],
        }
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            loader: Training data loader
        
        Returns:
            Dictionary with loss, mse, and optionally normality_penalty
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_norm_penalty = 0.0
        num_batches = 0
        
        for batch_x, batch_t in loader:
            # Move data to device (non_blocking for async transfer)
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_t = batch_t.to(self.device, non_blocking=True)
            
            # Zero gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(batch_x)
                    if self.loss_type == 'enhanced':
                        loss, components = self.criterion(output, batch_t)
                        total_mse += components['mse']
                        total_norm_penalty += components['normality_penalty']
                    else:
                        loss = self.criterion(output, batch_t)
                        total_mse += loss.item()
                
                # Scaled backward pass for mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                output = self.model(batch_x)
                if self.loss_type == 'enhanced':
                    loss, components = self.criterion(output, batch_t)
                    total_mse += components['mse']
                    total_norm_penalty += components['normality_penalty']
                else:
                    loss = self.criterion(output, batch_t)
                    total_mse += loss.item()
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Compute averages
        metrics = {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
        }
        
        if self.loss_type == 'enhanced':
            metrics['normality_penalty'] = total_norm_penalty / num_batches
        
        return metrics
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Validate on a dataset (no gradient computation).
        
        Args:
            loader: Validation data loader
        
        Returns:
            Dictionary with loss and mse metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_t in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_t = batch_t.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(batch_x)
                        if self.loss_type == 'enhanced':
                            loss, components = self.criterion(output, batch_t)
                            total_mse += components['mse']
                        else:
                            loss = self.criterion(output, batch_t)
                            total_mse += loss.item()
                else:
                    output = self.model(batch_x)
                    if self.loss_type == 'enhanced':
                        loss, components = self.criterion(output, batch_t)
                        total_mse += components['mse']
                    else:
                        loss = self.criterion(output, batch_t)
                        total_mse += loss.item()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
        }
    
    def train(self, train_loader: DataLoader, num_epochs: int,
              val_loader: Optional[DataLoader] = None,
              verbose: bool = True) -> Dict[str, List]:
        """
        Full training loop with early stopping based on best loss.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            val_loader: Optional validation loader
            verbose: Print progress each epoch
        
        Returns:
            Training history dictionary
        """
        best_loss = float('inf')
        best_model_state = None
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate if loader provided
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = None
            
            epoch_time = time.time() - start_time
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['mse'].append(train_metrics['mse'])
            self.history['epoch_time'].append(epoch_time)
            
            if self.loss_type == 'enhanced':
                self.history['normality_penalty'].append(train_metrics['normality_penalty'])
            
            # Track best model
            if train_metrics['loss'] < best_loss:
                best_loss = train_metrics['loss']
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            if verbose:
                msg = f"Epoch {epoch:03d}/{num_epochs:03d} | "
                msg += f"Loss: {train_metrics['loss']:.6f} | "
                msg += f"MSE: {train_metrics['mse']:.6f}"
                
                if self.loss_type == 'enhanced':
                    msg += f" | NormPenalty: {train_metrics['normality_penalty']:.6f}"
                
                if val_loader is not None:
                    msg += f" | Val Loss: {val_metrics['loss']:.6f}"
                
                msg += f" | Time: {epoch_time:.2f}s"
                print(msg)
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def save_model(self, path: str):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), path)
    
    def save_history(self, path: str):
        """Save training history as JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def plot_training_history(history: Dict[str, List], save_path: str,
                          loss_type: str = 'baseline'):
    """
    Plot training curves.
    
    Creates figure with:
        - Training loss vs epoch
        - MSE vs epoch
        - Normality penalty vs epoch (enhanced loss only)
    """
    num_plots = 2 if loss_type == 'baseline' else 3
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot training loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].grid(True, alpha=0.3)
    
    # Plot MSE
    axes[1].plot(epochs, history['mse'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE vs Epoch')
    axes[1].grid(True, alpha=0.3)
    
    # Plot normality penalty (enhanced only)
    if loss_type == 'enhanced' and history.get('normality_penalty'):
        axes[2].plot(epochs, history['normality_penalty'], 'r-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Normality Penalty')
        axes[2].set_title('Normality Penalty vs Epoch')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved training history plot to: {save_path}")
    plt.close()


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Train BP-CNN noise estimator')
    parser.add_argument('--loss', type=str, default='baseline',
                        choices=['baseline', 'enhanced'],
                        help='Loss function type')
    parser.add_argument('--lambda_reg', type=float, default=LAMBDA_REGULARIZATION,
                        help='Regularization weight for enhanced loss')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    args = parser.parse_args()
    
    print("=" * 60)
    print("BP-CNN Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Loss type: {args.loss}")
    if args.loss == 'enhanced':
        print(f"  Lambda regularization: {args.lambda_reg}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    
    # Load training data
    print("\n=== Loading Training Data ===")
    X, T, data_info = load_train_dataset(TRAIN_DATA_PATH)
    print(f"  Input shape: {tuple(X.shape)}")
    print(f"  Target shape: {tuple(T.shape)}")
    print(f"  SNR: {data_info['snr_db']} dB")
    print(f"  Eta: {data_info['eta']}")
    
    # Create data loaders with train/val split
    dataset = TensorDataset(X, T)
    
    # 90/10 train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create loaders with GPU-optimized settings
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               pin_memory=(device.type == 'cuda'), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=(device.type == 'cuda'), num_workers=0)
    
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = BPCNN1D(in_channels=1)
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_type=args.loss,
        lambda_reg=args.lambda_reg,
        lr=args.lr,
        device=device
    )
    
    # Train
    print("\n=== Training ===")
    history = trainer.train(train_loader, args.epochs, val_loader, verbose=True)
    
    # Save model
    model_path = MODEL_PATH.replace('.pt', f'_{args.loss}.pt')
    trainer.save_model(model_path)
    print(f"\nSaved model to: {model_path}")
    
    trainer.save_model(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    
    # Save history
    history_path = f"{RESULTS_DIR}/training_history_{args.loss}.json"
    trainer.save_history(history_path)
    print(f"Saved history to: {history_path}")
    
    # Plot history
    plot_path = f"{PLOTS_DIR}/training_loss_{args.loss}.png"
    plot_training_history(history, plot_path, args.loss)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
