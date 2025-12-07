#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

# run_experiments_extended.py
#
# EXTENDED BP-CNN Experiments - Balanced Version
# 
# This runs the full experimental suite with:
# - 6 eta values: 0.0, 0.3, 0.5, 0.6, 0.8, 0.9
# - 2 loss functions: baseline, enhanced
# - 10 SNR points: 0 to 6 dB
# - 2K blocks per SNR point (×2 seeds = 4K total per SNR)
# - 200K training blocks
# - 300 training epochs
# - 2 random seeds for statistical averaging
#
# Memory optimized: Processes BP decoder in batches of 500 blocks
# Estimated runtime: 10-12 hours on GPU (RTX 4060 + i9-14900HX)
# RAM usage: ~6-8 GB peak (safe for 16GB system)

import os
import sys
import json
import time
import gc
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# GPU Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    device, BLOCK_LEN, CODE_RATE, BP_MAX_ITER, BP_CNN_ITER_SPLIT,
    MODELS_DIR, PLOTS_DIR, RESULTS_DIR, DATA_DIR
)
from cnn_model import BPCNN1D, BaselineLoss, EnhancedLoss
from ldpc_codec import LDPCCode, compute_llr_awgn

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# EXTENDED EXPERIMENT CONFIGURATION (BALANCED - ~10-12 hours)
# ============================================================================

EXTENDED_CONFIG = {
    # More eta values for comprehensive analysis
    'eta_values': [0.0, 0.3, 0.5, 0.6, 0.8, 0.9],
    
    # SNR range: 0 to 6 dB (10 points for faster sweep)
    'snr_values': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
    
    # Training configuration - full 200K as requested
    'num_blocks_train': 200000,     # 200K training blocks
    'num_blocks_test': 5000,
    'batch_size': 256,              # Larger batch for GPU efficiency
    'num_epochs': 300,              # Full epochs for convergence
    'learning_rate': 1e-3,
    
    # Loss functions
    'loss_types': ['baseline', 'enhanced'],
    
    # Lambda values for enhanced loss (tuned per eta)
    'lambda_values': {
        0.0: 0.1,
        0.3: 1.0,
        0.5: 10.0,
        0.6: 10.0,
        0.8: 10.0,
        0.9: 10.0,
    },
    
    # BP-CNN configuration
    'bp_max_iter': 50,
    'bp_cnn_split': 25,
    
    # Balanced testing: 2K blocks × 2 seeds = 4K samples per SNR
    'blocks_per_snr': 2000,
    
    # 2 seeds for reasonable statistics while keeping runtime manageable
    'random_seeds': [42, 123],
    
    # Target BER (for logging purposes)
    'target_ber': 1e-4,
}


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track and estimate remaining time."""
    
    def __init__(self, total_configs: int, total_seeds: int):
        self.total_configs = total_configs
        self.total_seeds = total_seeds
        self.total_steps = total_configs * total_seeds
        self.completed_steps = 0
        self.start_time = time.time()
        self.step_times = []
    
    def step_complete(self, step_time: float):
        self.completed_steps += 1
        self.step_times.append(step_time)
    
    def get_eta(self) -> str:
        if not self.step_times:
            return "Calculating..."
        
        avg_time = np.mean(self.step_times)
        remaining_steps = self.total_steps - self.completed_steps
        remaining_seconds = avg_time * remaining_steps
        
        eta = datetime.now() + timedelta(seconds=remaining_seconds)
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        return f"{hours}h {minutes}m (ETA: {eta.strftime('%H:%M')})"
    
    def get_elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        return f"{hours}h {minutes}m"


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_training_data(num_blocks: int, block_len: int, snr_db: float, 
                           eta: float, ldpc_code: LDPCCode, 
                           seed: int = 42) -> Dict:
    """Generate LDPC-encoded training data with correlated noise."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate in batches to avoid memory issues with 200K blocks
    batch_size = 10000
    num_batches = (num_blocks + batch_size - 1) // batch_size
    
    all_y = []
    all_s = []
    all_n = []
    all_codewords = []
    all_info_bits = []
    
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0 / snr_linear
    
    # Build covariance matrix once
    N = block_len
    idx = np.arange(N)
    Sigma = eta ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma)
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_blocks)
        current_batch = batch_end - batch_start
        
        # Generate information bits and encode
        info_bits = np.random.randint(0, 2, (current_batch, ldpc_code.k))
        codewords = ldpc_code.encode(info_bits)
        
        # BPSK modulation
        s = 1.0 - 2.0 * codewords.astype(np.float32)
        
        # Generate correlated noise
        z = np.random.randn(current_batch, N).astype(np.float32)
        n = (z @ L.T * np.sqrt(noise_var)).astype(np.float32)
        
        # Received signal
        y = s + n
        
        all_y.append(y)
        all_s.append(s)
        all_n.append(n)
        all_codewords.append(codewords)
        all_info_bits.append(info_bits)
    
    return {
        'y': torch.from_numpy(np.vstack(all_y)),
        's': torch.from_numpy(np.vstack(all_s)),
        'n': torch.from_numpy(np.vstack(all_n)),
        'codewords': np.vstack(all_codewords),
        'info_bits': np.vstack(all_info_bits),
        'snr_db': snr_db,
        'eta': eta,
        'noise_var': noise_var,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_cnn(train_data: Dict, val_data: Dict, loss_type: str,
              lambda_reg: float, num_epochs: int, batch_size: int,
              lr: float, device: torch.device, verbose: bool = True) -> Tuple[BPCNN1D, Dict]:
    """Train CNN noise estimator with GPU optimization."""
    
    # Prepare data
    y_train = train_data['y'].unsqueeze(1)
    n_train = train_data['n'].unsqueeze(1)
    y_val = val_data['y'].unsqueeze(1).to(device, non_blocking=True)
    n_val = val_data['n'].unsqueeze(1).to(device, non_blocking=True)
    
    # Create model
    model = BPCNN1D(in_channels=1).to(device)
    
    # Loss function
    if loss_type == 'enhanced':
        criterion = EnhancedLoss(lambda_reg=lambda_reg)
    else:
        criterion = BaselineLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )
    
    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'mse': [],
        'epoch_time': [],
        'learning_rate': [],
    }
    
    num_batches = len(y_train) // batch_size
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(y_train))
        y_train_shuffled = y_train[perm]
        n_train_shuffled = n_train[perm]
        
        epoch_loss = 0.0
        epoch_mse = 0.0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            y_batch = y_train_shuffled[start_idx:end_idx].to(device, non_blocking=True)
            n_batch = n_train_shuffled[start_idx:end_idx].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    n_pred = model(y_batch)
                    if loss_type == 'enhanced':
                        loss, components = criterion(n_pred, n_batch)
                        mse = components['mse']
                    else:
                        loss = criterion(n_pred, n_batch)
                        mse = loss.item()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                n_pred = model(y_batch)
                if loss_type == 'enhanced':
                    loss, components = criterion(n_pred, n_batch)
                    mse = components['mse']
                else:
                    loss = criterion(n_pred, n_batch)
                    mse = loss.item()
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse
        
        epoch_loss /= num_batches
        epoch_mse /= num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    n_pred_val = model(y_val)
                    if loss_type == 'enhanced':
                        val_loss, _ = criterion(n_pred_val, n_val)
                    else:
                        val_loss = criterion(n_pred_val, n_val)
            else:
                n_pred_val = model(y_val)
                if loss_type == 'enhanced':
                    val_loss, _ = criterion(n_pred_val, n_val)
                else:
                    val_loss = criterion(n_pred_val, n_val)
            val_loss = val_loss.item()
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['mse'].append(epoch_mse)
        history['epoch_time'].append(epoch_time)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        if verbose and (epoch + 1) % 30 == 0:
            print(f"      Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {epoch_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================================
# BP AND BP-CNN DECODING - MEMORY OPTIMIZED
# ============================================================================

# Maximum blocks to process at once (keeps RAM under ~2GB)
BP_BATCH_SIZE = 500


def decode_bp_only_batched(y: np.ndarray, noise_var: float, ldpc_code: LDPCCode,
                           max_iter: int = 50) -> np.ndarray:
    """Standard BP decoding with batching for memory efficiency."""
    num_blocks = y.shape[0]
    decoded_all = []
    
    for start in range(0, num_blocks, BP_BATCH_SIZE):
        end = min(start + BP_BATCH_SIZE, num_blocks)
        y_batch = y[start:end]
        
        llr = compute_llr_awgn(y_batch, noise_var)
        decoded, _, _ = ldpc_code.decode_bp(llr, max_iter=max_iter)
        decoded_all.append(decoded)
    
    return np.vstack(decoded_all)


def decode_bp_cnn_batched(y: np.ndarray, noise_var: float, ldpc_code: LDPCCode,
                          cnn_model: BPCNN1D, device: torch.device,
                          bp_before: int = 25, bp_after: int = 25) -> np.ndarray:
    """BP-CNN decoding with batching for memory efficiency."""
    
    num_blocks = y.shape[0]
    decoded_all = []
    
    cnn_model.eval()
    
    for start in range(0, num_blocks, BP_BATCH_SIZE):
        end = min(start + BP_BATCH_SIZE, num_blocks)
        y_batch = y[start:end]
        
        # Phase 1: First BP iterations
        llr_1 = compute_llr_awgn(y_batch, noise_var)
        decoded_1, llr_total_1, c2v = ldpc_code.decode_bp_partial(llr_1, bp_before)
        
        # Estimate transmitted signal
        s_hat = 1.0 - 2.0 * decoded_1.astype(np.float64)
        
        # Estimate noise
        n_hat = y_batch - s_hat
        
        # Apply CNN
        with torch.no_grad():
            n_hat_tensor = torch.from_numpy(n_hat).float().unsqueeze(1).to(device)
            n_tilde = cnn_model(n_hat_tensor).squeeze(1).cpu().numpy()
        
        # Update received signal
        y_corrected = y_batch - n_tilde
        
        # Phase 2: Final BP iterations
        llr_2 = compute_llr_awgn(y_corrected, noise_var)
        decoded_final, _, _ = ldpc_code.decode_bp(llr_2, max_iter=bp_after)
        
        decoded_all.append(decoded_final)
        
        # Free memory
        del llr_1, decoded_1, llr_total_1, c2v, s_hat, n_hat
        del n_hat_tensor, n_tilde, y_corrected, llr_2, decoded_final
    
    return np.vstack(decoded_all)


def evaluate_ber_fer(y: np.ndarray, codewords: np.ndarray, noise_var: float,
                     ldpc_code: LDPCCode, cnn_model: BPCNN1D,
                     device: torch.device, config: Dict) -> Dict:
    """Evaluate BER and FER for BP and BP-CNN decoders."""
    
    num_blocks = y.shape[0]
    n_bits = num_blocks * ldpc_code.n
    
    results = {
        'num_blocks': num_blocks,
        'num_bits': n_bits,
    }
    
    # BP-only decoding (batched for memory)
    decoded_bp = decode_bp_only_batched(y, noise_var, ldpc_code, config['bp_max_iter'])
    
    bit_errors_bp = np.sum(decoded_bp != codewords)
    frame_errors_bp = np.sum(np.any(decoded_bp != codewords, axis=1))
    
    results['bp_only'] = {
        'bit_errors': int(bit_errors_bp),
        'frame_errors': int(frame_errors_bp),
        'ber': float(bit_errors_bp / n_bits),
        'fer': float(frame_errors_bp / num_blocks),
    }
    
    # Free memory before next decoder
    del decoded_bp
    
    # BP-CNN decoding (batched for memory)
    decoded_bp_cnn = decode_bp_cnn_batched(y, noise_var, ldpc_code, cnn_model, device,
                                           config['bp_cnn_split'], 
                                           config['bp_max_iter'] - config['bp_cnn_split'])
    
    bit_errors_bp_cnn = np.sum(decoded_bp_cnn != codewords)
    frame_errors_bp_cnn = np.sum(np.any(decoded_bp_cnn != codewords, axis=1))
    
    results['bp_cnn'] = {
        'bit_errors': int(bit_errors_bp_cnn),
        'frame_errors': int(frame_errors_bp_cnn),
        'ber': float(bit_errors_bp_cnn / n_bits),
        'fer': float(frame_errors_bp_cnn / num_blocks),
    }
    
    return results


# ============================================================================
# SNR SWEEP WITH MULTIPLE SEEDS
# ============================================================================

def run_snr_sweep_with_seeds(ldpc_code: LDPCCode, cnn_model: BPCNN1D,
                              eta: float, snr_list: List[float],
                              blocks_per_snr: int, device: torch.device,
                              config: Dict, seeds: List[int]) -> Dict:
    """Run BER evaluation with multiple seeds and averaging."""
    
    results = {
        'snr_db': snr_list,
        'ber_bp_mean': [],
        'ber_bp_std': [],
        'ber_bp_cnn_mean': [],
        'ber_bp_cnn_std': [],
        'fer_bp_mean': [],
        'fer_bp_std': [],
        'fer_bp_cnn_mean': [],
        'fer_bp_cnn_std': [],
        'all_runs': [],  # Store individual run data
    }
    
    for snr_db in snr_list:
        print(f"        SNR = {snr_db} dB ", end="", flush=True)
        
        ber_bp_runs = []
        ber_bp_cnn_runs = []
        fer_bp_runs = []
        fer_bp_cnn_runs = []
        
        for seed in seeds:
            # Generate test data for this SNR and seed
            test_data = generate_training_data(
                blocks_per_snr, BLOCK_LEN, snr_db, eta, ldpc_code, seed
            )
            
            # Evaluate
            snr_results = evaluate_ber_fer(
                test_data['y'].numpy(),
                test_data['codewords'],
                test_data['noise_var'],
                ldpc_code, cnn_model, device, config
            )
            
            ber_bp_runs.append(snr_results['bp_only']['ber'])
            ber_bp_cnn_runs.append(snr_results['bp_cnn']['ber'])
            fer_bp_runs.append(snr_results['bp_only']['fer'])
            fer_bp_cnn_runs.append(snr_results['bp_cnn']['fer'])
            
            # Free memory after each seed
            del test_data, snr_results
            gc.collect()
            
            print(".", end="", flush=True)
        
        # Compute statistics
        results['ber_bp_mean'].append(float(np.mean(ber_bp_runs)))
        results['ber_bp_std'].append(float(np.std(ber_bp_runs)))
        results['ber_bp_cnn_mean'].append(float(np.mean(ber_bp_cnn_runs)))
        results['ber_bp_cnn_std'].append(float(np.std(ber_bp_cnn_runs)))
        results['fer_bp_mean'].append(float(np.mean(fer_bp_runs)))
        results['fer_bp_std'].append(float(np.std(fer_bp_runs)))
        results['fer_bp_cnn_mean'].append(float(np.mean(fer_bp_cnn_runs)))
        results['fer_bp_cnn_std'].append(float(np.std(fer_bp_cnn_runs)))
        
        results['all_runs'].append({
            'snr_db': snr_db,
            'ber_bp': ber_bp_runs,
            'ber_bp_cnn': ber_bp_cnn_runs,
        })
        
        # Summary for this SNR
        print(f" BP={np.mean(ber_bp_runs):.2e}±{np.std(ber_bp_runs):.1e}, "
              f"BP-CNN={np.mean(ber_bp_cnn_runs):.2e}±{np.std(ber_bp_cnn_runs):.1e}")
    
    return results


# ============================================================================
# PLOTTING WITH ERROR BARS
# ============================================================================

def plot_ber_comparison_extended(all_results: Dict, save_path: str):
    """Plot BER vs SNR with error bars for all configurations."""
    
    eta_values = sorted([float(k) for k in all_results.keys()])
    num_etas = len(eta_values)
    
    # Create figure with subplots for each eta
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, eta in enumerate(eta_values):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        eta_key = eta
        
        for loss_type in ['baseline', 'enhanced']:
            if loss_type not in all_results[eta_key]:
                continue
            
            results = all_results[eta_key][loss_type]
            snr = results['snr_db']
            
            linestyle = '-' if loss_type == 'baseline' else '--'
            color_bp = 'red' if loss_type == 'baseline' else 'darkred'
            color_cnn = 'blue' if loss_type == 'baseline' else 'darkblue'
            
            # BP only with error bars
            ax.errorbar(snr, results['ber_bp_mean'], yerr=results['ber_bp_std'],
                       fmt=f'o{linestyle}', color=color_bp, linewidth=1.5, 
                       markersize=4, capsize=3,
                       label=f'BP ({loss_type})')
            
            # BP-CNN with error bars
            ax.errorbar(snr, results['ber_bp_cnn_mean'], yerr=results['ber_bp_cnn_std'],
                       fmt=f's{linestyle}', color=color_cnn, linewidth=1.5,
                       markersize=4, capsize=3,
                       label=f'BP-CNN ({loss_type})')
        
        ax.set_yscale('log')
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel('BER', fontsize=10)
        ax.set_title(f'η = {eta}', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([1e-5, 1])
        ax.set_xlim([min(snr)-0.5, max(snr)+0.5])
    
    # Hide unused subplots
    for idx in range(len(eta_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    plt.close()


def plot_improvement_summary(all_results: Dict, save_path: str):
    """Plot BER improvement ratio (BP/BP-CNN) for each configuration."""
    
    eta_values = sorted([float(k) for k in all_results.keys()])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for loss_idx, loss_type in enumerate(['baseline', 'enhanced']):
        ax = axes[loss_idx]
        
        for eta in eta_values:
            if loss_type not in all_results[eta]:
                continue
            
            results = all_results[eta][loss_type]
            snr = results['snr_db']
            
            # Compute improvement ratio
            improvement = [bp / max(cnn, 1e-10) 
                          for bp, cnn in zip(results['ber_bp_mean'], 
                                            results['ber_bp_cnn_mean'])]
            
            ax.plot(snr, improvement, 'o-', linewidth=2, markersize=6,
                   label=f'η = {eta}')
        
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('BER Improvement (BP / BP-CNN)', fontsize=12)
        ax.set_title(f'{loss_type.capitalize()} Loss', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 3])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_comparison_extended(all_histories: Dict, save_path: str):
    """Plot training curves for all configurations."""
    
    eta_values = sorted([float(k) for k in all_histories.keys()])
    num_etas = len(eta_values)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, eta in enumerate(eta_values):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        for loss_type in ['baseline', 'enhanced']:
            if loss_type not in all_histories[eta]:
                continue
            
            history = all_histories[eta][loss_type]
            epochs = range(1, len(history['train_loss']) + 1)
            
            linestyle = '-' if loss_type == 'baseline' else '--'
            color = 'blue' if loss_type == 'baseline' else 'green'
            
            ax.plot(epochs, history['train_loss'], 
                   linestyle=linestyle, color=color, linewidth=1.5,
                   label=f'{loss_type} (train)')
            ax.plot(epochs, history['val_loss'],
                   linestyle=linestyle, color=color, linewidth=1, alpha=0.5,
                   label=f'{loss_type} (val)')
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'η = {eta}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(eta_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def run_extended_experiments(config: Dict):
    """Run the extended experiment suite."""
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("BP-CNN EXTENDED EXPERIMENTS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    
    print(f"\nConfiguration:")
    print(f"  Eta values: {config['eta_values']}")
    print(f"  SNR values: {config['snr_values']} dB")
    print(f"  Training blocks: {config['num_blocks_train']:,}")
    print(f"  Test blocks per SNR: {config['blocks_per_snr']:,}")
    print(f"  Training epochs: {config['num_epochs']}")
    print(f"  Loss types: {config['loss_types']}")
    print(f"  Random seeds: {config['random_seeds']}")
    print(f"  Target BER: {config['target_ber']}")
    
    # Estimate runtime
    num_configs = len(config['eta_values']) * len(config['loss_types'])
    num_snr_tests = num_configs * len(config['snr_values']) * len(config['random_seeds'])
    print(f"\n  Total configurations: {num_configs}")
    print(f"  Total SNR evaluations: {num_snr_tests:,}")
    print(f"  Estimated runtime: 10-12 hours")
    
    # Create LDPC code
    print("\n" + "=" * 70)
    print("Creating LDPC Code")
    print("=" * 70)
    ldpc_code = LDPCCode(n=BLOCK_LEN, rate=CODE_RATE)
    
    # Storage
    all_results = {}
    all_histories = {}
    all_models = {}
    
    # Progress tracking
    progress = ProgressTracker(num_configs, 1)  # Seeds handled inside sweep
    
    config_num = 0
    for eta in config['eta_values']:
        print("\n" + "=" * 70)
        print(f"EXPERIMENTS FOR η = {eta}")
        print("=" * 70)
        
        all_results[eta] = {}
        all_histories[eta] = {}
        all_models[eta] = {}
        
        lambda_reg = config['lambda_values'].get(eta, 0.1)
        
        for loss_type in config['loss_types']:
            config_num += 1
            config_start = time.time()
            
            print(f"\n  [{config_num}/{num_configs}] Loss: {loss_type} (λ={lambda_reg})")
            print(f"  Remaining: {progress.get_eta()}")
            
            # Generate training data
            print(f"\n    Generating {config['num_blocks_train']:,} training blocks...")
            gen_start = time.time()
            train_data = generate_training_data(
                config['num_blocks_train'], BLOCK_LEN,
                1.0, eta, ldpc_code, seed=42
            )
            print(f"    Data generated in {(time.time()-gen_start)/60:.1f} min")
            
            # Split into train/val
            split_idx = int(0.9 * config['num_blocks_train'])
            val_data = {k: v[split_idx:] if torch.is_tensor(v) or isinstance(v, np.ndarray) else v 
                       for k, v in train_data.items()}
            train_data = {k: v[:split_idx] if torch.is_tensor(v) or isinstance(v, np.ndarray) else v 
                         for k, v in train_data.items()}
            
            print(f"    Training samples: {len(train_data['y']):,}")
            print(f"    Validation samples: {len(val_data['y']):,}")
            
            # Train CNN
            print(f"\n    Training CNN ({config['num_epochs']} epochs)...")
            train_start = time.time()
            
            model, history = train_cnn(
                train_data, val_data, loss_type, lambda_reg,
                config['num_epochs'], config['batch_size'],
                config['learning_rate'], device, verbose=True
            )
            
            train_time = time.time() - train_start
            print(f"    Training completed in {train_time/60:.1f} min")
            print(f"    Final loss: {history['train_loss'][-1]:.6f}")
            print(f"    Best val loss: {min(history['val_loss']):.6f}")
            
            all_histories[eta][loss_type] = history
            all_models[eta][loss_type] = model
            
            # Free training data memory
            del train_data, val_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save model
            model_path = f"{MODELS_DIR}/cnn_eta{eta}_{loss_type}_extended.pt"
            torch.save(model.state_dict(), model_path)
            print(f"    Saved model: {model_path}")
            
            # Run SNR sweep with multiple seeds
            print(f"\n    Running SNR sweep ({len(config['snr_values'])} points, "
                  f"{config['blocks_per_snr']:,} blocks, {len(config['random_seeds'])} seeds)...")
            sweep_start = time.time()
            
            snr_results = run_snr_sweep_with_seeds(
                ldpc_code, model, eta, config['snr_values'],
                config['blocks_per_snr'], device, config, config['random_seeds']
            )
            
            sweep_time = time.time() - sweep_start
            print(f"    SNR sweep completed in {sweep_time/60:.1f} min")
            
            all_results[eta][loss_type] = snr_results
            
            # Update progress
            config_time = time.time() - config_start
            progress.step_complete(config_time)
            
            # Save intermediate results
            intermediate_path = f"{RESULTS_DIR}/intermediate_results_{timestamp}.json"
            save_results_json(all_results, all_histories, config, intermediate_path)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    
    plot_ber_comparison_extended(all_results, f"{PLOTS_DIR}/ber_extended_{timestamp}.png")
    plot_improvement_summary(all_results, f"{PLOTS_DIR}/improvement_{timestamp}.png")
    plot_training_comparison_extended(all_histories, f"{PLOTS_DIR}/training_extended_{timestamp}.png")
    
    # Save final results
    final_path = f"{RESULTS_DIR}/extended_results_{timestamp}.json"
    save_results_json(all_results, all_histories, config, final_path)
    print(f"Saved results: {final_path}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    
    print(f"\nBest results (BP-CNN improvement over BP):")
    for eta in config['eta_values']:
        print(f"\n  η = {eta}:")
        for loss_type in config['loss_types']:
            if loss_type in all_results[eta]:
                results = all_results[eta][loss_type]
                # Find best improvement
                improvements = [bp / max(cnn, 1e-10) 
                              for bp, cnn in zip(results['ber_bp_mean'], 
                                                results['ber_bp_cnn_mean'])]
                best_idx = np.argmax(improvements)
                best_snr = results['snr_db'][best_idx]
                best_imp = improvements[best_idx]
                best_ber = results['ber_bp_cnn_mean'][best_idx]
                
                if best_imp > 1.0:
                    print(f"    {loss_type}: {best_imp:.2f}x improvement at {best_snr} dB "
                          f"(BER: {best_ber:.2e})")
                else:
                    print(f"    {loss_type}: No improvement (best ratio: {best_imp:.2f}x)")
    
    print("\n" + "=" * 70)
    print("EXTENDED EXPERIMENTS COMPLETE")
    print("=" * 70)
    
    return all_results, all_histories


def save_results_json(results: Dict, histories: Dict, config: Dict, path: str):
    """Save results to JSON file."""
    
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    save_data = {
        'config': {k: v for k, v in config.items() if not callable(v)},
        'results': convert(results),
        'histories': convert(histories),
    }
    
    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run extended BP-CNN experiments (8-12+ hours)',
    )
    
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (~30 min)')
    
    args = parser.parse_args()
    
    config = EXTENDED_CONFIG.copy()
    
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***\n")
        config['eta_values'] = [0.5, 0.8]
        config['snr_values'] = [0.0, 1.0, 2.0, 3.0, 4.0]
        config['num_blocks_train'] = 50000
        config['num_epochs'] = 50
        config['blocks_per_snr'] = 2000
        config['random_seeds'] = [42, 123]
    
    run_extended_experiments(config)


if __name__ == "__main__":
    main()
