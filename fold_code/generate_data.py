#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
generate_data.py - Training and Test Data Generation for BP-CNN

This module generates datasets for training and evaluating the BP-CNN decoder.
It implements the full signal chain:
    1. Random information bits
    2. LDPC encoding (optional)
    3. BPSK modulation: 0 -> +1, 1 -> -1
    4. Correlated Gaussian noise with covariance Σ_ij = η^|i-j|
    5. Noisy received signal y = s + n

The correlated noise model (AR(1) / Toeplitz covariance) represents channels
where noise samples are temporally correlated, which degrades standard BP
decoder performance but can be exploited by the CNN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from config import (
    device,
    BLOCK_LEN,
    INFO_LEN,
    CODE_RATE,
    NUM_BLOCKS_TRAIN,
    NUM_BLOCKS_TEST,
    TRAIN_SNR_DB,
    TEST_SNR_LIST_DB,
    ETA,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    DATA_DIR,
    H_MATRIX_PATH,
)

# Import LDPC codec if available
try:
    from ldpc_codec import LDPCCode, compute_llr_awgn
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False
    print("Warning: LDPC codec not available. Using uncoded BPSK.")


def generate_correlated_noise(num_blocks: int, block_len: int, 
                               snr_db: float, eta: float, 
                               dev: torch.device) -> torch.Tensor:
    """
    Generate correlated Gaussian noise with Toeplitz covariance structure.
    
    The covariance matrix is: Σ_ij = η^|i-j|
    This is an AR(1) process where each noise sample is correlated with
    its neighbors. Higher η means stronger correlation.
    
    Method:
        1. Build covariance matrix Σ
        2. Compute Cholesky factorization: Σ = L @ L^T
        3. Generate white noise z ~ N(0, I)
        4. Transform: n = z @ L^T gives n ~ N(0, Σ)
        5. Scale to achieve target SNR
    
    Args:
        num_blocks: Number of noise vectors to generate (batch size)
        block_len: Length of each noise vector (N)
        snr_db: Target SNR in dB
        eta: Correlation coefficient (0 <= eta < 1)
             η=0: white noise, η→1: highly correlated
        dev: PyTorch device (CPU or CUDA)
    
    Returns:
        n: Correlated noise tensor [num_blocks, block_len]
    """
    N = block_len

    # Build Toeplitz covariance matrix: Σ_ij = η^|i-j|
    # This creates a symmetric positive definite matrix
    idx = torch.arange(N, device=dev, dtype=torch.float32)
    Sigma = eta ** torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # [N, N]

    # Cholesky factorization: Σ = L @ L^T
    # L is lower triangular, used to transform white noise to correlated
    L = torch.linalg.cholesky(Sigma)

    # Generate white Gaussian noise z ~ N(0, I)
    z = torch.randn(num_blocks, N, device=dev)  # [B, N]

    # Transform to correlated noise: n = z @ L^T
    # This gives n ~ N(0, Σ)
    n = z @ L.T

    # Scale to achieve desired SNR
    # For BPSK with s ∈ {+1, -1}, signal power = 1
    signal_power = 1.0
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_variance = signal_power / snr_linear
    
    # Scale noise to target variance
    # For AR(1) covariance, Var(n_i) = 1 for all i, so scale directly
    n = n * (noise_variance ** 0.5)

    return n


def generate_dataset_simple(num_blocks: int, block_len: int, 
                            snr_db: float, eta: float, 
                            dev: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Generate dataset without LDPC encoding (uncoded BPSK).
    
    Useful for testing CNN noise estimation in isolation.
    
    Args:
        num_blocks: Number of blocks to generate
        block_len: Bits per block
        snr_db: SNR in dB
        eta: Noise correlation parameter
        dev: PyTorch device
    
    Returns:
        y: Noisy received signal [B, N]
        s: BPSK symbols [B, N]
        n: True noise [B, N]
        x: Original bits [B, N]
    """
    # Generate random bits x ∈ {0, 1}
    x = torch.randint(0, 2, (num_blocks, block_len), device=dev)

    # BPSK modulation: 0 -> +1, 1 -> -1
    s = 1.0 - 2.0 * x.float()  # [B, N], values ±1

    # Generate correlated noise
    n = generate_correlated_noise(num_blocks, block_len, snr_db, eta, dev)

    # Noisy received signal: y = s + n
    y = s + n

    return y.cpu(), s.cpu(), n.cpu(), x.cpu()


def generate_dataset_ldpc(num_blocks: int, snr_db: float, eta: float,
                          dev: torch.device, ldpc_code: 'LDPCCode') -> Tuple[torch.Tensor, ...]:
    """
    Generate dataset with LDPC encoding.
    
    Full pipeline: random bits -> LDPC encode -> BPSK -> add correlated noise
    
    Args:
        num_blocks: Number of blocks to generate
        snr_db: SNR in dB
        eta: Noise correlation parameter
        dev: PyTorch device
        ldpc_code: LDPC code object
    
    Returns:
        y: Noisy received signal [B, N]
        s: BPSK symbols [B, N]
        n: True noise [B, N]
        u: Codeword bits [B, N]
        x: Information bits [B, K]
    """
    k = ldpc_code.k  # Information bits
    n_code = ldpc_code.n  # Codeword length
    
    # Generate random information bits
    x = np.random.randint(0, 2, (num_blocks, k))
    
    # LDPC encode: k info bits -> n codeword bits
    u = ldpc_code.encode(x)  # [B, N]
    
    # Convert to PyTorch
    u_torch = torch.from_numpy(u).to(dev)
    
    # BPSK modulation: 0 -> +1, 1 -> -1
    s = 1.0 - 2.0 * u_torch.float()
    
    # Generate correlated noise
    n = generate_correlated_noise(num_blocks, n_code, snr_db, eta, dev)
    
    # Noisy received signal
    y = s + n
    
    return y.cpu(), s.cpu(), n.cpu(), torch.from_numpy(u), torch.from_numpy(x)


def generate_ldpc_data(ldpc_code: 'LDPCCode', num_blocks: int, snr_db: float, 
                       eta: float) -> dict:
    """
    Convenience function to generate LDPC-encoded data.
    
    Used by experiment scripts for consistent data generation.
    
    Args:
        ldpc_code: LDPC code object
        num_blocks: Number of blocks
        snr_db: SNR in dB
        eta: Correlation parameter
    
    Returns:
        Dictionary with y, s, n, codewords, info_bits
    """
    k = ldpc_code.k
    n_code = ldpc_code.n
    
    # Generate info bits and encode
    info_bits = np.random.randint(0, 2, (num_blocks, k))
    codewords = ldpc_code.encode(info_bits)
    
    # BPSK modulation
    s = 1.0 - 2.0 * codewords.astype(np.float64)
    
    # Generate correlated noise (using NumPy for consistency)
    idx = np.arange(n_code)
    Sigma = eta ** np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1))
    L = np.linalg.cholesky(Sigma)
    
    z = np.random.randn(num_blocks, n_code)
    n = z @ L.T
    
    # Scale for SNR
    noise_var = 10 ** (-snr_db / 10)
    n = n * np.sqrt(noise_var)
    
    # Received signal
    y = s + n
    
    return {
        'y': y,
        's': s,
        'n': n,
        'codewords': codewords,
        'info_bits': info_bits
    }


def generate_multiple_snr_dataset(num_blocks: int, block_len: int,
                                   snr_list_db: list, eta: float,
                                   dev: torch.device) -> dict:
    """
    Generate datasets for multiple SNR values.
    
    Useful for SNR sweep testing.
    """
    datasets = {}
    for snr_db in snr_list_db:
        y, s, n, x = generate_dataset_simple(num_blocks, block_len, snr_db, eta, dev)
        datasets[snr_db] = {
            'y': y, 's': s, 'n': n, 'x': x,
            'snr_db': snr_db, 'eta': eta
        }
    return datasets


def compute_empirical_snr(s: torch.Tensor, n: torch.Tensor) -> float:
    """Compute empirical SNR from signal and noise tensors."""
    signal_power = torch.mean(s ** 2).item()
    noise_power = torch.mean(n ** 2).item()
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


def compute_noise_correlation(n: torch.Tensor, max_lag: int = 20) -> np.ndarray:
    """Compute autocorrelation of noise samples for verification."""
    n_np = n.numpy().flatten()
    n_centered = n_np - np.mean(n_np)
    var = np.var(n_np)
    
    correlations = []
    for lag in range(max_lag + 1):
        if lag == 0:
            correlations.append(1.0)
        else:
            corr = np.mean(n_centered[:-lag] * n_centered[lag:]) / var
            correlations.append(corr)
    
    return np.array(correlations)


def plot_data_diagnostics(y: torch.Tensor, s: torch.Tensor, 
                          n: torch.Tensor, eta: float, save_dir: str):
    """
    Generate diagnostic plots for data verification.
    
    Creates 4-panel figure showing:
        1. Clean vs noisy signal comparison
        2. Noise histogram vs Gaussian fit
        3. Noise autocorrelation vs theoretical η^lag
        4. Empirical noise covariance matrix
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Signal comparison (first block, first 100 samples)
    ax = axes[0, 0]
    idx = 0
    t = range(min(100, y.shape[1]))
    ax.plot(t, s[idx, :len(t)].numpy(), label='Clean BPSK (s)', linewidth=1.5)
    ax.plot(t, y[idx, :len(t)].numpy(), label='Noisy (y)', alpha=0.7, linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.set_title('Clean vs Noisy Signal (First Block)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Noise histogram
    ax = axes[0, 1]
    n_flat = n.flatten().numpy()
    ax.hist(n_flat, bins=50, density=True, alpha=0.7, label='Empirical')
    
    # Overlay Gaussian fit
    x_range = np.linspace(n_flat.min(), n_flat.max(), 100)
    std = np.std(n_flat)
    gaussian = np.exp(-0.5 * (x_range / std) ** 2) / (std * np.sqrt(2 * np.pi))
    ax.plot(x_range, gaussian, 'r-', linewidth=2, label=f'Gaussian (σ={std:.3f})')
    
    ax.set_xlabel('Noise Value')
    ax.set_ylabel('Density')
    ax.set_title('Noise Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Autocorrelation
    ax = axes[1, 0]
    max_lag = 20
    empirical_corr = compute_noise_correlation(n, max_lag)
    theoretical_corr = eta ** np.arange(max_lag + 1)
    
    lags = np.arange(max_lag + 1)
    ax.stem(lags, empirical_corr, linefmt='b-', markerfmt='bo', basefmt='k-', label='Empirical')
    ax.plot(lags, theoretical_corr, 'r--', linewidth=2, label=f'Theoretical (η={eta})')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Correlation')
    ax.set_title('Noise Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Covariance matrix
    ax = axes[1, 1]
    n_samples = min(50, n.shape[1])
    n_subset = n[:100, :n_samples].numpy()
    cov_empirical = np.cov(n_subset.T)
    
    im = ax.imshow(cov_empirical, cmap='coolwarm', aspect='auto')
    plt.colorbar(im, ax=ax, label='Covariance')
    ax.set_xlabel('Sample j')
    ax.set_ylabel('Sample i')
    ax.set_title(f'Empirical Noise Covariance (first {n_samples} samples)')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/data_diagnostics.png', dpi=200)
    print(f"Saved: {save_dir}/data_diagnostics.png")
    plt.close()


def main():
    """Main function to generate training and test datasets."""
    
    print("=" * 60)
    print("BP-CNN Data Generation")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Block length: {BLOCK_LEN}")
    print(f"  Training blocks: {NUM_BLOCKS_TRAIN}")
    print(f"  Test blocks: {NUM_BLOCKS_TEST}")
    print(f"  Training SNR: {TRAIN_SNR_DB} dB")
    print(f"  Test SNRs: {TEST_SNR_LIST_DB} dB")
    print(f"  Correlation eta: {ETA}")
    print(f"  Device: {device}")
    
    # Check LDPC availability
    use_ldpc = LDPC_AVAILABLE and BLOCK_LEN == 576
    
    if use_ldpc:
        print("\n[Using LDPC encoding]")
        ldpc_code = LDPCCode(n=BLOCK_LEN, rate=CODE_RATE)
        print(f"  Code rate: {CODE_RATE}")
        print(f"  Information bits: {ldpc_code.k}")
        print(f"  Codeword length: {ldpc_code.n}")
    else:
        print("\n[Using uncoded BPSK]")
        ldpc_code = None
    
    # Generate training data
    print("\n=== Generating Training Data ===")
    if use_ldpc and ldpc_code is not None:
        y_train, s_train, n_train, u_train, x_train = generate_dataset_ldpc(
            NUM_BLOCKS_TRAIN, TRAIN_SNR_DB, ETA, device, ldpc_code
        )
        train_data = {
            'y': y_train, 's': s_train, 'n': n_train,
            'u': u_train, 'x': x_train,
            'snr_db': TRAIN_SNR_DB, 'eta': ETA, 'ldpc_encoded': True,
        }
    else:
        y_train, s_train, n_train, x_train = generate_dataset_simple(
            NUM_BLOCKS_TRAIN, BLOCK_LEN, TRAIN_SNR_DB, ETA, device
        )
        train_data = {
            'y': y_train, 's': s_train, 'n': n_train, 'x': x_train,
            'snr_db': TRAIN_SNR_DB, 'eta': ETA, 'ldpc_encoded': False,
        }
    
    # Verify SNR
    empirical_snr = compute_empirical_snr(s_train, n_train)
    print(f"  Target SNR: {TRAIN_SNR_DB:.2f} dB")
    print(f"  Empirical SNR: {empirical_snr:.2f} dB")
    print(f"  Dataset shape: y={tuple(y_train.shape)}, n={tuple(n_train.shape)}")
    
    torch.save(train_data, TRAIN_DATA_PATH)
    print(f"  Saved to: {TRAIN_DATA_PATH}")
    
    # Generate test data
    print("\n=== Generating Test Data ===")
    if use_ldpc and ldpc_code is not None:
        y_test, s_test, n_test, u_test, x_test = generate_dataset_ldpc(
            NUM_BLOCKS_TEST, TRAIN_SNR_DB, ETA, device, ldpc_code
        )
        test_data = {
            'y': y_test, 's': s_test, 'n': n_test, 'u': u_test, 'x': x_test,
            'snr_db': TRAIN_SNR_DB, 'eta': ETA, 
            'test_snr_list_db': TEST_SNR_LIST_DB, 'ldpc_encoded': True,
        }
    else:
        y_test, s_test, n_test, x_test = generate_dataset_simple(
            NUM_BLOCKS_TEST, BLOCK_LEN, TRAIN_SNR_DB, ETA, device
        )
        test_data = {
            'y': y_test, 's': s_test, 'n': n_test, 'x': x_test,
            'snr_db': TRAIN_SNR_DB, 'eta': ETA,
            'test_snr_list_db': TEST_SNR_LIST_DB, 'ldpc_encoded': False,
        }
    
    empirical_snr = compute_empirical_snr(s_test, n_test)
    print(f"  Empirical SNR: {empirical_snr:.2f} dB")
    print(f"  Dataset shape: y={tuple(y_test.shape)}, n={tuple(n_test.shape)}")
    
    torch.save(test_data, TEST_DATA_PATH)
    print(f"  Saved to: {TEST_DATA_PATH}")
    
    # Generate diagnostic plots
    print("\n=== Generating Diagnostic Plots ===")
    plot_data_diagnostics(y_train, s_train, n_train, ETA, DATA_DIR)
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
