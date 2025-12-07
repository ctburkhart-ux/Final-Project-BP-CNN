#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
simulate_ber.py - BER Simulation for BP and BP-CNN Decoders

This module implements the full BP-CNN decoding simulation as described in the
reference paper. It compares standard BP decoding against the BP-CNN approach.

Decoder Structures:
    1. BP-only: 50 iterations of belief propagation
    2. BP-CNN (25-CNN-25): 
       - 25 BP iterations
       - CNN noise estimation and correction
       - 25 more BP iterations

The simulation generates BER vs SNR curves to demonstrate the performance
advantage of BP-CNN for channels with correlated noise.

Pipeline:
    1. Generate LDPC-encoded data with correlated noise
    2. Run BP-only decoding
    3. Run BP-CNN decoding  
    4. Compare BER and FER metrics
    5. Generate comparison plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional
import json
import time

from config import (
    device,
    TEST_DATA_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    RESULTS_DIR,
    BLOCK_LEN,
    CODE_RATE,
    ETA,
    TEST_SNR_LIST_DB,
    BP_MAX_ITER,
    BP_CNN_ITER_SPLIT,
    H_MATRIX_PATH,
)
from cnn_model import BPCNN1D

# Import LDPC codec
try:
    from ldpc_codec import LDPCCode, compute_llr_awgn
    LDPC_AVAILABLE = True
except ImportError:
    LDPC_AVAILABLE = False
    print("Warning: LDPC codec not available.")


def load_test_dataset(path: str) -> Dict:
    """Load test dataset from saved .pt file."""
    data = torch.load(path)
    return data


def bpsk_hard_decision(y: np.ndarray) -> np.ndarray:
    """
    Hard decision BPSK detection.
    For BPSK with 0->+1, 1->-1: if y <= 0, decide bit=1
    """
    return (y <= 0).astype(np.int32)


def compute_ber(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Compute Bit Error Rate (BER) = number of bit errors / total bits."""
    return np.mean(x_hat != x_true)


def compute_fer(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Compute Frame Error Rate (FER) = frames with any error / total frames."""
    if x_hat.ndim == 1:
        return float(np.any(x_hat != x_true))
    frame_errors = np.any(x_hat != x_true, axis=1)
    return np.mean(frame_errors)


class BPCNNDecoder:
    """
    BP-CNN Iterative Decoder implementing the 25-CNN-25 structure.
    
    Decoding steps:
        1. Compute initial LLRs from received signal
        2. Run 25 BP iterations
        3. Extract noise estimate: n_hat = y - s_hat (where s_hat from BP output)
        4. Apply CNN: n_tilde = CNN(n_hat)
        5. Update signal: y_corrected = y - n_tilde
        6. Compute new LLRs and run 25 more BP iterations
    """
    
    def __init__(self, cnn_model: BPCNN1D, ldpc_code: LDPCCode,
                 bp_iter_before: int = 25, bp_iter_after: int = 25,
                 device: torch.device = None):
        """
        Initialize BP-CNN decoder.
        
        Args:
            cnn_model: Trained CNN for noise estimation
            ldpc_code: LDPC code object
            bp_iter_before: BP iterations before CNN (default 25)
            bp_iter_after: BP iterations after CNN (default 25)
            device: PyTorch device
        """
        self.cnn = cnn_model
        self.ldpc = ldpc_code
        self.bp_before = bp_iter_before
        self.bp_after = bp_iter_after
        self.device = device if device is not None else torch.device('cpu')
        
        self.cnn.to(self.device)
        self.cnn.eval()
    
    def decode_bp_only(self, y: np.ndarray, noise_var: float,
                       max_iter: int = 50) -> Tuple[np.ndarray, int, float]:
        """
        Standard BP decoding without CNN assistance.
        
        Args:
            y: Received signal [batch, n] or [n]
            noise_var: Noise variance for LLR computation
            max_iter: Maximum BP iterations
        
        Returns:
            decoded: Decoded codeword bits
            iterations: Number of iterations used
            decode_time: Decoding time in seconds
        """
        start_time = time.time()
        
        # Compute LLRs: LLR = 2*y/σ² for AWGN with BPSK
        llr = compute_llr_awgn(y, noise_var)
        decoded, _, iterations = self.ldpc.decode_bp(llr, max_iter)
        
        decode_time = time.time() - start_time
        return decoded, iterations, decode_time
    
    def decode_bp_cnn(self, y: np.ndarray, noise_var: float,
                      num_cnn_iterations: int = 1) -> Tuple[np.ndarray, Dict]:
        """
        BP-CNN iterative decoding (25-CNN-25 structure).
        
        Args:
            y: Received signal [batch, n]
            noise_var: Noise variance
            num_cnn_iterations: Number of CNN applications
        
        Returns:
            decoded: Decoded codeword bits
            info: Dictionary with debug information
        """
        start_time = time.time()
        
        single = y.ndim == 1
        if single:
            y = y.reshape(1, -1)
        
        batch_size = y.shape[0]
        y_current = y.copy()
        
        info = {
            'noise_estimates': [],
            'iterations_phase1': [],
            'iterations_phase2': [],
        }
        
        for cnn_iter in range(num_cnn_iterations):
            # Phase 1: First 25 BP iterations
            llr_1 = compute_llr_awgn(y_current, noise_var)
            decoded_1, llr_total_1, c2v = self.ldpc.decode_bp_partial(llr_1, self.bp_before)
            
            # Estimate transmitted signal from BP hard decisions
            # BPSK: bit 0 -> +1, bit 1 -> -1
            s_hat = 1.0 - 2.0 * decoded_1.astype(np.float64)
            
            # Estimate noise: n_hat = received - estimated_signal
            n_hat = y_current - s_hat
            
            # Apply CNN to refine noise estimate
            with torch.no_grad():
                n_hat_tensor = torch.from_numpy(n_hat).float().unsqueeze(1).to(self.device)
                n_tilde = self.cnn(n_hat_tensor).squeeze(1).cpu().numpy()
            
            info['noise_estimates'].append(n_tilde.copy())
            
            # Update received signal by subtracting CNN noise estimate
            # Use original y to avoid error accumulation
            y_current = y - n_tilde
        
        # Phase 2: Final 25 BP iterations on corrected signal
        llr_2 = compute_llr_awgn(y_current, noise_var)
        decoded_final, _, iterations_2 = self.ldpc.decode_bp(llr_2, self.bp_after)
        
        info['decode_time'] = time.time() - start_time
        
        if single:
            return decoded_final[0], info
        return decoded_final, info


class SimpleCNNDenoiser:
    """Simple CNN denoising without LDPC (for baseline comparison)."""
    
    def __init__(self, model: BPCNN1D, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def denoise(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Denoise received signal using CNN directly (no BP)."""
        with torch.no_grad():
            y_tensor = torch.from_numpy(y).float().unsqueeze(1).to(self.device)
            n_pred = self.model(y_tensor).squeeze(1).cpu().numpy()
        
        y_denoised = y - n_pred
        return y_denoised, n_pred


def evaluate_snr_point(y: np.ndarray, codewords: np.ndarray, 
                       snr_db: float, ldpc_code: LDPCCode,
                       cnn_model: BPCNN1D, device: torch.device,
                       max_blocks: int = 500) -> Dict:
    """
    Evaluate BER at a single SNR point for both BP and BP-CNN.
    
    Args:
        y: Received signals [num_blocks, n]
        codewords: True codewords [num_blocks, n]
        snr_db: SNR in dB
        ldpc_code: LDPC code
        cnn_model: Trained CNN model
        device: PyTorch device
        max_blocks: Maximum blocks to evaluate
    
    Returns:
        Dictionary with BER and timing results
    """
    num_blocks = min(y.shape[0], max_blocks)
    y = y[:num_blocks]
    codewords = codewords[:num_blocks]
    
    # Compute noise variance from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0 / snr_linear
    
    # Create decoder
    decoder = BPCNNDecoder(cnn_model, ldpc_code, 
                           bp_iter_before=BP_CNN_ITER_SPLIT,
                           bp_iter_after=BP_MAX_ITER - BP_CNN_ITER_SPLIT,
                           device=device)
    
    results = {
        'snr_db': snr_db,
        'num_blocks': num_blocks,
    }
    
    # BP-only decoding (50 iterations)
    print(f"    Running BP-only (50 iter)...", end=" ", flush=True)
    start = time.time()
    decoded_bp, _, _ = decoder.decode_bp_only(y, noise_var, max_iter=BP_MAX_ITER)
    bp_time = time.time() - start
    
    ber_bp = compute_ber(decoded_bp, codewords)
    fer_bp = compute_fer(decoded_bp, codewords)
    results['ber_bp_only'] = ber_bp
    results['fer_bp_only'] = fer_bp
    results['time_bp_only'] = bp_time
    print(f"BER={ber_bp:.2e}, Time={bp_time:.1f}s")
    
    # BP-CNN decoding (25-CNN-25)
    print(f"    Running BP-CNN (25-CNN-25)...", end=" ", flush=True)
    start = time.time()
    decoded_bp_cnn, info = decoder.decode_bp_cnn(y, noise_var)
    bp_cnn_time = time.time() - start
    
    ber_bp_cnn = compute_ber(decoded_bp_cnn, codewords)
    fer_bp_cnn = compute_fer(decoded_bp_cnn, codewords)
    results['ber_bp_cnn'] = ber_bp_cnn
    results['fer_bp_cnn'] = fer_bp_cnn
    results['time_bp_cnn'] = bp_cnn_time
    print(f"BER={ber_bp_cnn:.2e}, Time={bp_cnn_time:.1f}s")
    
    return results


def run_snr_sweep(ldpc_code: LDPCCode, cnn_model: BPCNN1D,
                  snr_list: List[float], eta: float,
                  device: torch.device, num_blocks: int = 500) -> Dict:
    """Run BER evaluation across multiple SNR values."""
    from generate_data import generate_correlated_noise
    
    print("\n" + "=" * 60)
    print("SNR Sweep Evaluation")
    print("=" * 60)
    print(f"SNR values: {snr_list} dB")
    print(f"Blocks per SNR: {num_blocks}")
    print(f"Correlation eta: {eta}")
    
    results = {
        'snr_db': [],
        'ber_bp_only': [],
        'ber_bp_cnn': [],
        'fer_bp_only': [],
        'fer_bp_cnn': [],
    }
    
    for snr_db in snr_list:
        print(f"\n--- SNR = {snr_db} dB ---")
        
        # Generate fresh data
        info_bits = np.random.randint(0, 2, (num_blocks, ldpc_code.k))
        codewords = ldpc_code.encode(info_bits)
        
        # BPSK modulation
        s = 1.0 - 2.0 * codewords.astype(float)
        
        # Generate correlated noise
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_linear
        
        N = ldpc_code.n
        idx = np.arange(N)
        Sigma = eta ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma)
        
        z = np.random.randn(num_blocks, N)
        n = z @ L.T * np.sqrt(noise_var)
        
        y = s + n
        
        # Evaluate
        snr_results = evaluate_snr_point(y, codewords, snr_db, ldpc_code,
                                          cnn_model, device, num_blocks)
        
        results['snr_db'].append(snr_db)
        results['ber_bp_only'].append(snr_results['ber_bp_only'])
        results['ber_bp_cnn'].append(snr_results['ber_bp_cnn'])
        results['fer_bp_only'].append(snr_results['fer_bp_only'])
        results['fer_bp_cnn'].append(snr_results['fer_bp_cnn'])
    
    return results


def plot_ber_comparison(results: Dict, save_path: str):
    """Plot BER vs SNR comparison between BP and BP-CNN."""
    plt.figure(figsize=(10, 6))
    
    snr = results['snr_db']
    
    plt.semilogy(snr, results['ber_bp_only'], 'ro-', 
                 linewidth=2, markersize=8, label='BP Only (50 iter)')
    plt.semilogy(snr, results['ber_bp_cnn'], 'bs-', 
                 linewidth=2, markersize=8, label='BP-CNN (25-CNN-25)')
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title(f'BER vs SNR: BP vs BP-CNN (η={ETA})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=200)
    print(f"Saved: {save_path}")
    plt.close()


def evaluate_simple_detection(data: Dict, model: BPCNN1D, device: torch.device) -> Dict:
    """Evaluate simple BPSK detection with CNN denoising (no LDPC)."""
    y = data['y'].numpy()
    n_true = data['n'].numpy()
    
    if 'u' in data:
        x_true = data['u'].numpy()  # Codewords
    else:
        x_true = data['x'].numpy()  # Raw bits
    
    results = {}
    
    # Without CNN
    x_hat_raw = bpsk_hard_decision(y)
    results['ber_no_cnn'] = compute_ber(x_hat_raw, x_true)
    
    # With CNN denoising
    denoiser = SimpleCNNDenoiser(model, device)
    y_denoised, n_pred = denoiser.denoise(y)
    
    x_hat_cnn = bpsk_hard_decision(y_denoised)
    results['ber_cnn_denoise'] = compute_ber(x_hat_cnn, x_true)
    
    # Noise estimation quality
    residual = n_true - n_pred
    results['orig_noise_mse'] = float(np.mean(n_true ** 2))
    results['residual_mse'] = float(np.mean(residual ** 2))
    results['noise_reduction_db'] = float(10 * np.log10(
        results['orig_noise_mse'] / max(results['residual_mse'], 1e-10)))
    
    return results, n_pred, residual


def main():
    parser = argparse.ArgumentParser(description='BP-CNN Simulation')
    parser.add_argument('--snr_sweep', action='store_true',
                        help='Run full SNR sweep')
    parser.add_argument('--num_blocks', type=int, default=200,
                        help='Blocks per SNR point')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to trained CNN model')
    args = parser.parse_args()
    
    print("=" * 60)
    print("BP-CNN Simulation")
    print("=" * 60)
    
    # Load test data
    print("\n=== Loading Test Data ===")
    try:
        data = load_test_dataset(TEST_DATA_PATH)
        print(f"  Received signal shape: {tuple(data['y'].shape)}")
        print(f"  SNR: {data['snr_db']} dB")
        print(f"  Eta: {data['eta']}")
    except FileNotFoundError:
        print("  Test data not found. Run generate_data.py first.")
        return
    
    # Load CNN model
    print("\n=== Loading CNN Model ===")
    model = BPCNN1D(in_channels=1).to(device)
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded from: {args.model_path}")
    except FileNotFoundError:
        print(f"  Model not found. Using untrained model.")
    
    model.eval()
    
    # Simple evaluation
    print("\n=== CNN Denoising Evaluation ===")
    simple_results, n_pred, residual = evaluate_simple_detection(data, model, device)
    
    print(f"  BER without CNN:       {simple_results['ber_no_cnn']:.4e}")
    print(f"  BER with CNN denoise:  {simple_results['ber_cnn_denoise']:.4e}")
    print(f"  Noise reduction:       {simple_results['noise_reduction_db']:.2f} dB")
    
    # Full BP-CNN evaluation
    if LDPC_AVAILABLE and args.snr_sweep:
        print("\n=== Creating LDPC Code ===")
        ldpc_code = LDPCCode(n=BLOCK_LEN, rate=CODE_RATE)
        
        snr_results = run_snr_sweep(
            ldpc_code, model, TEST_SNR_LIST_DB, ETA, device,
            num_blocks=args.num_blocks
        )
        
        plot_ber_comparison(snr_results, f'{PLOTS_DIR}/ber_vs_snr_bp_cnn.png')
        
        with open(f'{RESULTS_DIR}/snr_sweep_results.json', 'w') as f:
            json.dump(snr_results, f, indent=2)
    
    with open(f'{RESULTS_DIR}/simulation_results.json', 'w') as f:
        json.dump(simple_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
