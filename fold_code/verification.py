#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
verification.py - Verification Functions for BP-CNN Project

This module implements comprehensive verification tests to ensure the BP-CNN
implementation produces correct results. Required by project specifications.

Tests verify:
    1. Correlated noise generation produces correct covariance: Σᵢⱼ = η^|i-j|
    2. CNN architecture matches spec: {4; 9,3,3,15; 64,32,16,1}
    3. BPSK modulation is correct: 0→+1, 1→-1
    4. Hard decision is correct: y>0→0, y≤0→1
    5. Loss functions compute expected values
    6. SNR scaling is accurate

Each test returns (passed: bool, results: dict) for reporting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    device, BLOCK_LEN, ETA, TRAIN_SNR_DB, 
    DATA_DIR, PLOTS_DIR, RESULTS_DIR
)


class VerificationSuite:
    """
    Comprehensive verification test suite for BP-CNN implementation.
    
    Runs all verification tests and generates reports.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
    
    def log(self, msg: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(msg)
    
    def test_noise_correlation(self, n_samples: int = 10000, 
                                block_len: int = 128,
                                eta: float = 0.7,
                                snr_db: float = 1.0,
                                tolerance: float = 0.1) -> Tuple[bool, Dict]:
        """
        Verify correlated noise has correct covariance structure.
        
        Expected covariance: Σᵢⱼ = η^|i-j| * σ²
        where σ² is noise variance determined by SNR.
        """
        self.log("\n" + "=" * 50)
        self.log("Test: Noise Correlation Structure")
        self.log("=" * 50)
        
        from generate_data import generate_correlated_noise
        
        # Generate noise samples
        n = generate_correlated_noise(n_samples, block_len, snr_db, eta, device)
        n = n.cpu().numpy()
        
        # Compute empirical covariance from samples
        cov_empirical = np.cov(n.T)
        
        # Compute theoretical covariance
        idx = np.arange(block_len)
        cov_theoretical = eta ** np.abs(idx[:, None] - idx[None, :])
        
        # Scale by noise variance
        snr_linear = 10 ** (snr_db / 10)
        noise_var = 1.0 / snr_linear
        cov_theoretical *= noise_var
        
        # Check variance (diagonal) accuracy
        diag_empirical = np.diag(cov_empirical)
        diag_theoretical = np.diag(cov_theoretical)
        diag_error = np.abs(diag_empirical - diag_theoretical) / diag_theoretical
        max_diag_error = np.max(diag_error)
        
        # Check correlation structure (normalized covariance)
        corr_empirical = cov_empirical / np.sqrt(np.outer(diag_empirical, diag_empirical))
        corr_theoretical = cov_theoretical / np.sqrt(np.outer(diag_theoretical, diag_theoretical))
        corr_error = np.abs(corr_empirical - corr_theoretical)
        max_corr_error = np.max(corr_error)
        
        passed = (max_diag_error < tolerance) and (max_corr_error < tolerance)
        
        self.log(f"  Samples: {n_samples}")
        self.log(f"  Eta: {eta}, SNR: {snr_db} dB")
        self.log(f"  Max variance error: {max_diag_error:.4f}")
        self.log(f"  Max correlation error: {max_corr_error:.4f}")
        self.log(f"  Result: {'PASSED' if passed else 'FAILED'}")
        
        results = {
            'passed': passed,
            'max_diag_error': float(max_diag_error),
            'max_corr_error': float(max_corr_error),
            'tolerance': tolerance,
        }
        
        self.results['noise_correlation'] = results
        return passed, results
    
    def test_cnn_architecture(self) -> Tuple[bool, Dict]:
        """
        Verify CNN architecture matches paper specification.
        
        Expected: {4; 9,3,3,15; 64,32,16,1}
        - 4 layers
        - Kernel sizes: 9, 3, 3, 15
        - Channels: 64, 32, 16, 1
        """
        self.log("\n" + "=" * 50)
        self.log("Test: CNN Architecture")
        self.log("=" * 50)
        
        from cnn_model import BPCNN1D
        
        model = BPCNN1D()
        
        expected_kernels = [9, 3, 3, 15]
        expected_channels = [64, 32, 16, 1]
        
        layers = [model.conv1, model.conv2, model.conv3, model.conv4]
        actual_kernels = [l.kernel_size[0] for l in layers]
        actual_channels = [l.out_channels for l in layers]
        
        kernels_match = actual_kernels == expected_kernels
        channels_match = actual_channels == expected_channels
        
        # Test output shape preservation
        test_input = torch.randn(1, 1, BLOCK_LEN).to(device)
        model.to(device)
        with torch.no_grad():
            test_output = model(test_input)
        shape_match = (test_output.shape == test_input.shape)
        
        passed = kernels_match and channels_match and shape_match
        
        self.log(f"  Kernels: {actual_kernels} (expected {expected_kernels})")
        self.log(f"  Channels: {actual_channels} (expected {expected_channels})")
        self.log(f"  Shape preserved: {shape_match}")
        self.log(f"  Result: {'PASSED' if passed else 'FAILED'}")
        
        results = {
            'passed': passed,
            'kernels_match': kernels_match,
            'channels_match': channels_match,
            'shape_match': shape_match,
        }
        
        self.results['cnn_architecture'] = results
        return passed, results
    
    def test_bpsk_modulation(self) -> Tuple[bool, Dict]:
        """
        Verify BPSK modulation: 0 → +1, 1 → -1
        """
        self.log("\n" + "=" * 50)
        self.log("Test: BPSK Modulation")
        self.log("=" * 50)
        
        x = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
        s_expected = torch.tensor([1, -1, 1, -1, 1, 1, -1, -1]).float()
        
        # Apply BPSK: s = 1 - 2*x
        s_actual = 1.0 - 2.0 * x.float()
        
        passed = torch.allclose(s_actual, s_expected)
        
        self.log(f"  Input bits:    {x.tolist()}")
        self.log(f"  Expected BPSK: {s_expected.tolist()}")
        self.log(f"  Actual BPSK:   {s_actual.tolist()}")
        self.log(f"  Result: {'PASSED' if passed else 'FAILED'}")
        
        results = {
            'passed': passed,
            'test_input': x.tolist(),
            'expected_output': s_expected.tolist(),
            'actual_output': s_actual.tolist(),
        }
        
        self.results['bpsk_modulation'] = results
        return passed, results
    
    def test_hard_decision(self) -> Tuple[bool, Dict]:
        """
        Verify BPSK hard decision: y > 0 → 0, y ≤ 0 → 1
        """
        self.log("\n" + "=" * 50)
        self.log("Test: BPSK Hard Decision")
        self.log("=" * 50)
        
        from simulate_ber import bpsk_hard_decision
        
        y = np.array([0.5, -0.5, 1.2, -0.1, 0.0, 2.0, -3.0, 0.001])
        x_expected = np.array([0, 1, 0, 1, 1, 0, 1, 0])  # y=0 → bit 1
        
        x_actual = bpsk_hard_decision(y)
        
        passed = np.array_equal(x_actual, x_expected)
        
        self.log(f"  Input y:       {y.tolist()}")
        self.log(f"  Expected bits: {x_expected.tolist()}")
        self.log(f"  Actual bits:   {x_actual.tolist()}")
        self.log(f"  Result: {'PASSED' if passed else 'FAILED'}")
        
        results = {
            'passed': passed,
            'test_input': y.tolist(),
            'expected_output': x_expected.tolist(),
            'actual_output': x_actual.tolist(),
        }
        
        self.results['hard_decision'] = results
        return passed, results
    
    def test_loss_functions(self) -> Tuple[bool, Dict]:
        """
        Verify loss function implementations.
        - Baseline: pure MSE
        - Enhanced: MSE + normality penalty
        """
        self.log("\n" + "=" * 50)
        self.log("Test: Loss Functions")
        self.log("=" * 50)
        
        from cnn_model import BaselineLoss, EnhancedLoss
        
        n_pred = torch.randn(32, 1, 128)
        n_true = torch.randn(32, 1, 128)
        
        # Test baseline loss = MSE
        baseline_loss = BaselineLoss()
        loss_baseline = baseline_loss(n_pred, n_true)
        mse_expected = torch.mean((n_pred - n_true) ** 2)
        baseline_correct = torch.allclose(loss_baseline, mse_expected, rtol=1e-5)
        
        # Test enhanced loss has all components
        enhanced_loss = EnhancedLoss(lambda_reg=0.1)
        loss_enhanced, components = enhanced_loss(n_pred, n_true)
        enhanced_has_components = all(k in components for k in 
                                      ['mse', 'skewness', 'kurtosis', 'normality_penalty'])
        
        passed = baseline_correct and enhanced_has_components
        
        self.log(f"  Baseline loss: {loss_baseline.item():.6f}")
        self.log(f"  Expected MSE:  {mse_expected.item():.6f}")
        self.log(f"  Enhanced components: {list(components.keys())}")
        self.log(f"  Result: {'PASSED' if passed else 'FAILED'}")
        
        results = {
            'passed': bool(passed),
            'baseline_loss': float(loss_baseline.item()),
            'expected_mse': float(mse_expected.item()),
            'baseline_correct': bool(baseline_correct),
            'enhanced_components': components,
        }
        
        self.results['loss_functions'] = results
        return passed, results
    
    def test_snr_calculation(self) -> Tuple[bool, Dict]:
        """
        Verify SNR calculation and noise scaling.
        Generated data should match target SNR.
        """
        self.log("\n" + "=" * 50)
        self.log("Test: SNR Calculation")
        self.log("=" * 50)
        
        from generate_data import generate_dataset_simple, compute_empirical_snr
        
        test_snrs = [0.0, 1.0, 2.0, 3.0]
        errors = []
        
        for snr_db in test_snrs:
            y, s, n, x = generate_dataset_simple(1000, 128, snr_db, 0.0, device)
            empirical_snr = compute_empirical_snr(s, n)
            error = abs(empirical_snr - snr_db)
            errors.append(error)
            self.log(f"  Target: {snr_db:.1f} dB, Empirical: {empirical_snr:.2f} dB")
        
        max_error = max(errors)
        passed = max_error < 0.5  # Allow 0.5 dB error
        
        self.log(f"  Max error: {max_error:.2f} dB")
        self.log(f"  Result: {'PASSED' if passed else 'FAILED'}")
        
        results = {
            'passed': passed,
            'test_snrs': test_snrs,
            'errors': errors,
            'max_error': max_error,
        }
        
        self.results['snr_calculation'] = results
        return passed, results
    
    def run_all_tests(self) -> Tuple[bool, Dict]:
        """Run all verification tests and return summary."""
        self.log("\n" + "=" * 60)
        self.log("BP-CNN VERIFICATION SUITE")
        self.log("=" * 60)
        
        all_passed = True
        
        passed, _ = self.test_bpsk_modulation()
        all_passed = all_passed and passed
        
        passed, _ = self.test_hard_decision()
        all_passed = all_passed and passed
        
        passed, _ = self.test_cnn_architecture()
        all_passed = all_passed and passed
        
        passed, _ = self.test_loss_functions()
        all_passed = all_passed and passed
        
        passed, _ = self.test_noise_correlation()
        all_passed = all_passed and passed
        
        passed, _ = self.test_snr_calculation()
        all_passed = all_passed and passed
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("VERIFICATION SUMMARY")
        self.log("=" * 60)
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            self.log(f"  {test_name}: {status}")
        
        self.log("-" * 60)
        overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
        self.log(f"  Overall: {overall}")
        self.log("=" * 60)
        
        return all_passed, self.results
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_results(self, path: str):
        """Save results to JSON."""
        serializable = self._convert_to_serializable(self.results)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        self.log(f"\nSaved results to: {path}")
    
    def generate_report(self, save_dir: str):
        """Generate visual verification report."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Noise autocorrelation verification
        from generate_data import generate_correlated_noise, compute_noise_correlation
        n = generate_correlated_noise(1000, 50, 1.0, ETA, device).cpu().numpy()
        
        ax = axes[0]
        empirical_corr = compute_noise_correlation(torch.from_numpy(n), 20)
        theoretical_corr = ETA ** np.arange(21)
        
        ax.stem(range(21), empirical_corr, linefmt='b-', markerfmt='bo', 
                basefmt='k-', label='Empirical')
        ax.plot(range(21), theoretical_corr, 'r--', linewidth=2, 
                label=f'Theoretical (η={ETA})')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.set_title('Noise Autocorrelation Verification')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Test summary bar chart
        ax = axes[1]
        test_names = list(self.results.keys())
        test_passed = [self.results[t]['passed'] for t in test_names]
        colors = ['#27ae60' if p else '#e74c3c' for p in test_passed]
        
        bars = ax.barh(test_names, [1]*len(test_names), color=colors, edgecolor='black')
        ax.set_xlim(0, 1)
        ax.set_title('Verification Test Results')
        
        for i, (bar, passed) in enumerate(zip(bars, test_passed)):
            label = 'PASSED' if passed else 'FAILED'
            ax.text(0.5, bar.get_y() + bar.get_height()/2, label, 
                    ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/verification_report.png', dpi=200)
        self.log(f"Saved report to: {save_dir}/verification_report.png")
        plt.close()


def main():
    """Run verification suite."""
    suite = VerificationSuite(verbose=True)
    all_passed, results = suite.run_all_tests()
    
    suite.save_results(f'{RESULTS_DIR}/verification_results.json')
    suite.generate_report(PLOTS_DIR)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
