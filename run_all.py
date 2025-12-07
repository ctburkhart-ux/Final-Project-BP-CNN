#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

#!/usr/bin/env python3
"""
run_all.py - Main Pipeline Runner for BP-CNN Project

This script executes the complete BP-CNN experimental pipeline:
    1. Verification - Test basic functionality of all components
    2. Data Generation - Create training and test datasets
    3. Training - Train CNN noise estimator model
    4. Simulation - Run BER simulations comparing BP vs BP-CNN

Each step can be skipped via command-line flags for debugging or
partial re-runs. Results are saved to appropriate directories.

Usage:
    python run_all.py                    # Run full pipeline
    python run_all.py --skip-train       # Skip training step
    python run_all.py --loss enhanced    # Use enhanced loss function
    python run_all.py --epochs 50        # Train for 50 epochs
"""

import subprocess
import sys
import os
import argparse
import time

# Ensure we're in the correct directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command as list of strings
        description: Human-readable description for logging
    
    Returns:
        True if command succeeded, False otherwise
    """
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    print("-" * 60)
    print(f"Elapsed time: {elapsed:.2f}s")
    
    if result.returncode != 0:
        print(f"WARNING: Step failed with return code {result.returncode}")
        return False
    
    print(f"Step completed successfully!")
    return True


def main():
    """Main function with argument parsing and pipeline execution."""
    
    parser = argparse.ArgumentParser(description='Run BP-CNN full pipeline')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data generation step')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training step')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation step')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip verification step')
    parser.add_argument('--loss', type=str, default='baseline',
                        choices=['baseline', 'enhanced'],
                        help='Loss function type for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--snr-sweep', action='store_true',
                        help='Run SNR sweep in simulation')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("BP-CNN PROJECT - FULL PIPELINE")
    print("=" * 60)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Python: {sys.executable}")
    
    # Track status of each step
    steps_status = {}
    
    # Step 1: Verification (run first to catch issues early)
    if not args.skip_verify:
        success = run_command(
            [sys.executable, 'verification.py'],
            'Running verification tests'
        )
        steps_status['verification'] = success
    
    # Step 2: Data generation
    if not args.skip_data:
        success = run_command(
            [sys.executable, 'generate_data.py'],
            'Generating training and test data'
        )
        steps_status['data_generation'] = success
        
        if not success:
            print("ERROR: Data generation failed. Cannot continue.")
            return 1
    
    # Step 3: Training
    if not args.skip_train:
        train_cmd = [
            sys.executable, 'train_CNN.py',
            '--loss', args.loss,
            '--epochs', str(args.epochs)
        ]
        success = run_command(
            train_cmd,
            f'Training CNN model (loss={args.loss}, epochs={args.epochs})'
        )
        steps_status['training'] = success
        
        if not success:
            print("WARNING: Training failed. Simulation may use untrained model.")
    
    # Step 4: Simulation
    if not args.skip_sim:
        sim_cmd = [sys.executable, 'simulate_ber.py']
        if args.snr_sweep:
            sim_cmd.append('--snr_sweep')
        
        success = run_command(
            sim_cmd,
            'Running BER simulation'
        )
        steps_status['simulation'] = success
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    for step, success in steps_status.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {step}: {status}")
    
    all_success = all(steps_status.values())
    print("-" * 60)
    print(f"Overall: {'ALL STEPS COMPLETED' if all_success else 'SOME STEPS FAILED'}")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for subdir in ['data', 'models', 'plots', 'results']:
        path = os.path.join(BASE_DIR, subdir)
        if os.path.exists(path):
            files = os.listdir(path)
            if files:
                print(f"\n  {subdir}/")
                for f in files:
                    fpath = os.path.join(path, f)
                    size = os.path.getsize(fpath)
                    print(f"    - {f} ({size:,} bytes)")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
