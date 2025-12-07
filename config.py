#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
config.py - Central Configuration for BP-CNN LDPC Decoder Project

This module contains all hyperparameters and configuration settings for the
BP-CNN project. Centralizing configuration makes it easy to:
    1. Reproduce experiments with consistent settings
    2. Modify parameters without changing code
    3. Document the experimental setup

Key parameters match the reference paper specifications:
    - LDPC: Rate 3/4, block length 576
    - CNN: {4; 9,3,3,15; 64,32,16,1} architecture
    - BP: 50 max iterations, 25-CNN-25 split
    - Channel: SNR 0-3 dB, correlation η = 0-0.9
"""

import torch
import os

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Automatically detect and configure GPU if available
# GPU acceleration is critical for training performance

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("Using device: cpu (CUDA not available)")
    print("  Tip: Install PyTorch with CUDA support for GPU acceleration")

# Convenience function to get device (for imports)
def get_device():
    """Return the configured PyTorch device."""
    return device

# Alias for backward compatibility
DEVICE = device

# ============================================================================
# LDPC CODE PARAMETERS
# ============================================================================
# Reference paper: Rate 3/4, Block length 576 (IEEE 802.11n standard)

CODE_RATE = 3/4                          # Code rate R = k/n
BLOCK_LEN = 576                          # N: codeword length (bits)
INFO_LEN = int(BLOCK_LEN * CODE_RATE)    # K: information bits = 432

# Enable LDPC encoding (set False for uncoded BPSK testing)
USE_LDPC = True

# ============================================================================
# DATA GENERATION PARAMETERS
# ============================================================================

NUM_BLOCKS_TRAIN = 20000   # Number of codeword blocks for training
NUM_BLOCKS_TEST = 5000     # Number of codeword blocks for testing

# SNR settings (in dB)
# Paper evaluates performance from 0 to 3 dB
TRAIN_SNR_DB = 1.0                               # SNR for training data
TEST_SNR_LIST_DB = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # SNRs for evaluation

# Noise correlation parameter η
# Covariance matrix: Σ_ij = η^|i-j| (AR(1) / Toeplitz structure)
# η = 0: white noise (no correlation)
# η = 0.5-0.8: moderate correlation
# η = 0.9: strong correlation (BP struggles, CNN helps most)
ETA = 0.7

# ============================================================================
# BP DECODER PARAMETERS
# ============================================================================
# Belief Propagation (sum-product) decoder settings

BP_MAX_ITER = 50           # Maximum BP iterations before termination
BP_CNN_ITER_SPLIT = 25     # BP iterations before/after CNN (25-CNN-25 structure)
NUM_CNN_BP_ITERATIONS = 1  # Number of BP-CNN iteration cycles

# ============================================================================
# CNN ARCHITECTURE
# ============================================================================
# Paper specification: {4; 9,3,3,15; 64,32,16,1}
# 4 layers with kernel sizes [9,3,3,15] and channels [64,32,16,1]

CNN_LAYERS = 4
CNN_KERNELS = [9, 3, 3, 15]      # Kernel size per layer
CNN_CHANNELS = [64, 32, 16, 1]   # Output channels per layer

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

BATCH_SIZE = 128           # Mini-batch size for training
NUM_EPOCHS = 20            # Number of training epochs
LEARNING_RATE = 1e-3       # Adam optimizer learning rate

# Enhanced loss regularization weight
# Controls strength of normality penalty (skewness + kurtosis)
# Paper uses λ = 0.1 for low η, λ = 10 for high η
LAMBDA_REGULARIZATION = 0.1

# ============================================================================
# FILE PATHS
# ============================================================================
# Directory structure for data, models, plots, and results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Specific file paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.pt")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.pt")
MODEL_PATH = os.path.join(MODELS_DIR, "bp_cnn_noise_estimator.pt")
H_MATRIX_PATH = os.path.join(DATA_DIR, "H_matrix.npy")  # Parity check matrix
