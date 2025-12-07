#Colby Beaman 731007772
#Ethan Mullins 133006142
#Carson Burkhart 232005992

"""
ldpc_codec.py - LDPC Encoder and Belief Propagation Decoder

Implements Low-Density Parity-Check (LDPC) codes:
    1. Parity Check Matrix Generation - Creates regular LDPC H matrix
    2. Systematic Encoder - Encodes info bits to codewords via GF(2) operations
    3. Belief Propagation Decoder - Iterative sum-product decoding
    4. Partial BP - For 25-CNN-25 architecture integration

BP decoder passes beliefs (LLRs) between variable and check nodes iteratively.
LLR = log(P(bit=0|y) / P(bit=1|y)) = 2*y/σ² for AWGN with BPSK.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import os


def create_regular_ldpc_H(n: int = 576, rate: float = 0.75, 
                          col_weight: int = 3, seed: int = 42) -> np.ndarray:
    """
    Create regular LDPC parity check matrix H.
    
    H defines parity constraints: H @ codeword = 0 (mod 2)
    Regular means uniform column weight and approximately uniform row weight.
    
    Args:
        n: Codeword length (columns in H)
        rate: Code rate R = k/n
        col_weight: Column weight (variable node degree)
        seed: Random seed
    
    Returns:
        H: Parity check matrix (m, n) where m = n*(1-rate)
    """
    m = int(n * (1 - rate))  # Number of parity checks
    H = np.zeros((m, n), dtype=np.int32)
    np.random.seed(seed)
    
    # Track row weights for balance
    row_weights = np.zeros(m, dtype=np.int32)
    target_row_weight = int(col_weight * n / m)
    
    # Place 1s column by column
    for j in range(n):
        available_rows = np.where(row_weights < target_row_weight + 2)[0]
        
        if len(available_rows) >= col_weight:
            weights = row_weights[available_rows]
            sorted_idx = np.argsort(weights)
            selected = available_rows[sorted_idx[:col_weight]]
        else:
            selected = np.random.choice(m, col_weight, replace=False)
        
        for i in selected:
            H[i, j] = 1
            row_weights[i] += 1
    
    return H


class LDPCCode:
    """
    LDPC Code with systematic encoder and BP decoder.
    
    Attributes:
        n: Codeword length
        k: Information bits  
        m: Parity bits (m = n - k)
        H: Parity check matrix
    """
    
    def __init__(self, n: int = 576, rate: float = 0.75, 
                 H: Optional[np.ndarray] = None):
        """
        Initialize LDPC code.
        
        Args:
            n: Codeword length (default 576 per 802.11n)
            rate: Code rate (default 3/4)
            H: Optional parity check matrix
        """
        self.n = n
        self.rate = rate
        self.m = int(n * (1 - rate))
        self.k = n - self.m
        
        if H is not None:
            self.H = H
            self.m, self.n = H.shape
            self.k = self.n - self.m
        else:
            self.H = create_regular_ldpc_H(n, rate)
        
        self._build_adjacency()  # Build neighbor lists for BP
        self._setup_encoder()    # Setup systematic encoder
        
        print(f"LDPC Code: n={self.n}, k={self.k}, m={self.m}, rate={self.k/self.n:.3f}")
    
    def _build_adjacency(self):
        """Build adjacency lists for efficient BP message passing."""
        self.check_neighbors = [np.where(self.H[i, :] == 1)[0] for i in range(self.m)]
        self.var_neighbors = [np.where(self.H[:, j] == 1)[0] for j in range(self.n)]
        
        avg_check_degree = np.mean([len(cn) for cn in self.check_neighbors])
        avg_var_degree = np.mean([len(vn) for vn in self.var_neighbors])
        print(f"  Avg check node degree: {avg_check_degree:.1f}")
        print(f"  Avg variable node degree: {avg_var_degree:.1f}")
    
    def _setup_encoder(self):
        """Setup systematic encoder using Gaussian elimination over GF(2)."""
        H_work = self.H.copy().astype(np.float64)
        m, n = self.m, self.n
        
        col_perm = list(range(n))
        
        # Gaussian elimination with pivoting
        for i in range(m):
            pivot_found = False
            for j in range(i, n):
                actual_col = col_perm[j]
                if H_work[i, actual_col] == 1:
                    col_perm[i], col_perm[j] = col_perm[j], col_perm[i]
                    pivot_found = True
                    break
            
            if not pivot_found:
                for j in range(n):
                    if H_work[i, col_perm[j]] == 1:
                        col_perm[i], col_perm[j] = col_perm[j], col_perm[i]
                        pivot_found = True
                        break
            
            if pivot_found:
                pivot_col = col_perm[i]
                for ii in range(m):
                    if ii != i and H_work[ii, pivot_col] == 1:
                        H_work[ii, :] = (H_work[ii, :] + H_work[i, :]) % 2
        
        self.col_perm = col_perm
        self.info_positions = np.array(col_perm[:self.k])
        self.parity_positions = np.array(col_perm[self.k:])
        
        self.H_info = self.H[:, self.info_positions]
        self.H_parity = self.H[:, self.parity_positions]
        
        try:
            self.H_parity_inv = self._gf2_inverse(self.H_parity)
            self.encoder_valid = True
        except Exception as e:
            print(f"  Warning: Using iterative encoding: {e}")
            self.encoder_valid = False
    
    def _gf2_inverse(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix inverse over GF(2)."""
        n = A.shape[0]
        Aug = np.hstack([A.copy().astype(np.float64), np.eye(n)])
        
        for i in range(n):
            pivot_row = None
            for j in range(i, n):
                if Aug[j, i] == 1:
                    pivot_row = j
                    break
            
            if pivot_row is None:
                raise ValueError(f"Matrix is singular at row {i}")
            
            if pivot_row != i:
                Aug[[i, pivot_row]] = Aug[[pivot_row, i]]
            
            for j in range(n):
                if j != i and Aug[j, i] == 1:
                    Aug[j, :] = (Aug[j, :] + Aug[i, :]) % 2
        
        return Aug[:, n:].astype(np.int32)
    
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode information bits to codeword.
        
        Systematic encoding: codeword = [info | parity]
        
        Args:
            info_bits: Shape (batch, k) or (k,)
        Returns:
            codewords: Shape (batch, n) or (n,)
        """
        single = info_bits.ndim == 1
        if single:
            info_bits = info_bits.reshape(1, -1)
        
        batch_size = info_bits.shape[0]
        codewords = np.zeros((batch_size, self.n), dtype=np.int32)
        
        codewords[:, self.info_positions] = info_bits
        
        if self.encoder_valid:
            # parity = H_parity_inv @ (H_info @ info) mod 2
            syndrome = (info_bits @ self.H_info.T) % 2
            parity = (syndrome @ self.H_parity_inv.T) % 2
            codewords[:, self.parity_positions] = parity
        else:
            for b in range(batch_size):
                codewords[b] = self._encode_iterative(codewords[b])
        
        if single:
            return codewords[0]
        return codewords
    
    def _encode_iterative(self, codeword: np.ndarray) -> np.ndarray:
        """Iteratively solve for parity bits when matrix inverse unavailable."""
        max_iter = 100
        for _ in range(max_iter):
            syndrome = (self.H @ codeword) % 2
            if np.sum(syndrome) == 0:
                break
            for i in range(self.m):
                if syndrome[i] == 1:
                    for j in self.parity_positions:
                        if self.H[i, j] == 1:
                            codeword[j] = 1 - codeword[j]
                            break
                    break
        return codeword
    
    def decode_bp(self, llr_channel: np.ndarray, max_iter: int = 50,
                  early_stop: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Belief Propagation (Sum-Product) decoder.
        
        Iteratively passes messages between variable and check nodes:
        1. Variable-to-check: sum of incoming messages
        2. Check-to-variable: product in tanh domain (sum-product)
        
        Args:
            llr_channel: Channel LLRs (batch, n) or (n,)
                        LLR = log(P(x=0|y) / P(x=1|y))
            max_iter: Maximum BP iterations
            early_stop: Stop if all parity checks satisfied
        
        Returns:
            hard_decision: Decoded bits
            llr_total: Final LLRs
            iterations: Iterations performed
        """
        single = llr_channel.ndim == 1
        if single:
            llr_channel = llr_channel.reshape(1, -1)
        
        batch_size = llr_channel.shape[0]
        
        # Message arrays
        var_to_check = np.zeros((batch_size, self.m, self.n))
        check_to_var = np.zeros((batch_size, self.m, self.n))
        
        # Initialize variable-to-check messages with channel LLRs
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                var_to_check[:, i, j] = llr_channel[:, j]
        
        iterations_used = max_iter
        
        for iteration in range(max_iter):
            # Check node update (sum-product in tanh domain)
            for i in range(self.m):
                neighbors = self.check_neighbors[i]
                if len(neighbors) < 2:
                    continue
                
                for j in neighbors:
                    prod = np.ones(batch_size)
                    for jj in neighbors:
                        if jj != j:
                            msg = var_to_check[:, i, jj]
                            prod *= np.tanh(np.clip(msg / 2, -10, 10))
                    
                    prod = np.clip(prod, -0.9999999, 0.9999999)
                    check_to_var[:, i, j] = 2 * np.arctanh(prod)
            
            # Variable node update (sum of incoming messages)
            for j in range(self.n):
                neighbors = self.var_neighbors[j]
                for i in neighbors:
                    msg_sum = llr_channel[:, j].copy()
                    for ii in neighbors:
                        if ii != i:
                            msg_sum += check_to_var[:, ii, j]
                    var_to_check[:, i, j] = msg_sum
            
            # Check convergence
            if early_stop:
                llr_total = llr_channel.copy()
                for j in range(self.n):
                    for i in self.var_neighbors[j]:
                        llr_total[:, j] += check_to_var[:, i, j]
                
                hard_decision = (llr_total < 0).astype(np.int32)
                
                all_valid = True
                for b in range(batch_size):
                    syndrome = (self.H @ hard_decision[b]) % 2
                    if np.any(syndrome != 0):
                        all_valid = False
                        break
                
                if all_valid:
                    iterations_used = iteration + 1
                    break
        
        # Compute final LLRs and decisions
        llr_total = llr_channel.copy()
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                llr_total[:, j] += check_to_var[:, i, j]
        
        hard_decision = (llr_total < 0).astype(np.int32)
        
        if single:
            return hard_decision[0], llr_total[0], iterations_used
        return hard_decision, llr_total, iterations_used
    
    def decode_bp_partial(self, llr_channel: np.ndarray, num_iter: int,
                          check_to_var_init: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Partial BP decoding for 25-CNN-25 structure.
        
        Runs fixed number of iterations without early stopping.
        Returns messages for continuation after CNN processing.
        
        Returns:
            hard_decision: Current hard decisions
            llr_total: Current total LLRs  
            check_to_var: Messages for continuation
        """
        single = llr_channel.ndim == 1
        if single:
            llr_channel = llr_channel.reshape(1, -1)
        
        batch_size = llr_channel.shape[0]
        
        var_to_check = np.zeros((batch_size, self.m, self.n))
        
        if check_to_var_init is not None:
            check_to_var = check_to_var_init.copy()
        else:
            check_to_var = np.zeros((batch_size, self.m, self.n))
        
        # Initialize
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                msg_sum = llr_channel[:, j].copy()
                for ii in self.var_neighbors[j]:
                    if ii != i:
                        msg_sum += check_to_var[:, ii, j]
                var_to_check[:, i, j] = msg_sum
        
        # Fixed iterations
        for iteration in range(num_iter):
            for i in range(self.m):
                neighbors = self.check_neighbors[i]
                if len(neighbors) < 2:
                    continue
                
                for j in neighbors:
                    prod = np.ones(batch_size)
                    for jj in neighbors:
                        if jj != j:
                            msg = var_to_check[:, i, jj]
                            prod *= np.tanh(np.clip(msg / 2, -10, 10))
                    
                    prod = np.clip(prod, -0.9999999, 0.9999999)
                    check_to_var[:, i, j] = 2 * np.arctanh(prod)
            
            for j in range(self.n):
                neighbors = self.var_neighbors[j]
                for i in neighbors:
                    msg_sum = llr_channel[:, j].copy()
                    for ii in neighbors:
                        if ii != i:
                            msg_sum += check_to_var[:, ii, j]
                    var_to_check[:, i, j] = msg_sum
        
        llr_total = llr_channel.copy()
        for j in range(self.n):
            for i in self.var_neighbors[j]:
                llr_total[:, j] += check_to_var[:, i, j]
        
        hard_decision = (llr_total < 0).astype(np.int32)
        
        if single:
            return hard_decision[0], llr_total[0], check_to_var
        return hard_decision, llr_total, check_to_var
    
    def check_codeword(self, codeword: np.ndarray) -> bool:
        """Check if codeword satisfies all parity checks: H @ c = 0."""
        syndrome = (self.H @ codeword) % 2
        return np.all(syndrome == 0)
    
    def get_info_bits(self, codeword: np.ndarray) -> np.ndarray:
        """Extract information bits from systematic codeword."""
        if codeword.ndim == 1:
            return codeword[self.info_positions]
        return codeword[:, self.info_positions]
    
    def save(self, path: str):
        """Save LDPC code to file."""
        np.savez(path, H=self.H, info_positions=self.info_positions,
                 parity_positions=self.parity_positions)
    
    @classmethod
    def load(cls, path: str) -> 'LDPCCode':
        """Load LDPC code from file."""
        data = np.load(path)
        code = cls(H=data['H'])
        return code


def compute_llr_awgn(y: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Compute LLR for AWGN channel with BPSK modulation.
    
    For BPSK (0->+1, 1->-1) over AWGN with variance σ²:
    LLR = log(P(x=0|y) / P(x=1|y)) = 2*y / σ²
    
    Args:
        y: Received signal
        noise_var: Noise variance σ²
    
    Returns:
        LLR values
    """
    return 2 * y / noise_var


def compute_llr_awgn_torch(y: torch.Tensor, noise_var: float) -> torch.Tensor:
    """PyTorch version of LLR computation."""
    return 2 * y / noise_var


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LDPC Code")
    print("=" * 60)
    
    code = LDPCCode(n=576, rate=0.75)
    
    print("\n--- Testing Encoder ---")
    info_bits = np.random.randint(0, 2, code.k)
    codeword = code.encode(info_bits)
    
    print(f"Info bits shape: {info_bits.shape}")
    print(f"Codeword shape: {codeword.shape}")
    print(f"Codeword valid: {code.check_codeword(codeword)}")
    
    print("\n--- Testing BP Decoder ---")
    s = 1.0 - 2.0 * codeword.astype(float)
    
    snr_db = 2.0
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1.0 / snr_linear
    noise = np.random.randn(code.n) * np.sqrt(noise_var)
    y = s + noise
    
    llr = compute_llr_awgn(y, noise_var)
    decoded, llr_total, iters = code.decode_bp(llr, max_iter=50)
    
    print(f"SNR: {snr_db} dB")
    print(f"Iterations used: {iters}")
    print(f"Decoded valid: {code.check_codeword(decoded)}")
    print(f"Bit errors: {np.sum(decoded != codeword)}")
    
    print("\n" + "=" * 60)
