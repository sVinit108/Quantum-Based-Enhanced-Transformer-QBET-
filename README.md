# Quantum-Based-Enhanced-Transformer-QBET
Major revision submitted at Evolving Systems (Springer), 2025

Official implementation of **QBET (Quantum-Based Enhanced Transformer)**, a hybrid quantum-classical transformer architecture for natural language processing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

QBET integrates trainable quantum components into transformer architectures, achieving **93.3 perplexity** on Penn Treebank and **211.5 perplexity** on WikiText-2, outperforming both classical baselines (Transformer: 97.0 PPL, FNet: 117.7 PPL) and the quantum baseline Quixer (122.0 PPL on PTB). All experiments are conducted via classical simulation of quantum circuits using TorchQuantum, consistent with standard practice in quantum ML research.

**Key Features:**
- **Position-Aware Quantum Mixing (PAQM)**: Variational quantum circuits (RY, RZ, RX gates + CNOT entanglement) with hybrid sinusoidal and learned positional encoding
- **Sparse Quantum Attention**: Selective quantum enhancement on high-importance tokens via surrogate-gradient training
- **Fast Convergence**: Reaches best validation perplexity in 5–6 epochs vs. 30 epochs for classical baselines
- **Modular Design**: Configurable quantum components (qubits, circuit depth, attention tokens)

## Architecture
```
Input Tokens → Embedding → [Quantum Mixing → Sparse Attention → FFN] × L → Output
```

Each layer contains:
1. **PAQM (Quantum Mixing Layer)**: Projects embeddings to quantum amplitudes, applies parameterized quantum circuits with positional encoding, measures expectation values
2. **Sparse Attention**: Classical attention + optional quantum enhancement on top-K important tokens
3. **Feed-Forward Network**: Standard FFN (supports MoE with `num_experts>1`)

## Installation

### Requirements
- Python 3.11+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/sVinit108/QBET.git
cd QBET

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
torch>=2.0.0
torchtext>=0.15.0
torchquantum>=0.2.0
datasets>=2.14.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Quick Start

### Training on Penn Treebank
```bash
python run.py -m QBET -d cuda
```

### Configuration

Edit hyperparameters in `run.py`:
```python
QBET_hparams = {
    "dimension": 128,          # Embedding dimension
    "num_heads": 2,            # Attention heads
    "num_layers": 2,           # Transformer layers
    "n_qubits": 4,             # Qubits for quantum mixing (PAQM)
    "attn_qubits": 3,          # Qubits for quantum attention
    "quantum_tokens": 8,       # Top-K tokens for quantum enhancement
    "batch_size": 32,
    "lr": 0.001,               # Learning rate (cosine annealing)
    "epochs": 15,              # Max epochs (early stopping at patience=5)
}
```

### Custom Dataset

Modify `setup_training.py` to load your dataset:
```python
# Replace Penn Treebank loading
raw_dset = load_dataset("dataset_name")
```

## Results

### Penn Treebank (PTB)

| Model | Dimension | Layers | Val PPL | Epochs |
|-------|-----------|--------|---------|--------|
| LSTM | 128 | 2 | 127.1 (±3.1) | 30 |
| FNet | 128 | 2 | 117.7 (±0.8) | 30 |
| Transformer | 128 | 1 | 97.0 (±0.3) | 30 |
| Performer | 128 | 2 | 99.7 (±1.0) | 50 |
| Linformer | 128 | 2 | 93.0 (±0.8) | 50 |
| Quixer | 512, 6 qubits | cubic | 122.0 (±2.2) | 30 |
| **QBET (Ours)** | **128, 4+3 qubits** | **L=2, H=2** | **93.3 (±0.8)** | **5** |

### WikiText-2

| Model | Dimension | Layers | Val PPL | Epochs |
|-------|-----------|--------|---------|--------|
| LSTM | 128 | 2 | 308.1 (±2.1) | 30 |
| FNet | 128 | 2 | 287.0 (±0.6) | 30 |
| Transformer | 128 | 1 | 246.7 (±0.4) | 30 |
| Performer | 128 | 2 | 223.6 (±2.3) | 12 |
| Linformer | 128 | 2 | 220.6 (±1.1) | 27 |
| Quixer | 512, 6 qubits | cubic | 317.5 (±3.2) | 30 |
| **QBET (Ours)** | **128, 4+3 qubits** | **L=2, H=2** | **211.5 (±0.8)** | **6** |

All results averaged over 10 seeds. Standard deviations reported.

### Ablation Study (PTB / WikiText-2)

| Configuration | PTB PPL | Δ PTB | Wiki PPL | Δ Wiki |
|---------------|---------|-------|----------|--------|
| QBET (Full Model) | 93.3 | — | 211.5 | — |
| Q-Mix Only (PAQM) | 97.3 | +4.0 | 215.8 | +4.3 |
| Q-Imp Only | 120.6 | +27.3 | 229.4 | +17.9 |
| No Quantum Components | 122.1 | +28.8 | 267.5 | +56.0 |

PAQM is the dominant contributor. Removing it raises PTB PPL by 28.8 points; removing only quantum attention raises it by 4.0 points.

## Project Structure
```
QBET/
├── QBET/
│   ├── QBET.py              # Main model architecture
│   ├── setup_training.py    # Training loop and data loading
│   ├── baseline_models.py   # LSTM, Transformer, FNet baselines
│   └── quixer_model.py      # Quixer baseline implementation
├── run.py                   # Entry point for training
├── requirements.txt         # Python dependencies
└── README.md
```

## Key Components

### Quantum Mixing Layer (PAQM)
```python
class PositionAwareQuantumMixing(nn.Module):
    # 1. Projects embeddings → L2-normalized amplitude vector (2^n_qubits dim)
    # 2. Initializes quantum state via amplitude embedding
    # 3. Applies hybrid positional encoding (fixed sinusoidal + learned RY/RZ)
    # 4. Applies variational layers: RY → RZ → RX per qubit + CNOT entanglement
    # 5. Measures Pauli-Z expectation values → projects back to embedding space
    # 6. Gated residual connection (learnable scalar gate)
```

### Sparse Attention with Quantum Enhancement
```python
class SparseAttention(nn.Module):
    # Standard Q, K, V projections
    # Sparse pattern: local window (w=32) + global tokens (g=8)
    # Importance network selects top-K tokens for quantum enhancement
    # Shallow PQC (depth 2, 5 gates) computes quantum attention bias
    # Surrogate gradient for efficient backpropagation
```

### Surrogate Gradient Training
Classical MLP approximates quantum gradients during backpropagation, enabling 10–15× speedup over parameter-shift methods:
```python
# Forward: real quantum expectation values computed
quantum_output = quantum_circuit(q_device, input)

# Backward: surrogate provides gradient estimates (straight-through estimator)
if training:
    surrogate_grad = surrogate_network(input)
    output = quantum_output.detach() + (surrogate_grad - surrogate_grad.detach())
```

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (NVIDIA RTX 2080 or equivalent)
- RAM: 16GB
- Training time: ~14 sec/iteration on RTX A6000 (with quantum attention enabled)

**Recommended:**
- GPU: 24GB VRAM (NVIDIA RTX 3090/4090, A6000)
- RAM: 32GB+

> **Note on inference time:** With quantum attention disabled (`use_quantum_attention=False`), inference reduces to ~0.53 ms/iter — within one order of magnitude of classical baselines. The overhead arises entirely from classical simulation of quantum circuits, not from the model architecture itself.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Built with [TorchQuantum](https://github.com/mit-han-lab/torchquantum) for quantum circuit simulation
- Baseline implementations inspired by [Quixer](https://arxiv.org/abs/2406.04305)
- Trained on Penn Treebank and WikiText-2/103 datasets

For questions or issues, please open a GitHub issue or contact [vdsharma_m24@ce.vjti.ac.in]

---

**Note:** This implementation uses classical simulation of quantum circuits via TorchQuantum. QBET is positioned as a hybrid quantum-inspired architecture; all claims are limited to what is demonstrated under simulation at small scale. For deployment on actual quantum hardware, circuit compilation and noise mitigation strategies will be required.
