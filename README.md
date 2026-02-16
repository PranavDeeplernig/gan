# Alpha Gamma Model (AG-SGN)

This repository contains the Alpha Gamma Spatiotemporal Graph Network (AG-SGN) optimized for high-frequency Order Book forecasting and GEX-Market Dynamics analysis.

## Project Structure

- `core/`: Core KAN (Kolmogorov-Arnold Network) primitives.
- `tkan_dataset_builder/`:
    - `kan_gnn.py`: The AlphaGammaNet architecture.
    - `train_graph_net.py`: Vectorized training loop.
    - `optimize_dataset.py`: Script to convert sparse graphs to dense tensors (B=32, N=40 optimized).
    - `graph_feature_engineer.py`: Logic for GEX/Greeks feature extraction.
- `graph_data_output/`: Standard sparse graph datasets (converted to dense via optimize script).
- `alpha_gamma_visualizer.py`: Interactive Plotly dashboard for market intelligence.

## Setup & Training

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Optimize Dataset**:
   The provided datasets in `graph_data_output/` are in sparse format. Convert them to dense tensors for high-speed training:
   ```bash
   python tkan_dataset_builder/optimize_dataset.py
   ```

3. **Launch Training**:
   ```bash
   python tkan_dataset_builder/train_graph_net.py
   ```

4. **Visualize**:
   ```bash
   python alpha_gamma_visualizer.py --date YYYY-MM-DD
   ```

## Quick Start (GPU VM)

1. **Clone & Install**:
   ```bash
   git clone https://github.com/PranavDeeplernig/gan.git
   cd gan
   pip install -r requirements.txt
   ```

2. **Reconstruct Normalized Tensors**:
   *The large tensors are excluded from Git. Reconstruct them from shards:*
   ```bash
   python3 tkan_dataset_builder/optimize_dataset.py
   ```

3. **Train with Auto-Git**:
   ```bash
   python3 tkan_dataset_builder/train_graph_net.py
   ```

## Key Features

- **Normalization Fixed**: All features are Z-scored (Mean 0, Std 1) using `norm_stats.pth` to prevent KAN saturation.
- **Sharded Dataset**: Original 100MB+ graph files are split into smaller chunks (`part_x.pt`) to bypass GitHub file limits.
- **GPU/VRAM Acceleration**: If a CUDA GPU is found, the dataset is pre-loaded into VRAM for near-zero latency training.
- **Auto-Git Sync**: The training loop automatically commits and pushes best model weights to GitHub.

## Performance (XGBoost Benchmark)
- **Data Signal**: 56.34% Validation Accuracy (Verified).
- **Architecture**: AG-SGN (T-KAN + GNN).
- **KAN-GAT**: Physics-inspired Force Law attention using learned B-splines.
- **T-KAN**: Temporal memory based on KAN-GRU cells.
- **Dense Tensors**: 90% RAM reduction and 10x speedup over standard GNN batching.
