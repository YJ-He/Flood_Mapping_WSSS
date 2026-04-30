#!/bin/bash
# Setup script for Flood_Mapping_WSSS
# Tested with Python 3.8-3.11, PyTorch >= 1.12

set -e

ENV_NAME="flood_wsss"

echo "=== Step 1: Create conda environment ==="
conda create -n ${ENV_NAME} python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "=== Step 2: Install PyTorch (adjust CUDA version as needed) ==="
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== Step 3: Install other dependencies ==="
pip install -r requirements.txt

echo "=== Step 4: Build tree_filter CUDA extension (optional, only for WSL training) ==="
echo "If you only need supervised learning (train_flood.py), skip this step."
echo "To build: cd utils/lib_tree_filter && python setup.py build develop"

echo ""
echo "=== Setup complete! ==="
echo "Activate environment: conda activate ${ENV_NAME}"
echo ""
echo "Quick start:"
echo "  1. Compute dataset stats:  python tools/compute_stats.py <train_images_dir>"
echo "  2. Edit config:            configs/config_UAVflood.cfg  (set root_dir, img_mean, img_std)"
echo "  3. Train:                  python train_flood.py -config configs/config_UAVflood.cfg"
echo "  4. Test:                   python test_flood.py -config configs/config_UAVflood.cfg -weight <ckpt_path>"
