"""
Compute per-channel mean and std for multi-band TIFF image datasets.

Usage:
    python tools/compute_stats.py ../Floodnet/data/mixed_dataset/SAR/train/images

Output values can be pasted directly into the .cfg config file.
"""
import os
import sys
import numpy as np
import tifffile
from tqdm import tqdm


def compute_stats(data_dir):
    tif_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])

    if not tif_files:
        print(f"No .tif files found in {data_dir}")
        sys.exit(1)

    sample = tifffile.imread(tif_files[0])
    if sample.ndim == 2:
        n_channels = 1
    elif sample.ndim == 3 and sample.shape[0] < sample.shape[2]:
        n_channels = sample.shape[0]
    else:
        n_channels = sample.shape[-1] if sample.ndim == 3 else 1

    print(f"Found {len(tif_files)} TIFF files, {n_channels} channels")
    print(f"Sample dtype: {sample.dtype}, shape: {sample.shape}")

    channel_sum = np.zeros(n_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(n_channels, dtype=np.float64)
    n_pixels = 0

    for f in tqdm(tif_files, desc="Computing"):
        img = tifffile.imread(f).astype(np.float64)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.ndim == 3 and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        h, w, c = img.shape
        channel_sum += img.sum(axis=(0, 1))
        channel_sq_sum += (img ** 2).sum(axis=(0, 1))
        n_pixels += h * w

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sq_sum / n_pixels - mean ** 2)

    print(f"\nPer-channel mean: {mean}")
    print(f"Per-channel std:  {std}")
    print(f"\n--- Paste into .cfg file ---")
    print(f"img_mean = {','.join(f'{v:.6f}' for v in mean)}")
    print(f"img_std = {','.join(f'{v:.6f}' for v in std)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/compute_stats.py <image_dir>")
        print("Example: python tools/compute_stats.py ../Floodnet/data/mixed_dataset/SAR/train/images")
        sys.exit(1)
    compute_stats(sys.argv[1])
