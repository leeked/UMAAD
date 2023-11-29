import sys

sys.path.append('..')

import argparse
import os
import gc
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--ratio', type=float, default=0.25, help='Ratio of missing data')
parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--pix_norm', action='store_true', help='Pixel normalization')
parser.add_argument('--no_pix_norm', action='store_false', help='No pixel normalization')
args = parser.parse_args()

SEED = args.seed

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LR = args.lr
RATIO = args.ratio
PIX_NORM = args.pix_norm and args.no_pix_norm

print(f'= ' * 30)
print(f'Seed: {SEED}')
print(f'Ratio: {RATIO}')
print(f'Learning rate: {LR}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Number of epochs: {NUM_EPOCHS}')
print(f'Pixel normalization: {PIX_NORM}')
print(f'= ' * 30)

def main():
    # Datasets
    ...

    # Instantiate model
    ...

    # Criterion and optimizers
    ...

    # If CUDA, move to CUDA
    ...

    # Train the model
    ...

    # Save the model
    ...

    # Evaluate the model
    ...

if __name__ == "__main__":
    main()
