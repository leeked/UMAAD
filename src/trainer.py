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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
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
seed_everything(SEED, workers=True)

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
    # Set up the datasets
    dataset = _placeholder_dataset_(batch_size=BATCH_SIZE, num_workers=8)

    # Set up the model
    model = _placeholder_model_(
        lr=LR, 
        ratio=RATIO,
        num_heads=16,
        depth=24,
        dec_num_heads=16,
        dec_depth=8,
        pix_norm=PIX_NORM,
    )
    print(model)

    root_dir = f'model_logs/window/config-{RATIO}-{LR}-{BATCH_SIZE}-{NUM_EPOCHS}-{PIX_NORM}/SEED-{SEED}'
    
    # Set up the trainer
    trainer = Trainer(
        default_root_dir=root_dir + '/checkpoints',
        accelerator='gpu',
        devices=1,
        max_epochs=NUM_EPOCHS,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(
                monitor='validation_epoch_average',
                filename='{SEED}-{epoch:02d}-{validation_epoch_average:.4f}',
                save_top_k=1,
                mode='min',
            ),
        ],
#            gradient_clip_val=0.5,
#            gradient_clip_algorithm='norm',
        log_every_n_steps=10,
    )
    
    # Train the model
    start = time.time()
    trainer.fit(model, dataset)
    end = time.time()
    print(f'Training time in {(end - start) / 60:.0f} minutes and {(end - start) % 60:.0f} seconds')

    if model.global_rank != 0:
        sys.exit(0)

    best_model_path = trainer.checkpoint_callback.best_model_path
    
    print(f"best_model_path: {best_model_path}")

    # Test the model
    model = _placeholder_model_.load_from_checkpoint(best_model_path).cuda()
    model.eval()
    model.freeze()

    print(f'Eval on Test Data')
    test_dir = './data/.../test/'
    with torch.no_grad():
        for scan in tqdm(os.listdir(test_dir)):
            pass

if __name__ == "__main__":
    main()
