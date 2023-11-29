import sys

sys.path.append('..')

import argparse
import gc
import os
import shutil
import time

import matplotlib.pyplot as plt
import models_mae
import numpy as np
import torch
import util.lr_decay as lrd
import util.misc as misc
from tqdm import tqdm
import util.lr_sched as lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--ratio', type=float, default=0.25, help='Ratio of missing data')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
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
EPOCHS = args.num_epochs
LR = args.lr
RATIO = args.ratio
PIX_NORM = args.pix_norm and args.no_pix_norm

print(f'= ' * 30)
print(f'Seed: {SEED}')
print(f'Ratio: {RATIO}')
print(f'Learning rate: {LR}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Number of epochs: {EPOCHS}')
print(f'Pixel normalization: {PIX_NORM}')
print(f'= ' * 30)

# Taken from MAE colab notebook
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def main():
    # Datasets


    # Instantiate model
    chkpt_dir = 'src/checkpoints/mae_finetune_vit_large.pth'
    model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    # Criterion and optimizers
    param_groups = lrd.param_groups_lrd(model, weight_decay=0.05,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=0.75
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()

    exit()

    # If CUDA, move to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    # Train the model
    accum_iter = 1
    best_so_far = (None, inf)

    model.train(True)
    for epoch in range(EPOCHS):
        for iter_step, (samples, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            if iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, iter_step / len(dataloader) + epoch)
            
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, pred, mask = model(samples)
            
            loss_value = loss.item()

            loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False, update_grad=(iter_step + 1) % accum_iter == 0)
            if (iter_step + 1) % accum_iter == 0:
                optimizer.zero_step()

            torch.cuda.synchronize()

            if iter_step % 10 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}] Iter [{iter_step + 1}/{len(dataloader)}] Loss: {loss_value}')

    # Save the model
    torch.save(model.state_dict(), 'src/checkpoints/ours/mae_finetune_vit_large.pth')

if __name__ == "__main__":
    main()
