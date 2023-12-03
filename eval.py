# Written with some code borrowed from MAE visualization colab notebook and finetune engine

import sys

sys.path.append('..')

import argparse
import gc
import os
import shutil
import time

import matplotlib.pyplot as plt
import src.models_mae as models_mae
import mvtec_dataset as mvtec
import numpy as np
import torch
import src.util.lr_decay as lrd
import src.util.lr_sched as lr_sched
import src.util.misc as misc
from torchvision.transforms import v2
from tqdm import tqdm
from src.util.misc import NativeScalerWithGradNormCount as NativeScaler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint', required=True)
args = parser.parse_args()

SEED = args.seed

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

print(f'= ' * 30)
print(f'Seed: {SEED}')
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
    # create folder to save
    if not os.path.exists(f'eval'):
        os.makedirs(f'eval')

    transforms = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)),
    ])

    mask_transforms = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224), antialias=True),
    ])

    # Load model from checkpoint
    print('Loading model from checkpoint...')
    model = models_mae.mae_vit_large_patch16()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    print('Model loaded.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('Evaluating the model...')
    print('Loading test dataset...')
    test_dataset = mvtec.MVTecDataset('./data', training=False, input_transform=transforms, mask_transform=mask_transforms)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for iter_step, (samples, masks, labels) in enumerate(test_dl):
            if labels.item() == 0:
                continue
            samples = samples.to(device, non_blocking=True) # (B, 3, 224, 224)

            loss, pred, mask = model(samples, mask_ratio=0)

            pred = model.unpatchify(pred)

            samples = samples.detach().cpu()
            pred = pred.detach().cpu()

            # Take the MSE of the reconstruction
            difference_map = torch.mean((samples - pred)**2, dim=1)

            # vis
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(samples[0].permute(1, 2, 0))
            ax[0].set_title('Original')
            ax[1].imshow(pred[0].permute(1, 2, 0))
            ax[1].set_title('Reconstruction')
            ax[2].imshow(difference_map[0])
            ax[2].set_title('Difference Map')
            plt.savefig(f'eval/iter={iter_step}.png')
            plt.close()

            if iter_step == 1:
                break
if __name__ == "__main__":
    main()
