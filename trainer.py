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
parser.add_argument('--ratio', type=float, default=0.75, help='Ratio of missing data')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--accum_iter', type=int, default=1, help='Number of iterations to accumulate gradients')
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
ACCUM_ITER = args.accum_iter

print(f'= ' * 30)
print(f'Seed: {SEED}')
print(f'Ratio: {RATIO}')
print(f'Learning rate: {LR}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Number of epochs: {EPOCHS}')
print(f'Number of iterations to accumulate gradients: {ACCUM_ITER}')
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
    transforms = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225))
    ])
    train_dataset = mvtec.MVTecDataset('./data', object_name='pill', training=True, input_transform=transforms)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # Instantiate model
    chkpt_dir = 'src/checkpoints/mae_visualize_vit_large_ganloss.pth'
    model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    # Criterion and optimizers
    # param_groups = lrd.param_groups_lrd(model, weight_decay=0.05,
    #     no_weight_decay_list=model.no_weight_decay(),
    #     layer_decay=0.75
    # )
    param_groups = lrd.param_groups_lrd(model, weight_decay=0.05,
        layer_decay=0.75
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()

    # If CUDA, move to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    # Train the model
    accum_iter = ACCUM_ITER

    model.train(True)
    for epoch in range(EPOCHS):
        for iter_step, (samples, masks, labels) in enumerate(train_dl):
            optimizer.zero_grad()

            if iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(optimizer, iter_step / len(train_dl) + epoch)
            
            samples = samples.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss, pred, mask = model(samples, ratio=RATIO)
            
            loss_value = loss.item()

            loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False, update_grad=(iter_step + 1) % accum_iter == 0)
            if (iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            if iter_step % 10 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}] Iter [{iter_step + 1}/{len(train_dl)}] Loss: {loss_value}')

    # Save the model
    torch.save(model.state_dict(), 'src/checkpoints/ours/mae_finetune_vit_large.pth')

    # Visualize one example
    model.eval()
    with torch.no_grad():
        for iter_step, (samples, masks, labels) in enumerate(train_dl):
            samples = samples.to(device, non_blocking=True)

            loss, pred, mask = model(samples, ratio=RATIO)

            pred = model.unpatchify(pred)
            pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

            # visualize the mask
            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
            mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

            samples = torch.einsum('nchw->nhwc', samples).detach().cpu()

            print(f'pred shape: {pred.shape}')
            print(f'mask shape: {mask.shape}')
            print(f'samples shape: {samples.shape}')

            # Masked image
            masked_image = samples * (1 - mask)

            # MAE reconstructed pasted with visible patches
            pasted_image = samples * (1 - mask) + pred * mask

            # Visualize the image, mask, and prediction
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(samples[0])
            ax[0].set_title('Original')
            ax[1].imshow(masked_image[0])
            ax[1].set_title('Masked')
            ax[2].imshow(pred[0])
            ax[2].set_title('Reconstructed')
            ax[3].imshow(pasted_image[0])
            ax[3].set_title('Reconstructed + Visible')
            plt.savefig(f'sample.png')
            plt.close()

            break

if __name__ == "__main__":
    main()
