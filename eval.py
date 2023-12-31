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
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from src.util.misc import NativeScalerWithGradNormCount as NativeScaler

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint', required=True)
parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian blur')
args = parser.parse_args()

SEED = args.seed
SIGMA = args.sigma

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

print(f'= ' * 30)
print(f'Seed: {SEED}')
print(f'Sigma: {args.sigma}')
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
    if not os.path.exists(f'eval/{SIGMA}'):
        os.makedirs(f'eval/{SIGMA}')

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
    tpr_per_threshold = {}
    fpr_per_threshold = {}
    precision_per_threshold = {}
    recall_per_threshold = {}
    with torch.no_grad():
        for threshold in np.arange(0.0, 1.1, 0.1):
            print(f'Calculating eval metrics for threshold: {threshold}')
            tpr_per_threshold[threshold] = []
            fpr_per_threshold[threshold] = []
            precision_per_threshold[threshold] = []
            recall_per_threshold[threshold] = []
            step = 0
            for _, (samples, masks, labels) in tqdm(enumerate(test_dl), total=len(test_dl)):
                if labels.item() == 0:
                    continue
                samples = samples.to(device, non_blocking=True) # (B, 3, 224, 224)

                # Reconstruct n times to account for random masking
                maps = []
                reconstructed = []
                for i in range(10):
                    loss, pred, mask = model(samples, mask_ratio=0.75)

                    pred = model.unpatchify(pred)

                    # Calculate difference map
                    difference_map = (samples - pred)**2

                    # Remove noise with Gaussian for each channel
                    difference_map = GaussianBlur(kernel_size=7, sigma=SIGMA)(difference_map)

                    difference_map = torch.sum(difference_map, dim=1)

                    maps.append(difference_map)
                    reconstructed.append(pred)
                
                # Average out the maps and predicted reconstructions
                difference_map = torch.stack(maps).mean(dim=0)
                pred = torch.stack(reconstructed).mean(dim=0)

                samples = samples.detach().cpu()
                difference_map = difference_map.detach().cpu()
                pred = pred.detach().cpu()

                # Normalize difference_map to [0, 1]
                difference_map = (difference_map - difference_map.min()) / (difference_map.max() - difference_map.min())

                # Threshold difference_map to get binary mask
                difference_map = (difference_map > threshold).float()

                # Calculate TPR, FPR, precision, recall
                TP = (difference_map * masks).sum()
                FP = (difference_map * (1 - masks)).sum()
                FN = ((1 - difference_map) * masks).sum()
                TN = ((1 - difference_map) * (1 - masks)).sum()

                eps = 1e-8

                TPR = TP / (TP + FN + eps)
                FPR = FP / (FP + TN + eps)
                precision = TP / (TP + FP + eps)
                recall = TP / (TP + FN + eps)

                tpr_per_threshold[threshold].append(TPR.item())
                fpr_per_threshold[threshold].append(FPR.item())
                precision_per_threshold[threshold].append(precision.item())
                recall_per_threshold[threshold].append(recall.item())

                # vis every 100 steps
                if step % 100 == 0:
                    # Un-normalize the images
                    samples = samples * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    samples = samples + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    samples = torch.einsum('nchw->nhwc', samples)

                    pred = pred * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    pred = pred + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    pred = torch.einsum('nchw->nhwc', pred)

                    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                    ax[0].imshow(samples[0], cmap='gray')
                    ax[0].set_title('Original')
                    ax[1].imshow(pred[0], cmap='gray')
                    ax[1].set_title('Reconstructed')
                    ax[2].imshow(difference_map.permute(1, 2, 0), cmap='gray')
                    ax[2].set_title('Difference Map')
                    ax[3].imshow(masks[0][0], cmap='gray')
                    ax[3].set_title('Ground Truth')

                    plt.savefig(f'eval/{SIGMA}/threshold={threshold}_item={step}.png')
                    plt.close()

                step += 1

            tpr_per_threshold[threshold] = np.mean(tpr_per_threshold[threshold])
            fpr_per_threshold[threshold] = np.mean(fpr_per_threshold[threshold])
            precision_per_threshold[threshold] = np.mean(precision_per_threshold[threshold])
            recall_per_threshold[threshold] = np.mean(recall_per_threshold[threshold])

            print(f'Threshold: {threshold}, avg TPR: {tpr_per_threshold[threshold]}, avg FPR: {fpr_per_threshold[threshold]}, avg precision: {precision_per_threshold[threshold]}, avg recall: {recall_per_threshold[threshold]}')

    # Calculate AUC
    thresholds = list(tpr_per_threshold.keys())
    thresholds.sort()
    tpr = [tpr_per_threshold[threshold] for threshold in thresholds]
    fpr = [fpr_per_threshold[threshold] for threshold in thresholds]
    precision = [precision_per_threshold[threshold] for threshold in thresholds]
    recall = [recall_per_threshold[threshold] for threshold in thresholds]

    # Sort fpr and recall in ascending order, and match the corresponding tpr and precision
    fpr = np.array(fpr)
    recall = np.array(recall)
    tpr = np.array(tpr)
    precision = np.array(precision)

    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]

    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    
    roc_auc = np.trapz(tpr, fpr)
    print(f'ROC AUC: {roc_auc}')

    pr_auc = np.trapz(precision, recall)
    print(f'Precision-Recall AUC: {pr_auc}')

    # Plot ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curve (AUC={roc_auc})')

    # Include random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.savefig(f'eval/{SIGMA}/roc_curve.png')
    plt.close()

    # Plot precision-recall curve
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC={pr_auc})')
    plt.savefig(f'eval/{SIGMA}/pr_curve.png')
    plt.close()

if __name__ == "__main__":
    main()
