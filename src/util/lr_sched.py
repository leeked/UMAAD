# Taken directly from https://github.com/facebookresearch/mae/tree/main

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, min_lr=1e-6, lr=1e-3, epochs=50):
    """Decay the learning rate with half-cycle cosine after warmup"""
    lr = min_lr + (lr - min_lr) * 0.5 * \
        (1. + math.cos(math.pi * (epoch) / (epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr