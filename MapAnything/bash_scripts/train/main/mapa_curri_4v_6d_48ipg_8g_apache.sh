#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1

# Logging Configs
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
module load cuda/12.4 nccl/2.18.3-cuda.12.1 nccl_efa/1.24.1-nccl.2.18.3-cuda.12.0 libfabric-aws/2.1.0amzn5.0 openmpi5/5.0.6

# AWS Multi-Node Configs
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288

torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=aws \
    dataset=megatrain_6d_518_many_ar_48ipg_8g dataset.num_workers=12 \
    dataset.num_views=4 \
    loss=overall_loss_weigh_pm_higher \
    model=mapanything \
    model/task=aug_training \
    model.encoder.gradient_checkpointing=true \
    train_params=lower_encoder_lr \
    train_params.epochs=100 \
    train_params.warmup_epochs=10 \
    train_params.keep_freq=40 \
    train_params.max_num_of_imgs_per_gpu=48 \
    hydra.run.dir='${root_experiments_dir}/mapanything/training/mapa_curri_4v_6d_48ipg_8g_apache'
