#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "1 2 benchmark_512_eth3d_snpp_tav2"
    "1 4 benchmark_512_eth3d_snpp_tav2"
    "1 8 benchmark_512_eth3d_snpp_tav2"
    "1 16 benchmark_512_eth3d_snpp_tav2"
    "1 24 benchmark_512_eth3d_snpp_tav2"
    "1 32 benchmark_512_eth3d_snpp_tav2"
    "1 50 benchmark_512_eth3d_snpp_tav2"
    "1 100 benchmark_512_eth3d_snpp_tav2"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=pow3r_ba \
        amp=0 \
        model/task=calibrated_sfm \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/dense_'"${num_views}"'_view/pow3r_calibrated_sfm'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
