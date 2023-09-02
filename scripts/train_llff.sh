#!/bin/bash
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script for training on the LLFF dataset.
SCENE=orchids
EXPERIMENT="llff"
TRAIN_DIR=/workspace/projects/geconerf/$SCENE
DATA_DIR=/workspace/projects/datasets/nerf_llff_data/$SCENE

# rm $TRAIN_DIR/*
python -m mipnerf.train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=/workspace/projects/geconerf/configs/llff.gin \
  --logtostderr \
  --input_dataset=$scene \
  --lr_threshold=0.02\
  --start_reg_step=900 \
  --num_nearby_views=5 \
  --window_parameter=15000 \
done