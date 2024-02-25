#!/bin/bash
set -e

cd ../airway_segmentation

# baseline with Feature Recalibration
# training
python main.py --model baseline_fr -b 1 --save-dir baseline_fr \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 1 --epoch 60 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 0

# validation
python main.py --model baseline_fr -b 1 --save-dir baseline_fr \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 61 --epoch 62 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 1 --debugval 1 \
--resume './results/baseline_fr/060.ckpt'

# testing
python main.py --model baseline_fr -b 1 --save-dir baseline_fr \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 61 --epoch 62 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 1 --test 1 \
--resume './results/baseline_fr/060.ckpt'

# baseline with Feature Recalibration with Attention Distillation
# training
python main.py --model baseline_fr -b 1 --save-dir baseline_fr_ad \
--sadencoder 0 --saddecoder 1 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 1 --epoch 60 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 0 \
--resume './results/baseline_fr/060.ckpt'

# validation
python main.py --model baseline_fr -b 1 --save-dir baseline_fr_ad \
--sadencoder 0 --saddecoder 1 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 61 --epoch 62 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 1 --debugval 1 \
--resume './results/baseline_fr_ad/060.ckpt'

# testing
python main.py --model baseline_fr -b 1 --save-dir baseline_fr_ad \
--sadencoder 0 --saddecoder 1 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 61 --epoch 62 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 1 --test 1 \
--resume './results/baseline_fr_ad/060.ckpt'