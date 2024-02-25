#!/bin/bash
set -e

cd ../airway_segmentation

# baseline with Feature Recalibration with Deep Supervision
# training
python main.py --model baseline_fr_ds -b 1 --save-dir baseline_fr_ds \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 1 --epoch 60 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 0 --deepsupervision 1

# validation
python main.py --model baseline_fr_ds -b 1 --save-dir baseline_fr_ds \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 61 --epoch 62 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 1 --debugval 1 \
--resume './results/baseline_fr_ds/060.ckpt' --deepsupervision 1

# testing
python main.py --model baseline_fr_ds -b 1 --save-dir baseline_fr_ds \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 --worker 1 \
--start-epoch 61 --epoch 62 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 1 --test 1 \
--resume './results/baseline_fr_ds/060.ckpt' --deepsupervision 1