#!/bin/bash
set -e

test_epoch=60

cd ../artery-vein_segmentation

# baseline with Feature Recalibration
# training
python main.py --model baseline_fr -b 1 --save-dir baseline-fr \
--sadencoder 0 --saddecoder 0 \
--cubesize 64 176 176 --stridet 60 144 144 --stridev 60 144 144 --worker 1 \
--start-epoch 1 --epoch 60 --sgd 0 --featsave 0 \
--randsel 0 --resumepart 0

# validation
python main.py --model baseline_fr -b 2 --save-dir baseline-fr \
--sadencoder 0 --saddecoder 0 \
--cubesize 64 176 176 --stridet 60 144 144 --stridev 60 144 144 --worker 1 \
--start-epoch $test_epoch --epoch $(($test_epoch+1)) --sgd 0 --featsave 1 \
--randsel 0 --resumepart 0 --debugval 1 \
--resume './results/baseline-fr/060.ckpt'

# testing
python main.py --model baseline_fr -b 2 --save-dir baseline-fr \
--sadencoder 0 --saddecoder 0 \
--cubesize 64 176 176 --stridet 60 144 144 --stridev 60 144 144 --worker 1 \
--start-epoch $test_epoch --epoch $(($test_epoch+1)) --sgd 0 --featsave 1 \
--randsel 0 --resumepart 0 --test 1 \
--resume './results/baseline-fr/060.ckpt'

# baseline with Feature Recalibration and Attention Distillation
# training
python main.py --model baseline_fr -b 1 --save-dir baseline-fr-ad \
--sadencoder 0 --saddecoder 1 \
--cubesize 64 176 176 --stridet 60 144 144 --stridev 60 144 144 --worker 1 \
--start-epoch 1 --epoch 60 --sgd 0 --featsave 0 \
--randsel 0 --resumepart 0 \
--resume './results/baseline-fr/060.ckpt'

# validation
python main.py --model baseline_fr -b 2 --save-dir baseline-fr-ad \
--sadencoder 0 --saddecoder 1 \
--cubesize 64 176 176 --stridet 60 144 144 --stridev 60 144 144 --worker 1 \
--start-epoch $test_epoch --epoch $(($test_epoch+1)) --sgd 0 --featsave 1 \
--randsel 0 --resumepart 0 --debugval 1 \
--resume './results/baseline-fr-ad/060.ckpt'

# testing
python main.py --model baseline_fr -b 2 --save-dir baseline-fr-ad \
--sadencoder 0 --saddecoder 1 \
--cubesize 64 176 176 --stridet 60 144 144 --stridev 60 144 144 --worker 1 \
--start-epoch $test_epoch --epoch $(($test_epoch+1)) --sgd 0 --featsave 1 \
--randsel 0 --resumepart 0 --test 1 \
--resume './results/baseline-fr-ad/060.ckpt'
