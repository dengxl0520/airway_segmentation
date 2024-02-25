CUDA_VISIBLE_DEVICES=0 python main.py --model baseline_fr_ds \
--dataset ATM -b 2 --workers 8 \
--save-dir baseline_fr_ds_ATM \
--sadencoder 0 --saddecoder 0 \
--cubesize 80 192 304 --stridet 64 96 152 --stridev 64 72 72 \
--start-epoch 1 --epoch 60 --sgd 0 \
--randsel 0 --resumepart 0 --featsave 0 --deepsupervision 1
