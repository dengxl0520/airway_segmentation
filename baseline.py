import torch
import torch.nn as nn
import numpy as np
from models.model_mj_in3sad import UNet3D
# Baseline


config = {'pad_value': 0,
          'augtype': {'flip': True, 'swap': False, 'smooth': False, 'jitter': True, 'split_jitter': True},
          'startepoch': 0, 'lr_stage': np.array([10, 20, 40, 60]), 'lr': np.array([3e-3, 3e-4, 3e-5, 3e-6]),
          'dataset_path': 'preprocessed_datasets', 'dataset_split': './split_dataset.pickle'}


def get_model(args=None):
	net = UNet3D(in_channels=1, out_channels=1, coord=True)
	print(net)
	print('# of network parameters:', sum(param.numel() for param in net.parameters()))
	return config, net


if __name__ == '__main__':
	_, model = get_model()
