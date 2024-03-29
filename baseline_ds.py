import torch
import torch.nn as nn
import numpy as np
from baseline import config
from models.model_mj_in3sad_ds import UNet3D
# Baseline + Deep Supervision


def get_model(args=None):
	net = UNet3D(in_channels=1, out_channels=1, coord=True)
	print(net)
	print('# of network parameters:', sum(param.numel() for param in net.parameters()))
	return config, net


if __name__ == '__main__':
	_, model = get_model()
