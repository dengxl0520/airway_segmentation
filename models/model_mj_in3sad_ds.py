import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .model_mj_in3sad import UNet3D as Base


class UNet3D(Base):
	"""
	Baseline model for pulmonary airway segmentation
	with deep supervision
	"""
	def __init__(self, in_channels=1, out_channels=1, coord=True):
		"""
		:param in_channels: input channel numbers
		:param out_channels: output channel numbers
		:param coord: boolean, True=Use coordinates as position information, False=not
		"""
		super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, coord=coord)
		self.upsampling4 = nn.Upsample(scale_factor=4)
		self.upsampling8 = nn.Upsample(scale_factor=8)
		self.dsconv6 = nn.Conv3d(128, 1, 3, 1, 1)  # deep supervision
		self.dsconv7 = nn.Conv3d(64, 1, 3, 1, 1)  # deep supervision
		self.dsconv8 = nn.Conv3d(32, 1, 3, 1, 1)  # deep supervision

	def forward(self, input, coordmap=None):
		"""
		:param input: shape = (batch_size, num_channels, D, H, W) \
		:param coordmap: shape = (batch_size, 3, D, H, W)
		:return: output segmentation tensors list, attention mapping
		"""
		conv1 = self.conv1(input)
		x = self.pooling(conv1)
		
		conv2 = self.conv2(x)
		x = self.pooling(conv2)
		
		conv3 = self.conv3(x)
		x = self.pooling(conv3)
		
		conv4 = self.conv4(x)
		x = self.pooling(conv4)

		conv5 = self.conv5(x)

		x = self.upsampling(conv5)
		x = torch.cat([x, conv4], dim=1)
		conv6 = self.conv6(x)
		ds_6 = self.sigmoid(self.upsampling8(self.dsconv6(conv6)))

		x = self.upsampling(conv6)
		x = torch.cat([x, conv3], dim=1)
		conv7 = self.conv7(x)
		ds_7 = self.sigmoid(self.upsampling4(self.dsconv7(conv7)))
		
		x = self.upsampling(conv7)
		x = torch.cat([x, conv2], dim=1)
		conv8 = self.conv8(x)
		ds_8 = self.sigmoid(self.upsampling(self.dsconv8(conv8)))
		
		x = self.upsampling(conv8)

		if (self.coord is True) and (coordmap is not None):
			x = torch.cat([x, conv1, coordmap], dim=1)
		else:
			x = torch.cat([x, conv1], dim=1)

		conv9 = self.conv9(x)
		
		x = self.conv10(conv9)

		x = self.sigmoid(x)
		
		mapping3 = torch.sum(torch.pow(conv3, exponent=2), dim=1, keepdim=True)
		mapping4 = torch.sum(torch.pow(conv4, exponent=2), dim=1, keepdim=True)
		mapping5 = torch.sum(torch.pow(conv5, exponent=2), dim=1, keepdim=True)
		mapping6 = torch.sum(torch.pow(conv6, exponent=2), dim=1, keepdim=True)
		mapping7 = torch.sum(torch.pow(conv7, exponent=2), dim=1, keepdim=True)
		mapping8 = torch.sum(torch.pow(conv8, exponent=2), dim=1, keepdim=True)
		mapping9 = torch.sum(torch.pow(conv9, exponent=2), dim=1, keepdim=True)

		return [x, ds_6, ds_7, ds_8], [mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9]


# class UNet3D(nn.Module):
# 	"""
# 	Baseline model for pulmonary airway segmentation
# 	with deep supervision
# 	"""
#
# 	def __init__(self, in_channels=1, out_channels=1, coord=True):
# 		"""
# 		:param in_channels: input channel numbers
# 		:param out_channels: output channel numbers
# 		:param coord: boolean, True=Use coordinates as position information, False=not
# 		"""
# 		super(UNet3D, self).__init__()
# 		self.in_channels = in_channels
# 		self.out_channels = out_channels
# 		self.coord = coord
# 		## CBR Deep Supervision
# 		self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))
# 		self.upsampling = nn.Upsample(scale_factor=2)
# 		self.upsampling4 = nn.Upsample(scale_factor=4)
# 		self.upsampling8 = nn.Upsample(scale_factor=8)
# 		# use instance normalization instead of batch normalization
# 		self.conv1 = nn.Sequential(
# 			nn.Conv3d(in_channels=self.in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
# 			nn.InstanceNorm3d(8),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(8, 16, 3, 1, 1),
# 			nn.InstanceNorm3d(16),
# 			nn.ReLU(inplace=True))
#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
# 			nn.InstanceNorm3d(16),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(16, 32, 3, 1, 1),
# 			nn.InstanceNorm3d(32),
# 			nn.ReLU(inplace=True))
#
# 		self.conv3 = nn.Sequential(
# 			nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
# 			nn.InstanceNorm3d(32),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(32, 64, 3, 1, 1),
# 			nn.InstanceNorm3d(64),
# 			nn.ReLU(inplace=True))
#
# 		self.conv4 = nn.Sequential(
# 			nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
# 			nn.InstanceNorm3d(64),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(64, 128, 3, 1, 1),
# 			nn.InstanceNorm3d(128),
# 			nn.ReLU(inplace=True))
#
# 		self.conv5 = nn.Sequential(
# 			nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
# 			nn.InstanceNorm3d(128),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(128, 256, 3, 1, 1),
# 			nn.InstanceNorm3d(256),
# 			nn.ReLU(inplace=True))
#
# 		self.conv6 = nn.Sequential(
# 			nn.Conv3d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
# 			nn.InstanceNorm3d(128),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(128, 128, 3, 1, 1),
# 			nn.InstanceNorm3d(128),
# 			nn.ReLU(inplace=True))
# 		self.dsconv6 = nn.Conv3d(128, 1, 3, 1, 1)  # deep supervision
#
# 		self.conv7 = nn.Sequential(
# 			nn.Conv3d(128 + 64, 64, 3, 1, 1),
# 			nn.InstanceNorm3d(64),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm3d(64),
# 			nn.ReLU(inplace=True))
# 		self.dsconv7 = nn.Conv3d(64, 1, 3, 1, 1)  # deep supervision
#
# 		self.conv8 = nn.Sequential(
# 			nn.Conv3d(64 + 32, 32, 3, 1, 1),
# 			nn.InstanceNorm3d(32),
# 			nn.ReLU(inplace=True),
# 			nn.Conv3d(32, 32, 3, 1, 1),
# 			nn.InstanceNorm3d(32),
# 			nn.ReLU(inplace=True))
# 		self.dsconv8 = nn.Conv3d(32, 1, 3, 1, 1)  # deep supervision
#
# 		if self.coord:
# 			self.conv9 = nn.Sequential(
# 				nn.Conv3d(32 + 16 + 3, 16, 3, 1, 1),
# 				nn.InstanceNorm3d(16),
# 				nn.ReLU(inplace=True),
# 				nn.Conv3d(16, 16, 3, 1, 1),
# 				nn.InstanceNorm3d(16),
# 				nn.ReLU(inplace=True))
# 		else:
# 			self.conv9 = nn.Sequential(
# 				nn.Conv3d(32 + 16, 16, 3, 1, 1),
# 				nn.InstanceNorm3d(16),
# 				nn.ReLU(inplace=True),
# 				nn.Conv3d(16, 16, 3, 1, 1),
# 				nn.InstanceNorm3d(16),
# 				nn.ReLU(inplace=True))
#
# 		self.sigmoid = nn.Sigmoid()
#
# 		self.conv10 = nn.Conv3d(16, self.out_channels, 1, 1, 0)
#
# 	def forward(self, input, coordmap=None):
# 		"""
# 		:param input_tensor: input, shape = (batch_size, num_channels, D, H, W) \
# 		:param coordinate_tensor: coordmap, shape = (batch_size, num_channels, D, H, W)
# 		:return: output segmentation tensors list, attention mapping
# 		"""
# 		conv1 = self.conv1(input)
# 		x = self.pooling(conv1)
#
# 		conv2 = self.conv2(x)
# 		x = self.pooling(conv2)
#
# 		conv3 = self.conv3(x)
# 		x = self.pooling(conv3)
#
# 		conv4 = self.conv4(x)
# 		x = self.pooling(conv4)
#
# 		conv5 = self.conv5(x)
#
# 		x = self.upsampling(conv5)
# 		x = torch.cat([x, conv4], dim=1)
# 		conv6 = self.conv6(x)
# 		ds_6 = self.sigmoid(self.upsampling8(self.dsconv6(conv6)))
#
# 		x = self.upsampling(conv6)
# 		x = torch.cat([x, conv3], dim=1)
# 		conv7 = self.conv7(x)
# 		ds_7 = self.sigmoid(self.upsampling4(self.dsconv7(conv7)))
#
# 		x = self.upsampling(conv7)
# 		x = torch.cat([x, conv2], dim=1)
# 		conv8 = self.conv8(x)
# 		ds_8 = self.sigmoid(self.upsampling(self.dsconv8(conv8)))
#
# 		x = self.upsampling(conv8)
#
# 		if (self.coord is True) and (coordmap is not None):
# 			x = torch.cat([x, conv1, coordmap], dim=1)
# 		else:
# 			x = torch.cat([x, conv1], dim=1)
#
# 		conv9 = self.conv9(x)
#
# 		x = self.conv10(conv9)
#
# 		x = self.sigmoid(x)
#
# 		mapping3 = torch.sum(torch.pow(conv3, exponent=2), dim=1, keepdim=True)
# 		mapping4 = torch.sum(torch.pow(conv4, exponent=2), dim=1, keepdim=True)
# 		mapping5 = torch.sum(torch.pow(conv5, exponent=2), dim=1, keepdim=True)
# 		mapping6 = torch.sum(torch.pow(conv6, exponent=2), dim=1, keepdim=True)
# 		mapping7 = torch.sum(torch.pow(conv7, exponent=2), dim=1, keepdim=True)
# 		mapping8 = torch.sum(torch.pow(conv8, exponent=2), dim=1, keepdim=True)
# 		mapping9 = torch.sum(torch.pow(conv9, exponent=2), dim=1, keepdim=True)
#
# 		return [x, ds_6, ds_7, ds_8], [mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9]


if __name__ == '__main__':
	net = UNet3D(in_channels=1, out_channels=1)
	print(net)
	print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
# Number of network parameters: 4124900 Baseline + Deep Supervision

