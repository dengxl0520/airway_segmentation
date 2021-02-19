import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ProjectExciteLayer(nn.Module):
	"""
		Redesign the spatial information integration method for
		feature recalibration, based on the original
		Project & Excite Module
	"""
	def __init__(self, num_channels, D, H, W, reduction_ratio=2):
		"""
		:param num_channels: No of input channels
		:param D, H, W: Spatial dimension of the input feature cube
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(ProjectExciteLayer, self).__init__()
		num_channels_reduced = num_channels // reduction_ratio
		self.reduction_ratio = reduction_ratio
		self.convModule = nn.Sequential(
			nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1),\
			nn.ReLU(inplace=True),\
			nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1),\
			nn.Sigmoid())
		self.spatialdim = [D, H, W]
		self.D_squeeze = nn.Conv3d(in_channels=D, out_channels=1, kernel_size=1, stride=1)
		self.H_squeeze = nn.Conv3d(in_channels=H, out_channels=1, kernel_size=1, stride=1)
		self.W_squeeze = nn.Conv3d(in_channels=W, out_channels=1, kernel_size=1, stride=1)

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
		:return: output tensor, mapping
		"""
		squared_tensor = torch.pow(input_tensor, exponent=2)

		# Project:
		# Average along channels and different axes
		D, H, W = self.spatialdim[0], self.spatialdim[1], self.spatialdim[2]
		D_channel = input_tensor.permute(0, 2, 1, 3, 4) # B, D, C, H, W
		H_channel = input_tensor.permute(0, 3, 2, 1, 4) # B, H, D, C, W

		squeeze_tensor_1D = self.D_squeeze(D_channel) # B, 1, C, H, W

		squeeze_tensor_W = squeeze_tensor_1D.permute(0, 3, 1, 2, 4) # B, H, 1, C, W
		squeeze_tensor_W = self.H_squeeze(squeeze_tensor_W).permute(0, 3, 2, 1, 4) # B, C, 1, 1, W

		squeeze_tensor_H = squeeze_tensor_1D.permute(0, 4, 1, 3, 2) # B, W, 1, H, C
		squeeze_tensor_H = self.W_squeeze(squeeze_tensor_H).permute(0, 4, 2, 3, 1) # B, C, 1, H, 1

		squeeze_tensor_D = self.H_squeeze(H_channel).permute(0, 4, 2, 1, 3) # B, W, D, 1, C
		squeeze_tensor_D = self.W_squeeze(squeeze_tensor_D).permute(0, 4, 2, 3, 1) # B, C, D, 1, 1

		final_squeeze_tensor = squeeze_tensor_W + squeeze_tensor_H + squeeze_tensor_D
		# Excitation:
		final_squeeze_tensor = self.convModule(final_squeeze_tensor)
		
		output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

		feature_mapping = torch.sum(squared_tensor, dim=1, keepdim=True)

		return output_tensor, feature_mapping

class UNet3D(nn.Module):
	"""
	Baseline model with Feature Recalibration module
	for pulmonary airway segmentation
	with deep supervision
	"""
	def __init__(self, in_channels=1, out_channels=1, coord=True, Dmax=80, Hmax=192, Wmax=304):
		"""
		:param in_channels: input channel numbers
		:param out_channels: output channel numbers
		:param coord: boolean, True=Use coordinates as position information, False=not
		:param Dmax: the size of the largest feature cube in depth, default=80
		:param Hmax: the size of the largest feature cube in height, default=192
		:param Wmax: the size of the largest feature cube in width, default=304
		"""
		super(UNet3D, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.coord = coord
		## CBR Deep Supervision
		self.pooling = nn.MaxPool3d(kernel_size=(2,2,2))
		self.upsampling = nn.Upsample(scale_factor=2)
		self.upsampling4 = nn.Upsample(scale_factor=4)
		self.upsampling8 = nn.Upsample(scale_factor=8)
		# use instance normalization instead of batch normalization
		self.conv1 = nn.Sequential(
			nn.Conv3d(in_channels=self.in_channels, out_channels = 8, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(8),
			nn.ReLU(inplace=True),
			nn.Conv3d(8, 16, 3, 1, 1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True))
		self.pe1 = ProjectExciteLayer(16, Dmax, Hmax, Wmax)
		
		self.conv2 = nn.Sequential(
			nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			nn.Conv3d(16, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True))
		self.pe2 = ProjectExciteLayer(32, Dmax//2, Hmax//2, Wmax//2)

		self.conv3 = nn.Sequential(
			nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True))
		self.pe3 = ProjectExciteLayer(64, Dmax//4, Hmax//4, Wmax//4)
	
		self.conv4 = nn.Sequential(
			nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True),
			nn.Conv3d(64, 128, 3, 1, 1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True))
		self.pe4 = ProjectExciteLayer(128, Dmax//8, Hmax//8, Wmax//8)

		self.conv5 = nn.Sequential(
			nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True),
			nn.Conv3d(128, 256, 3, 1, 1),
			nn.InstanceNorm3d(256),
			nn.ReLU(inplace=True))
		self.pe5 = ProjectExciteLayer(256, Dmax//16, Hmax//16, Wmax//16)

		self.conv6 = nn.Sequential(
			nn.Conv3d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True),
			nn.Conv3d(128, 128, 3, 1, 1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True))
		self.pe6 = ProjectExciteLayer(128, Dmax//8, Hmax//8, Wmax//8)
		self.dsconv6 = nn.Conv3d(128, 1, 3, 1, 1) # deep supervision
		
		self.conv7 = nn.Sequential(
			nn.Conv3d(128 + 64, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True),
			nn.Conv3d(64, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True))
		self.pe7 = ProjectExciteLayer(64, Dmax//4, Hmax//4, Wmax//4)
		self.dsconv7 = nn.Conv3d(64, 1, 3, 1, 1) # deep supervision
		
		self.conv8 = nn.Sequential(
			nn.Conv3d(64 + 32, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True))
		self.pe8 = ProjectExciteLayer(32, Dmax//2, Hmax//2, Wmax//2)
		self.dsconv8 = nn.Conv3d(32, 1, 3, 1, 1) # deep supervision
		
		if self.coord:
			self.conv9 = nn.Sequential(
				nn.Conv3d(32 + 16 + 3, 16, 3, 1, 1),
				nn.InstanceNorm3d(16),
				nn.ReLU(inplace=True),
				nn.Conv3d(16, 16, 3, 1, 1),
				nn.InstanceNorm3d(16),
				nn.ReLU(inplace=True))
		else:
			self.conv9 = nn.Sequential(
				nn.Conv3d(32 + 16, 16, 3, 1, 1),
				nn.InstanceNorm3d(16),
				nn.ReLU(inplace=True),
				nn.Conv3d(16, 16, 3, 1, 1),
				nn.InstanceNorm3d(16),
				nn.ReLU(inplace=True))
		self.pe9 = ProjectExciteLayer(16, Dmax, Hmax, Wmax)

		self.conv10 = nn.Conv3d(16, self.out_channels, 1, 1, 0)
	
		self.sigmoid = nn.Sigmoid()

	def forward(self, input, coordmap=None):
		"""
		:param input_tensor: input, shape = (batch_size, num_channels, D, H, W) \
		:param coordinate_tensor: coordmap, shape = (batch_size, num_channels, D, H, W)
		:return: output segmentation tensors list, attention mapping
		"""
		conv1 = self.conv1(input)
		conv1, _ = self.pe1(conv1)
		x = self.pooling(conv1)
		
		conv2 = self.conv2(x)
		conv2, _ = self.pe2(conv2)
		x = self.pooling(conv2)
		
		conv3 = self.conv3(x)
		conv3, mapping3 = self.pe3(conv3)
		x = self.pooling(conv3)
		
		conv4 = self.conv4(x)
		conv4, mapping4 = self.pe4(conv4)
		x = self.pooling(conv4)

		conv5 = self.conv5(x)
		conv5, mapping5 = self.pe5(conv5)

		x = self.upsampling(conv5)
		x = torch.cat([x, conv4], dim=1)
		conv6 = self.conv6(x)
		conv6, mapping6 = self.pe6(conv6)
		ds_6 = self.sigmoid(self.upsampling8(self.dsconv6(conv6)))
		
		x = self.upsampling(conv6)
		x = torch.cat([x, conv3], dim=1)
		conv7 = self.conv7(x)
		conv7, mapping7 = self.pe7(conv7)
		ds_7 = self.sigmoid(self.upsampling4(self.dsconv7(conv7)))
		
		x = self.upsampling(conv7)
		x = torch.cat([x, conv2], dim=1)
		conv8 = self.conv8(x)
		conv8, mapping8 = self.pe8(conv8)
		ds_8 = self.sigmoid(self.upsampling(self.dsconv8(conv8)))
		
		x = self.upsampling(conv8)

		if (self.coord is True) and (coordmap is not None):
			x = torch.cat([x, conv1, coordmap], dim=1)
		else:
			x = torch.cat([x, conv1], dim=1)

		conv9 = self.conv9(x)
		conv9, mapping9 = self.pe9(conv9)

		x = self.conv10(conv9)

		x = self.sigmoid(x)

		return [x, ds_6, ds_7, ds_8], [mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9]

if __name__ == '__main__':
	net = UNet3D(in_channels=1, out_channels=1)
	print (net)
	print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
# Number of network parameters: 4237283  Baseline + Feature Recalibration + Deep Supervision
