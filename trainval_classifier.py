import numpy as np
import os
import time
import torch
from tqdm import tqdm
from utils import save_itk, load_itk_image, dice_coef_np, ppv_np,\
sensitivity_np, acc_np, combine_total_avg, combine_total, normalize_min_max
from loss import dice_loss, binary_cross_entropy, focal_loss, sad_loss
from torch.cuda import empty_cache
import csv
from scipy.ndimage.interpolation import zoom

th_bin = 0.5


def get_lr(epoch, args):
	"""
	:param epoch: current epoch number
	:param args: global arguments args
	:return: learning rate of the next epoch
	"""
	if args.lr is None:
		assert epoch <= args.lr_stage[-1]
		lrstage = np.sum(epoch > args.lr_stage)
		lr = args.lr_preset[lrstage]
	else:
		lr = args.lr
	return lr


def train_casenet(epoch, model, data_loader, optimizer, args, save_dir):
	"""
	:param epoch: current epoch number
	:param model: CNN model
	:param data_loader: training data
	:param optimizer: training optimizer
	:param args: global arguments args
	:param save_dir: save directory
	:return: performance evaluation of the current epoch
	"""
	model.train()
	starttime = time.time()
	sidelen = args.stridet
	margin = args.cubesize

	lr = get_lr(epoch, args)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	assert (lr is not None)
	optimizer.zero_grad()

	lossHist = []
	dice_total = []
	ppv_total = []
	acc_total = []
	dice_hard_total = []
	sensitivity_total = []
	traindir = os.path.join(save_dir, 'train')
	if not os.path.exists(traindir):
		os.mkdir(traindir)
	training_log = os.path.join(traindir, 'train_log.txt')

	for i, (x, y, coord, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(tqdm(data_loader)):
		######Wrap Tensor##########
		NameID = NameID[0]
		SplitID = SplitID[0]    
		batchlen = x.size(0)
		x = x.cuda()
		y = y.cuda()
		###############################
		coord = coord.cuda()
		casePreds, attentions = model(x, coord)

		if args.deepsupervision:
			casePred = casePreds[0]
			ds6, ds7, ds8 = casePreds[1], casePreds[2], casePreds[3]
			loss = dice_loss(casePred, y) + dice_loss(ds6, y) + dice_loss(ds7, y) + dice_loss(ds8, y)
		else:
			casePred = casePreds
			loss = dice_loss(casePred, y)

		# loss += binary_cross_entropy(casePred, y)
		loss += focal_loss(casePred, y)

		if args.sadencoder == 1:
			# attentions 0, 1, 2
			gamma_sad = [0.1, 0.1, 0.1]
			for iter_sad in range(2):
				loss += (gamma_sad[iter_sad])*sad_loss(attentions[iter_sad], attentions[iter_sad+1], encoder_flag=True)

		if args.saddecoder == 1:
			# attentions 3, 4, 5, 6
			gamma_sad = [0.1, 0.1, 0.1]
			for iter_sad in range(3, 6):
				loss += (gamma_sad[iter_sad-3])*sad_loss(attentions[iter_sad], attentions[iter_sad+1], encoder_flag=False)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# for evaluation
		lossHist.append(loss.item())
		# segmentation calculating metrics#######################
		outdata = casePred.cpu().data.numpy()
		segdata = y.cpu().data.numpy()
		segdata = (segdata > th_bin)

		for j in range(batchlen):
			dice = dice_coef_np(outdata[j, 0], segdata[j, 0])
			segpred = (outdata[j, 0] > th_bin)
			dicehard = dice_coef_np(segpred, segdata[j, 0])
			ppv = ppv_np(segpred, segdata[j, 0])
			sensiti = sensitivity_np(segpred, segdata[j, 0])
			acc = acc_np(segpred, segdata[j, 0])

			##########################################################################
			dice_total.append(dice)
			ppv_total.append(ppv)
			sensitivity_total.append(sensiti)
			acc_total.append(acc)
			dice_hard_total.append(dicehard)

	##################################################################################
	
	endtime = time.time()
	lossHist = np.array(lossHist)
	mean_dice = np.mean(np.array(dice_total))
	mean_dice_hard = np.mean(np.array(dice_hard_total))
	mean_ppv = np.mean(np.array(ppv_total))
	mean_sensiti = np.mean(np.array(sensitivity_total))
	mean_acc = np.mean(np.array(acc_total))
	mean_loss = np.mean(lossHist)

	print('Train, epoch %d, loss %.4f, accuracy %.4f, sensitivity %.4f, dice %.4f, dice hard %.4f, ppv %.4f, time %3.2f, lr % .5f '
		  %(epoch, mean_loss, mean_acc, mean_sensiti, mean_dice, mean_dice_hard, mean_ppv, endtime-starttime,lr))
	print ()
	empty_cache()
	return mean_loss, mean_acc, mean_sensiti, mean_dice, mean_ppv


def val_casenet(epoch, model, data_loader, args, save_dir, test_flag=False):
	"""
	:param epoch: current epoch number
	:param model: CNN model
	:param data_loader: evaluation and testing data
	:param args: global arguments args
	:param save_dir: save directory
	:param test_flag: current mode of validation or testing
	:return: performance evaluation of the current epoch
	"""
	model.eval()
	starttime = time.time()

	sidelen = args.stridev
	if args.cubesizev is not None:
		margin = args.cubesizev
	else:
		margin = args.cubesize

	name_total = []
	lossHist = []

	dice_total = []
	ppv_total = []
	sensitivity_total = []
	dice_hard_total = []
	acc_total = []

	if test_flag:
		valdir = os.path.join(save_dir, 'test%03d'%(epoch))
		state_str = 'test'
	else:
		valdir = os.path.join(save_dir, 'val%03d'%(epoch))
		state_str = 'val'
	if not os.path.exists(valdir):
		os.mkdir(valdir)

	p_total = {}
	x_total = {}
	y_total = {}
	feat3_total = {}
	feat4_total = {}
	feat5_total = {}
	feat6_total = {}

	with torch.no_grad():
		for i, (x, y, coord, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(tqdm(data_loader)):
			######Wrap Tensor##########
			NameID = NameID[0]
			SplitID = SplitID[0] 
			batchlen = x.size(0)
			x = x.cuda()
			y = y.cuda()
			####################################################
			coord = coord.cuda()
			casePreds, attentions = model(x, coord)
			
			if args.deepsupervision:
				casePred = casePreds[0]
				ds6, ds7, ds8 = casePreds[1], casePreds[2], casePreds[3]
				loss = dice_loss(casePred, y) + dice_loss(ds6, y) + dice_loss(ds7, y) + dice_loss(ds8, y)
			else:
				casePred = casePreds
				loss = dice_loss(casePred, y)

			# loss += binary_cross_entropy(casePred, y)
			loss += focal_loss(casePred, y)

			if args.sadencoder == 1:
				# attentions 0, 1, 2
				gamma_sad = [0.1, 0.1, 0.1]
				for iter_sad in range(2):
					loss += gamma_sad[iter_sad]*sad_loss(attentions[iter_sad], attentions[iter_sad+1], encoder_flag=True)
			if args.saddecoder == 1:
				# attentions 3, 4, 5, 6
				gamma_sad = [0.1, 0.1, 0.1]
				for iter_sad in range(3, 6):
					loss += gamma_sad[iter_sad-3]*sad_loss(attentions[iter_sad], attentions[iter_sad+1], encoder_flag=False)

			# for evaluation
			lossHist.append(loss.item())

			#####################seg data#######################
			outdata = casePred.cpu().data.numpy()
			#######################################################################
			segdata = y.cpu().data.numpy()
			segdata = (segdata > th_bin)
			xdata = x.cpu().data.numpy()
			origindata = org.numpy()
			spacingdata = spac.numpy()

			feat3 = attentions[3].cpu().data.numpy()
			feat4 = attentions[4].cpu().data.numpy()
			feat5 = attentions[5].cpu().data.numpy()
			feat6 = attentions[6].cpu().data.numpy()
			#######################################################################
			#################REARRANGE THE DATA BY SPLIT ID########################
			for j in range(batchlen):
				curxdata = (xdata[j, 0]*255)
				curydata = segdata[j, 0]
				segpred = outdata[j, 0]
				curorigin = origindata[j].tolist()
				curspacing = spacingdata[j].tolist()
				cursplitID = int(SplitID[j])
				assert (cursplitID >= 0)
				curName = NameID[j]
				curnzhw = nzhw[j]
				curshape = ShapeOrg[j]

				if not (curName in x_total.keys()):
					x_total[curName] = []
				if not (curName in y_total.keys()):
					y_total[curName] = []
				if not (curName in p_total.keys()):
					p_total[curName] = []
				if not (curName in feat3_total.keys()):
					feat3_total[curName] = []
				if not (curName in feat4_total.keys()):
					feat4_total[curName] = []
				if not (curName in feat5_total.keys()):
					feat5_total[curName] = []
				if not (curName in feat6_total.keys()):
					feat6_total[curName] = []

				# curxinfo = [curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]
				curyinfo = [curydata, cursplitID, curnzhw, curshape, curorigin, curspacing]
				curpinfo = [segpred, cursplitID, curnzhw, curshape, curorigin, curspacing]
				# x_total[curName].append(curxinfo)
				y_total[curName].append(curyinfo)
				p_total[curName].append(curpinfo)

				if args.featsave:
					curfeat3 = feat3[j, 0]
					curfeat4 = feat4[j, 0]
					curfeat5 = feat5[j, 0]
					curfeat6 = feat6[j, 0]
					curfeat3 = zoom(curfeat3, 8, order=0, mode='nearest')
					curfeat4 = zoom(curfeat4, 4, order=0, mode='nearest')
					curfeat5 = zoom(curfeat5, 2, order=0, mode='nearest')
					curf3info = [curfeat3, cursplitID, curnzhw, curshape, curorigin, curspacing]
					curf4info = [curfeat4, cursplitID, curnzhw, curshape, curorigin, curspacing]
					curf5info = [curfeat5, cursplitID, curnzhw, curshape, curorigin, curspacing]
					curf6info = [curfeat6, cursplitID, curnzhw, curshape, curorigin, curspacing]
					feat3_total[curName].append(curf3info)
					feat4_total[curName].append(curf4info)
					feat5_total[curName].append(curf5info)
					feat6_total[curName].append(curf6info)

	# combine all the cases together
	for curName in x_total.keys():
		# curx = x_total[curName]
		cury = y_total[curName]
		curp = p_total[curName]
		# x_combine, xorigin, xspacing = combine_total(curx, sidelen, margin)
		y_combine, curorigin, curspacing = combine_total(cury, sidelen, margin)
		p_combine, porigin, pspacing = combine_total_avg(curp, sidelen, margin)
		p_combine_bw = (p_combine > th_bin)
		# curpath = os.path.join(valdir, '%s-case-org.nii.gz'%(curName))
		curypath = os.path.join(valdir, '%s-case-gt.nii.gz'%(curName))
		curpredpath = os.path.join(valdir, '%s-case-pred.nii.gz'%(curName))
		# save_itk(x_combine.astype(dtype='uint8'), curorigin, curspacing, curpath)
		save_itk(y_combine.astype(dtype='uint8'), curorigin, curspacing, curypath)
		save_itk(p_combine_bw.astype(dtype='uint8'), curorigin, curspacing, curpredpath)

		if args.featsave:
			curf3 = feat3_total[curName]
			curf4 = feat4_total[curName]
			curf5 = feat5_total[curName]
			curf6 = feat6_total[curName]
			f3, forg, fspac = combine_total_avg(curf3, sidelen, margin)
			f4, _, _ = combine_total_avg(curf4, sidelen, margin)
			f5, _, _ = combine_total_avg(curf5, sidelen, margin)
			f6, _, _ = combine_total_avg(curf6, sidelen, margin)
			f3 = normalize_min_max(f3)*255
			f4 = normalize_min_max(f4)*255
			f5 = normalize_min_max(f5)*255
			f6 = normalize_min_max(f6)*255
			curf3path = os.path.join(valdir, '%s-case-f3.nii.gz'%(curName))
			curf4path = os.path.join(valdir, '%s-case-f4.nii.gz'%(curName))
			curf5path = os.path.join(valdir, '%s-case-f5.nii.gz'%(curName))
			curf6path = os.path.join(valdir, '%s-case-f6.nii.gz'%(curName))
			save_itk(f3.astype(dtype='uint8'), curorigin, curspacing, curf3path)
			save_itk(f4.astype(dtype='uint8'), curorigin, curspacing, curf4path)
			save_itk(f5.astype(dtype='uint8'), curorigin, curspacing, curf5path)
			save_itk(f6.astype(dtype='uint8'), curorigin, curspacing, curf6path)

		########################################################################
		curdicehard = dice_coef_np(p_combine_bw, y_combine)
		curdice = dice_coef_np(p_combine, y_combine)
		curppv = ppv_np(p_combine_bw, y_combine)
		cursensi = sensitivity_np(p_combine_bw, y_combine)
		curacc = acc_np(p_combine_bw, y_combine)
		########################################################################
		dice_total.append(curdice)
		ppv_total.append(curppv)
		acc_total.append(curacc)
		name_total.append(curName)
		sensitivity_total.append(cursensi)
		dice_hard_total.append(curdicehard)
		del cury, curp, y_combine, p_combine_bw, p_combine

	endtime = time.time()
	lossHist = np.array(lossHist)

	all_results = {'lidc':[], 'exact09':[]}

	with open(os.path.join(valdir, 'val_results.csv'), 'w') as csvout:
		writer = csv.writer(csvout)
		row = ['name', 'val acc', 'val sensi', 'val dice', 'val ppv']
		writer.writerow(row)

		for i in range(len(name_total)):
			name = name_total[i]
			if len(name) > 20:
				keyw = 'lidc'
			elif name[0] == 'C':
				keyw = 'exact09'

			row = [name_total[i],float(round(acc_total[i]*100,3)),float(round(sensitivity_total[i]*100,3)),
				   float(round(dice_total[i]*100,3)),float(round(ppv_total[i]*100,3))]
			all_results[keyw].append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
			writer.writerow(row)

		lidc_results = np.mean(np.array(all_results['lidc']),axis = 0)
		exact09_results = np.mean(np.array(all_results['exact09']), axis = 0)

		lidc_results2 = np.std(np.array(all_results['lidc']),axis = 0)
		exact09_results2 = np.std(np.array(all_results['exact09']), axis = 0)

		all_res_mean = np.mean(np.array(all_results['lidc']+all_results['exact09']),axis = 0)
		all_res_std = np.std(np.array(all_results['lidc']+all_results['exact09']),axis = 0)

		lidc_mean = ['lidc mean', lidc_results[0], lidc_results[1], lidc_results[2], lidc_results[3]]
		lidc_std = ['lidc std', lidc_results2[0], lidc_results2[1], lidc_results2[2], lidc_results2[3]]

		ex_mean = ['exact09 mean', exact09_results[0], exact09_results[1], exact09_results[2], exact09_results[3]]
		ex_std = ['exact09 std', exact09_results2[0], exact09_results2[1], exact09_results2[2], exact09_results2[3]]

		all_mean = ['all mean', all_res_mean[0], all_res_mean[1], all_res_mean[2], all_res_mean[3]]
		all_std = ['all std', all_res_std[0], all_res_std[1], all_res_std[2], all_res_std[3]]

		writer.writerow(lidc_mean)
		writer.writerow(lidc_std)
		writer.writerow(ex_mean)
		writer.writerow(ex_std)
		writer.writerow(all_mean)
		writer.writerow(all_std)
		csvout.close()

	mean_dice = np.mean(np.array(dice_total))
	mean_dice_hard = np.mean(np.array(dice_hard_total))
	mean_ppv = np.mean(np.array(ppv_total))
	mean_sensiti = np.mean(np.array(sensitivity_total))
	mean_acc = np.mean(np.array(acc_total))
	mean_loss = np.mean(lossHist)
	print('%s, epoch %d, loss %.4f, accuracy %.4f, sensitivity %.4f, dice %.4f, dice hard %.4f, ppv %.4f, time %3.2f'
		  %(state_str, epoch, mean_loss, mean_acc, mean_sensiti, mean_dice,mean_dice_hard, mean_ppv, endtime-starttime))
	print()
	empty_cache()
	return mean_loss, mean_acc, mean_sensiti, mean_dice, mean_ppv

