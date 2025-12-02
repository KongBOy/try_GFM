"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Dataset processing.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""

from config import *
from util import *
import torch
import cv2
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import logging
import pickle
from torchvision import transforms
from torch.autograd import Variable
from skimage.transform import resize
import matplotlib.pyplot as plt
#########################
## Data transformer
#########################
class MattingTransform(object):
	def __init__(self):
		super(MattingTransform, self).__init__()

	def __call__(self, *argv):
		# fig, ax = plt.subplots(ncols=9, nrows=1, figsize=(18, 4))
		# ax[0].imshow(argv[0].astype(np.uint8))  ### ori
		# ax[1].imshow(argv[1].astype(np.uint8))  ### mask
		# ax[2].imshow(argv[2].astype(np.uint8))  ### fg, 
		# ax[3].imshow(argv[3].astype(np.uint8))  ### bg, 
		# ax[4].imshow(argv[4].astype(np.uint8))  ### trimap, 
		# ax[5].imshow(argv[5].astype(np.uint8))  ### dilation, 
		# ax[6].imshow(argv[6].astype(np.uint8))  ### erosion, 
		# ax[7].imshow(argv[7].astype(np.uint8))  ### dilation_subtraction, 
		# ax[8].imshow(argv[8].astype(np.uint8))  ### erosion_subtraction
		# fig.tight_layout()
		# plt.show()
		ori = argv[0]
		h, w, c = ori.shape
		rand_ind = random.randint(0, len(CROP_SIZE) - 1)
		crop_size = CROP_SIZE[rand_ind] if CROP_SIZE[rand_ind]<min(h, w) else 320
		resize_size = RESIZE_SIZE
		### generate crop centered in transition area randomly
		trimap = argv[1]
		trimap_crop = trimap[:h-crop_size, :w-crop_size]
		target = np.where(trimap_crop == 128) if random.random() < 0.5 else np.where(trimap_crop > -100)
		if len(target[0])==0:
			target = np.where(trimap_crop > -100)

		rand_ind = np.random.randint(len(target[0]), size = 1)[0]
		cropx, cropy = target[1][rand_ind], target[0][rand_ind]
		# # flip the samples randomly
		flip_flag=True if random.random()<0.5 else False
		# generate samples (crop, flip, resize)
		argv_transform = []
		for item in argv:
			item = item[cropy:cropy+crop_size, cropx:cropx+crop_size]
			if flip_flag:
				item = cv2.flip(item, 1)
			item = cv2.resize(item, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
			argv_transform.append(item)
		# fig, ax = plt.subplots(ncols=9, nrows=1, figsize=(18, 4))
		# ax[0].imshow(argv_transform[0].astype(np.uint8))  ### ori
		# ax[1].imshow(argv_transform[1].astype(np.uint8))  ### mask
		# ax[2].imshow(argv_transform[2].astype(np.uint8))  ### fg, 
		# ax[3].imshow(argv_transform[3].astype(np.uint8))  ### bg, 
		# ax[4].imshow(argv_transform[4].astype(np.uint8))  ### trimap, 
		# ax[5].imshow(argv_transform[5].astype(np.uint8))  ### dilation, 
		# ax[6].imshow(argv_transform[6].astype(np.uint8))  ### erosion, 
		# ax[7].imshow(argv_transform[7].astype(np.uint8))  ### dilation_subtraction, 
		# ax[8].imshow(argv_transform[8].astype(np.uint8))  ### erosion_subtraction
		# fig.tight_layout()
		# plt.show()
		return argv_transform

#########################
## Data Loader
#########################
class MattingDataset(torch.utils.data.Dataset):
	def __init__(self, args, transform):
		
		self.samples=[]
		self.transform = transform
		self.logging = args.logging
		self.BG_CHOICE = args.bg_choice
		self.backbone = args.backbone
		self.FG_CF = True if args.fg_generate=='closed_form' else False
		self.RSSN_DENOISE = args.rssn_denoise
		
		print('===> Loading training set')
		self.samples += generate_paths_for_dataset(args)
		print(f"\t--crop_size: {CROP_SIZE} | resize: {RESIZE_SIZE}")
		print("\t--Valid Samples: {}".format(len(self.samples)))

	def __getitem__(self,index):
		# Prepare training sample paths
		ori_path = self.samples[index][0]
		mask_path = self.samples[index][1]
		fg_path = self.samples[index][2] if self.FG_CF else None
		bg_path = self.samples[index][3] if (self.FG_CF or self.BG_CHOICE!='original') else None
		fg_path_denoise = self.samples[index][4] if (self.BG_CHOICE=='hd' and self.RSSN_DENOISE) else None
		bg_path_denoise = self.samples[index][5] if (self.BG_CHOICE=='hd' and self.RSSN_DENOISE) else None
		# Prepare ori/mask/fg/bg (mandatary)
		ori = np.array(Image.open(ori_path))
		mask = trim_img(np.array(Image.open(mask_path)))
		fg = process_fgbg(ori, mask, True, fg_path)
		bg = process_fgbg(ori, mask, False, bg_path)
		# fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 8))
		# ax[0].imshow(fg.astype(np.uint8))
		# ax[1].imshow(bg.astype(np.uint8))
		# plt.tight_layout()
		# plt.show()
		# Prepare composite for hd/coco
		if self.BG_CHOICE == 'hd':
			fg_denoise = process_fgbg(ori, mask, True, fg_path_denoise) if self.RSSN_DENOISE else None
			bg_denoise = process_fgbg(ori, mask, True, bg_path_denoise) if self.RSSN_DENOISE else None
			### 把 背景 resize成 前景大小, 50% 機率替換成denoise版本, 50% 機率背景模糊, 50% 機率加上高斯雜訊
			ori, fg, bg = generate_composite_rssn(fg, bg, mask, fg_denoise, bg_denoise)
		elif self.BG_CHOICE == 'coco':
			ori, fg, bg = generate_composite_coco(fg, bg, mask)
		# Generate trimap/dilation/erosion online
		kernel_size_tt = 25
		kernel_size_ftbt = 50
		trimap = gen_trimap_with_dilate(mask, kernel_size_tt)
		dilation = gen_dilate(mask, kernel_size_ftbt)
		erosion = gen_erosion(mask, kernel_size_ftbt)
		dilation_subtraction = dilation-mask
		erosion_subtraction = mask-erosion
		# Data transformation to generate samples
		# crop/flip/resize
		# fig, ax = plt.subplots(ncols=9, nrows=1, figsize=(18, 4))
		# ax[0].imshow(ori                 .astype(np.uint8))  ### ori
		# ax[1].imshow(mask                .astype(np.uint8))  ### mask
		# ax[2].imshow(fg                  .astype(np.uint8))  ### fg, 
		# ax[3].imshow(bg                  .astype(np.uint8))  ### bg, 
		# ax[4].imshow(trimap              .astype(np.uint8))  ### trimap, 
		# ax[5].imshow(dilation            .astype(np.uint8))  ### dilation, 
		# ax[6].imshow(erosion             .astype(np.uint8))  ### erosion, 
		# ax[7].imshow(dilation_subtraction.astype(np.uint8))  ### dilation_subtraction, 
		# ax[8].imshow(erosion_subtraction .astype(np.uint8))  ### erosion_subtraction
		# fig.tight_layout()
		# plt.show()
		argv = self.transform(ori, mask, fg, bg, trimap, dilation, erosion, dilation_subtraction, erosion_subtraction)
		argv_transform = []
		for item in argv:
			if item.ndim<3:
				item = torch.from_numpy(item.astype(np.float32)[np.newaxis, :, :])
			else:
				item = torch.from_numpy(item.astype(np.float32)).permute(2, 0, 1)
			argv_transform.append(item)

		[ori, mask, fg, bg, trimap, dilation, erosion, dilation_subtraction, erosion_subtraction] = argv_transform
		return ori, mask, fg, bg, trimap, dilation, erosion, dilation_subtraction, erosion_subtraction

	def __len__(self):
		return len(self.samples)
