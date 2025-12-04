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
	def __init__(self, args):
		super(MattingTransform, self).__init__()
		self.args = args

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

		''' 在 trimap == 128 的地方 隨機 crop 影像出來訓練, 原始code 隨機點挑出來後當左上角往右下角crop, 我是覺得 隨機點挑出來當中心點往左右上下crop會更好 '''
		### CROP_SIZE 是一個 [], 比如 [640, 960, 1280], 這邊試想隨機從裡面挑一個 size出來
		rand_ind = random.randint(0, len(self.args.kong_CROP_SIZE) - 1)
		crop_size = self.args.kong_CROP_SIZE[rand_ind] if self.args.kong_CROP_SIZE[rand_ind]<min(h, w) else 320

		### 丟進 model 的大小, 原始code在config.py 裡面設320
		resize_size = RESIZE_SIZE
		### 可以crop的範圍圈出來
		trimap = argv[4]
		if  (self.args.crop_method == "ord_LeftTop"): trimap_crop = trimap[ 			     : h - crop_size      ,     	        : w - crop_size      ]
		elif(self.args.crop_method == "center"     ): trimap_crop = trimap[ crop_size // 2  : h - crop_size // 2 , crop_size // 2  : w - crop_size // 2 ]

		### 找 trimap == 128 的地方 的座標點, random 的從裡面選一個點 來 crop影像
		target = np.where(trimap_crop == 128) # if random.random() < 1.0 else np.where(trimap_crop > -100)
		### 如果 trimap 沒有 == 128的地方, 那就整個 可以crop的範圍內選隨機一點
		if len(target[0])==0:
			target = np.where(trimap_crop > -100)
		rand_ind = np.random.randint(len(target[0]), size = 1)[0]

		### 隨機選的這一點 設定為左上角 或者為 中心點 為起點,  可以用原版的 左上角當起點往右下角drop, 或者我覺得比較合理的 中心點為起點往上下左右crop
		if  (self.args.crop_method == "ord_LeftTop"): cropx, cropy = target[1][rand_ind] 				   , target[0][rand_ind] 
		elif(self.args.crop_method == "center"     ): cropx, cropy = target[1][rand_ind] + crop_size // 2 , target[0][rand_ind] + crop_size // 2 

		### 0.5的機率左右翻轉
		flip_flag=True if random.random()<0.5 else False

		### 實際去 crop影像 和 翻轉, 最後 resize 成 320
		argv_transform = []
		for item in argv:
			if  (self.args.crop_method == "ord_LeftTop"): item = item[cropy 				   : cropy + crop_size      , cropx 				  : cropx + crop_size	   ]
			elif(self.args.crop_method == "center"     ): item = item[cropy - crop_size // 2  : cropy + crop_size // 2 , cropx - crop_size // 2  : cropx + crop_size // 2 ]
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
		self.args = args
		self.samples=[]
		self.transform = transform
		self.logging = args.logging
		self.BG_CHOICE = args.bg_choice
		self.backbone = args.backbone
		self.FG_CF = True if args.fg_generate=='closed_form' else False
		self.RSSN_DENOISE = args.rssn_denoise
		
		print('===> Loading training set')
		self.samples += generate_paths_for_dataset(args)
		print(f"\t--crop_size: {self.args.kong_CROP_SIZE} | resize: {RESIZE_SIZE}")
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
		kernel_size_tt = self.args.ksize
		kernel_size_ftbt = self.args.ksize * 2
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
