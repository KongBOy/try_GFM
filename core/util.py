"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Utilization file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""

import os
import shutil
import torch
import numpy as np
import cv2
from PIL import Image
from config import *
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
##########################
### Pure functions
##########################
def extract_pure_name(original_name):
	pure_name, extention = os.path.splitext(original_name)
	return pure_name

def listdir_nohidden(path):
	new_list = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			new_list.append(f)
	new_list.sort()
	return new_list

def create_folder_if_not_exists(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def check_if_folder_exists(folder_path):
	return os.path.exists(folder_path)
	

def refresh_folder(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	else:
		shutil.rmtree(folder_path)
		os.makedirs(folder_path)

def generate_composite_img(img, alpha_channel):
	b_channel, g_channel, r_channel = cv2.split(img)
	b_channel = b_channel * alpha_channel
	g_channel = g_channel * alpha_channel
	r_channel = r_channel * alpha_channel
	alpha_channel = (alpha_channel*255).astype(b_channel.dtype)
	img_BGRA = cv2.merge((r_channel,g_channel,b_channel,alpha_channel))
	return img_BGRA

##########################
### Functions for fusion 
##########################
def collaborative_matting(rosta, glance_sigmoid, focus_sigmoid):
	if rosta =='TT':
		values, index = torch.max(glance_sigmoid,1)
		index = index[:,None,:,:].float()
		### index <===> [0, 1, 2]
		### bg_mask <===> [1, 0, 0]
		bg_mask = index.clone()
		bg_mask[bg_mask==2]=1
		bg_mask = 1- bg_mask
		### trimap_mask <===> [0, 1, 0]
		trimap_mask = index.clone()
		trimap_mask[trimap_mask==2]=0
		### fg_mask <===> [0, 0, 1]
		fg_mask = index.clone()
		fg_mask[fg_mask==1]=0
		fg_mask[fg_mask==2]=1
		focus_sigmoid = focus_sigmoid
		trimap_mask = trimap_mask
		fusion_sigmoid = focus_sigmoid*trimap_mask+fg_mask
	elif rosta == 'BT':
		values, index = torch.max(glance_sigmoid,1)
		index = index[:,None,:,:].float()
		fusion_sigmoid = index - focus_sigmoid
		fusion_sigmoid[fusion_sigmoid<0]=0
	else:
		values, index = torch.max(glance_sigmoid,1)
		index = index[:,None,:,:].float()
		fusion_sigmoid = index + focus_sigmoid
		fusion_sigmoid[fusion_sigmoid>1]=1
	fusion_sigmoid = fusion_sigmoid
	return fusion_sigmoid

def get_masked_local_from_global_test(global_result, local_result):
	weighted_global = np.ones(global_result.shape)
	weighted_global[global_result==255] = 0
	weighted_global[global_result==0] = 0
	fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
	return fusion_result
	
def gen_trimap_from_segmap_e2e(segmap):
	trimap = np.argmax(segmap, axis=1)[0]
	trimap = trimap.astype(np.int64)	
	trimap[trimap==1]=128
	trimap[trimap==2]=255
	return trimap.astype(np.uint8)

def gen_bw_from_segmap_e2e(segmap):
	bw = np.argmax(segmap, axis=1)[0]
	bw = bw.astype(np.int64)
	bw[bw==1]=255
	return bw.astype(np.uint8)

def save_test_result(save_dir, predict):
	predict = (predict * 255).astype(np.uint8)
	cv2.imwrite(save_dir, predict)


#######################################
### Function to generate training data
#######################################
def generate_paths_for_dataset(args):
	BG_CHOICE = args.bg_choice
	FG_GENERATE = args.fg_generate
	RSSN_DENOISE = args.rssn_denoise

	if(args.dataset_using == "Rebar"):
		ORI_PATH = DATASET_PATHS_DICT['Rebar']['TRAIN']['ORIGINAL_PATH']
		MASK_PATH = DATASET_PATHS_DICT['Rebar']['TRAIN']['MASK_PATH']
		SAMPLE_BAGS = 1 if BG_CHOICE=='original' else DATASET_PATHS_DICT['Rebar']['TRAIN']['SAMPLE_BAGS']
		FG_PATH = DATASET_PATHS_DICT['Rebar']['TRAIN']['FG_PATH'] if FG_GENERATE=='closed_form' else None 
	elif(args.dataset_using == "AM2K"):
		ORI_PATH = DATASET_PATHS_DICT['AM2K']['TRAIN']['ORIGINAL_PATH']
		MASK_PATH = DATASET_PATHS_DICT['AM2K']['TRAIN']['MASK_PATH']
		SAMPLE_BAGS = 1 if BG_CHOICE=='original' else DATASET_PATHS_DICT['AM2K']['TRAIN']['SAMPLE_BAGS']
		FG_PATH = DATASET_PATHS_DICT['AM2K']['TRAIN']['FG_PATH'] if FG_GENERATE=='closed_form' else None 
	
	if BG_CHOICE=='hd':
		BG_PATH = DATASET_PATHS_DICT['BG20K']['TRAIN']['ORIGINAL_PATH']
		bg_list = listdir_nohidden(BG_PATH)
		if RSSN_DENOISE:
			BG_PATH_DENOISE = DATASET_PATHS_DICT['BG20K']['TRAIN']['ORIGINAL_DENOISE_PATH']
			FG_PATH_DENOISE = DATASET_PATHS_DICT['AM2K']['TRAIN']['FG_DENOISE_PATH']
			bg_list_denoise = listdir_nohidden(BG_PATH_DENOISE)
	elif BG_CHOICE=='coco':
		BG_PATH = DATASET_PATHS_DICT['COCO']['TRAIN']['ORIGINAL_PATH']
		bg_list = listdir_nohidden(BG_PATH)
	elif FG_GENERATE=='closed_form':
		BG_PATH = DATASET_PATHS_DICT['AM2K']['TRAIN']['BG_PATH']

	mask_list = listdir_nohidden(MASK_PATH)
	total_number = len(mask_list) * SAMPLE_BAGS
	paths_list = []

	for mask_name in mask_list:
		path_list = []
		ori_path = ORI_PATH+extract_pure_name(mask_name)+'.jpg'
		mask_path = MASK_PATH+mask_name
		path_list.append(ori_path)
		path_list.append(mask_path)
		fg_path = FG_PATH+mask_name if FG_GENERATE=='closed_form' else None
		path_list.append(fg_path)
		if BG_CHOICE!='original':
			bg_path = BG_PATH+bg_list[mask_list.index(mask_name)]
		elif FG_GENERATE=='closed_form':
			bg_path = BG_PATH+extract_pure_name(mask_name)+'.jpg'
		else:
			bg_path = None
		path_list.append(bg_path)
		if BG_CHOICE =='hd' and RSSN_DENOISE:
			fg_path_denoise = FG_PATH_DENOISE+mask_name
			bg_path_denoise = BG_PATH_DENOISE+bg_list[mask_list.index(mask_name)]
			path_list.append(fg_path_denoise)
			path_list.append(bg_path_denoise)
		paths_list.append(path_list)

	return paths_list

def trim_img(img):
	if img.ndim>2:
		img = img[:,:,0]
	return img

def resize_img(ori, img):
	# 用 skimage resize 做完的結果 會被弄到 0~1, 拉回 0~255
	img = resize(img, ori.shape)*255.0
	return img

def process_fgbg(ori, mask, is_fg, fgbg_path=None):
	if fgbg_path is not None:
		img = np.array(Image.open(fgbg_path))
	else:
		mask_3 = (mask/255.0)[:, :, np.newaxis].astype(np.float32)
		img = ori*mask_3 if is_fg else ori*(1-mask_3)
		# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 8))
		# ax[0].imshow(ori    .astype(np.uint8))
		# ax[1].imshow(mask_3 .astype(np.uint8))
		# ax[2].imshow(img    .astype(np.uint8))
		# ax[3].imshow(img    .astype(np.float32)/255.)
		# fig.tight_layout()
		# print(img.min())
		# print(img.max())
		# plt.show()
	return img

def add_guassian_noise(img, fg, bg):
	row,col,ch= img.shape
	mean = 0
	sigma = 10
	gauss = np.random.normal(mean,sigma,(row,col,ch))
	gauss = gauss.reshape(row,col,ch)
	noisy_img = np.uint8(img + gauss)
	noisy_fg = np.uint8(fg + gauss)
	noisy_bg = np.uint8(bg + gauss)
	return noisy_img, noisy_fg, noisy_bg

def generate_composite_rssn(fg, bg, mask, fg_denoise=None, bg_denoise=None):
	## resize bg accordingly
	h, w, c = fg.shape
	### clone 一份 mask 成 alpha 並把數值縮放到0~1之間
	alpha = np.zeros((h, w, 1), np.float32)
	alpha[:, :, 0] = mask / 255.
	### bg縮放成跟 fg 一樣的大小
	bg = resize_img(fg, bg)
	### 如果有給 fg_denoise 的話, 有50% 的機率 把 fg, bg 替換成 fg_denoise, bg_denoise(縮放成跟fg_denoise一樣大)
	if fg_denoise is not None and random.random()<0.5:
		fg = fg_denoise
		bg = resize_img(fg, bg_denoise)
	### 50% 的機率 把 背景弄糊
	if random.random()<0.5:
		rand_kernel = random.choice([20,30,40,50,60])
		bg = cv2.blur(bg, (rand_kernel,rand_kernel))
	### 前景 + 背景 用 alpha 做合成
	composite = alpha * fg + (1 - alpha) * bg
	composite = composite.astype(np.uint8)
	### 50% 的機率把整張圖做高斯雜訊
	if random.random()<0.5:
		composite, fg, bg = add_guassian_noise(composite, fg, bg)
	return composite, fg, bg

def generate_composite_coco(fg, bg, mask):
	h, w, c = fg.shape
	alpha = np.zeros((h, w, 1), np.float32)
	alpha[:, :, 0] = mask / 255.
	bg = resize_img(fg, bg)
	composite = alpha * fg + (1 - alpha) * bg
	composite = composite.astype(np.uint8)
	return composite, fg, bg


def gen_trimap_with_dilate(alpha, kernel_size):	
	# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
	### 可能是前景的部分 都保留下來 (1~255 的地方弄成 1, 剩下 歸0)
	fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	# ax[0].imshow(fg_and_unknown)
	### 保證是前景的部分 才留下來 (只有255 的地方弄成 1, 剩下 歸0)
	fg = np.array(np.equal(alpha, 255).astype(np.float32))
	# ax[1].imshow(fg)
	### 可能是前景的部分 做膨脹, 大概就是 可能原本可能是前景的部分會不會沒抓的很完整, 把可能的區域再往外擴一些應該就可以擴到全部了
	dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=1)
	### 保證是前景的部分 做侵蝕, 更保證了裡面的東西 一定事前景
	erode = cv2.erode(fg, kernel, iterations=1)
	### 可能是前景的部分再往外擴一些 扣除 更保證是前景的部分, 就留下了 不確定 但很有可能是前景的的區域 囉
	trimap = erode *255 + (dilate-erode)*128
	# ax[2].imshow(trimap)
	# plt.show()
	return trimap.astype(np.uint8)

def gen_dilate(alpha, kernel_size):
	### 同上原理, 可能是前景的部分 都保留下來 做膨脹, 大概就是 可能原本可能是前景的部分會不會沒抓的很完整, 把可能的區域再往外擴一些應該就可以擴到全部了
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
	fg_and_unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	dilate =  cv2.dilate(fg_and_unknown, kernel, iterations=1)*255
	return dilate.astype(np.uint8)

def gen_erosion(alpha, kernel_size): 
	### 同上原理, 保證是前景的部分 才留下來 做侵蝕, 更保證了裡面的東西 一定事前景
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
	fg = np.array(np.equal(alpha, 255).astype(np.float32))
	erode = cv2.erode(fg, kernel, iterations=1)*255
	return erode.astype(np.uint8)
