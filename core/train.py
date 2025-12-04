"""
Bridging Composite and Real: Towards End-to-end Deep Image Matting [IJCV-2021]
Main train file.

Copyright (c) 2021, Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/JizhiziLi/GFM
Paper link (Arxiv): https://arxiv.org/abs/2010.16188

"""
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import logging
import numpy as np
import datetime
import time
from config import *
from util import *
from evaluate import *
from gfm import GFM
from data import MattingDataset, MattingTransform

class Kong_args():
	def __init__(self):
		self.gpuNums         = 1
		self.nEpochs         = 100
		self.lr              = 0.00001
		self.threads         = 0  ### 8
		self.backbone        = "r34"
		self.rosta           = "TT"
		self.batchSize       = 16    ###  batchsize=`expr $batchsizePerGPU \* $GPUNum`
		self.bg_choice       = "hd"  ### "coco"
		self.fg_generate     = "closed_form"
		self.rssn_denoise    = True
		self.model_save_dir  = "models/trained/kong_train"
		self.logname         = "train_log"
		self.dataset_using   = "AM2K"

		self.ksize			 = 25  ### 原始動物Dataset用25完全沒問題
		self.kong_CROP_SIZE  = [640, 960, 1280]
		self.crop_method     = "ord_LeftTop"

class Rebar_args():
	def __init__(self):
		self.gpuNums         = 1
		self.nEpochs         = 10000
		self.lr              = 0.00001
		self.threads         = 0  ### 8
		self.backbone        = "r34"
		self.rosta           = "TT"
		self.batchSize       = 3    ###  batchsize=`expr $batchsizePerGPU \* $GPUNum`
		self.bg_choice       = "hd"  ### "coco"
		self.fg_generate     = "alpha_blending"
		self.rssn_denoise    = False
		self.model_save_dir  = "models/trained/kong_train"
		self.logname         = "train_log"

		self.dataset_using   = "Rebar"
		self.ksize			 = 25  ### 發現 trimap 完全沒有 白色mask, 只剩 灰色不確定區域 和 黑色不是區域
		self.kong_CROP_SIZE  = [640, 960, 1280]
		self.crop_method     = "ord_LeftTop"

		self.load_pretrained_model = True
		self.checkpoint_path = "models/trained/kong_trainckpt_epoch4000.pth"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2025/12/03/星期三
class Rebar_args_ksize5():
	def __init__(self):
		self.gpuNums         = 1
		self.nEpochs         = 15000
		self.lr              = 0.00001
		self.threads         = 0  ### 8
		self.backbone        = "r34"
		self.rosta           = "TT"
		self.batchSize       = 3    ###  batchsize=`expr $batchsizePerGPU \* $GPUNum`
		self.bg_choice       = "hd"  ### "coco"
		self.fg_generate     = "alpha_blending"
		self.model_save_dir  = "models/trained/kong_train_ksize5/"
		self.logname         = "train_log"


		self.dataset_using   = "Rebar"
		self.ksize 			 = 5  ### 這樣子 trimap 才有 白色區域喔
		self.kong_CROP_SIZE  = [640, 960, 1280]
		self.crop_method     = "ord_LeftTop"

		self.load_pretrained_model = False
		self.checkpoint_path = "models/trained/kong_train_ksize5/ckpt_epoch0.pth"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2025/12/04/星期四
class Rebar_args_ksize5_fixSize():
	def __init__(self):
		self.gpuNums         = 1
		self.nEpochs         = 500
		self.lr              = 0.00001
		self.threads         = 0  ### 8
		self.backbone        = "r34"
		self.rosta           = "TT"
		self.batchSize       = 3    ###  batchsize=`expr $batchsizePerGPU \* $GPUNum`
		self.bg_choice       = "hd"  ### "coco"
		self.fg_generate     = "alpha_blending"
		self.rssn_denoise    = False
		self.model_save_dir  = "models/trained/kong_train_ksize5/"
		self.logname         = "train_log"

		self.dataset_using   = "Rebar"
		self.ksize 			 = 5      ### 這樣子 trimap 才有 白色區域喔
		self.kong_CROP_SIZE  = [320]  ### fixSize就直接設 model 的 input大小好了
		self.crop_method     = "ord_LeftTop"

		self.load_pretrained_model = False
		self.checkpoint_path = "models/trained/kong_train_ksize5/ckpt_epoch0.pth"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### 2025/12/04/星期四
class Rebar_args_ksize5_fixSize_CenterCrop():
	def __init__(self):
		self.gpuNums         = 1
		self.nEpochs         = 500
		self.lr              = 0.00001
		self.threads         = 0  ### 8
		self.backbone        = "r34"
		self.rosta           = "TT"
		self.batchSize       = 3    ###  batchsize=`expr $batchsizePerGPU \* $GPUNum`
		self.bg_choice       = "hd"  ### "coco"
		self.fg_generate     = "alpha_blending"
		self.rssn_denoise    = False
		self.model_save_dir  = "models/trained/kong_train_ksize5/"
		self.logname         = "train_log"

		self.dataset_using   = "Rebar"
		self.ksize 			 = 5      ### 這樣子 trimap 才有 白色區域喔
		self.kong_CROP_SIZE  = [320]  ### fixSize就直接設 model 的 input大小好了
		self.crop_method     = "Center"

		self.load_pretrained_model = False
		self.checkpoint_path = "models/trained/kong_train_ksize5/ckpt_epoch0.pth"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### 2025/12/04/星期四
class Rebar_args_ksize5_HaveSmallSize_CenterCrop():
	def __init__(self):
		self.gpuNums         = 1
		self.nEpochs         = 500
		self.lr              = 0.00001
		self.threads         = 0  ### 8
		self.backbone        = "r34"
		self.rosta           = "TT"
		self.batchSize       = 10    ###  batchsize=`expr $batchsizePerGPU \* $GPUNum`
		self.bg_choice       = "hd"  ### "coco"
		self.fg_generate     = "alpha_blending"
		self.rssn_denoise    = False
		self.model_save_dir  = "models/trained/kong_train_ksize5/"
		self.logname         = "train_log"


		self.dataset_using   = "Rebar"
		self.ksize 			 = 5  ### 這樣子 trimap 才有 白色區域喔
		self.kong_CROP_SIZE  = [kong_size * 40 for kong_size in range(1, 20)]
		self.crop_method     = "center"

		self.load_pretrained_model = True
		self.checkpoint_path = "models/trained/kong_train_ksize5/ckpt_epoch14000.pth"
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### Parsing arguments ######### 
def get_args():
	parser = argparse.ArgumentParser(description='Arguments for the training purpose.')
	# backbone: the backbone of GFM, we provide four backbones - r34, r34_2b, d121 and r101.
	# rosta (Representations of Semantic and Transition areas): we provide three types - TT, FT, and BT. 
	# We also present RIM indicates RoSTa Integration Module.
	# bg_choice: original (ORI-Track), hd (COMP-Track, high resolution background, BG20K), coco (COMP-Track,  MS COCO dataset)
	# fg_generate: the way to generate foregrounds and backgrounds in training, closed_form (needs extra fg/bg generation following closed_form method), alpha_blending (no need for extra fg and bg)
	# rssn_denoise: the flag to use extra desnoie images in RSSN in COMP-Track (hd)
	# model_save_dir: path to save the last checkpoint
	# logname: name of the logging files
	parser.add_argument('--gpuNums', type=int, default=1, help='number of gpus')
	parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for, 500 for ORI-Track and 100 for COMP-Track')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.00001')
	parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
	parser.add_argument('--backbone', type=str, required=False, default='r34',choices=["r34","r34_2b","d121","r101"], help="backbone of GFM")
	parser.add_argument('--rosta', type=str, required=False, default='TT',choices=["TT","FT","BT","RIM"], help="representations of semantic and tarnsition areas")
	parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
	parser.add_argument('--bg_choice', type=str, required=True, choices=["original","hd","coco"], help="background choice for training, ORI-Track (original) and COMP-Track (hd or coco)")
	parser.add_argument('--fg_generate', type=str, required=True, default='alpha_blending', choices=["closed_form","alpha_blending"], help="options to generate the fg bg in training")
	parser.add_argument('--rssn_denoise', action='store_true', help='the flag to use denoise images in RSSN in COMP-Track (hd)')
	parser.add_argument('--model_save_dir', type=str, help="where to save the final model")
	parser.add_argument('--logname', type=str, default='train_log', help="name of the logging file")

	args = parser.parse_args()
	print(args)
	return args

def load_dataset(args):
	train_transform = MattingTransform(args)
	train_set = MattingDataset(args, train_transform)
	train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True)
	return train_loader

def load_model(args):
	print("\t--Model backbone: {}".format(args.backbone))
	print("\t--Model rosta: {}".format(args.rosta))
	print("\t--BG choice: {}".format(args.bg_choice))
	print("\t--FG generate method: {}".format(args.fg_generate))
	print("\t--Denoise choice: {}".format(args.rssn_denoise))
	model = GFM(args).to(args.device)
	start_epoch = 1
	return model, start_epoch

def format_second(secs):
	h = int(secs / 3600)
	m = int((secs % 3600) / 60)
	s = int(secs % 60)
	ss = "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h,m,s)
	return ss    

def train(args, model, optimizer, train_loader, epoch):
	### 這邊chatgpt說 整個丟入 GPU 在 多 worker的情況下可能會出事, 所以我改成下面 item = Variable(item).to(args.device) 這邊 放GPU這樣子
	# model = torch.nn.DataParallel(model).to(args.device)
	model = torch.nn.DataParallel(model)
	model.train()
	t0 = time.time()

	loss_each_epoch=[]
	# print("===============================")
	for iteration, batch in enumerate(train_loader, 1):
		torch.cuda.empty_cache()
		batch_new = []
		for item in batch:
			item = Variable(item).to(args.device)
			batch_new.append(item)
		[ori, mask, fg, bg, trimap, dilation, erosion, dilation_subtraction, erosion_subtraction] = batch_new
		# print("                 ori.min()):", ori.min())  ### 0
		# print("                 ori.max()):", ori.max())  ### 255
		# print("                mask.min()):", mask.min())  ### 0
		# print("                mask.max()):", mask.max())  ### 255
		# print("                  fg.min()):", fg.min())  ### 0
		# print("                  fg.max()):", fg.max())  ### 255
		# print("                  bg.min()):", bg.min())  ### 0
		# print("                  bg.max()):", bg.max())  ### 255
		# print("              trimap.min()):", trimap.min())  ### 0
		# print("              trimap.max()):", trimap.max())  ### 255
		# print("            dilation.min()):", dilation.min())  ### 0
		# print("            dilation.max()):", dilation.max())  ### 255
		# print("             erosion.min()):", erosion.min())  ### 0, 鋼筋太細了
		# print("             erosion.max()):", erosion.max())  ### 0
		# print("dilation_subtraction.min()):", dilation_subtraction.min())  ### 0
		# print("dilation_subtraction.max()):", dilation_subtraction.max())  ### 255
		# print(" erosion_subtraction.min()):", erosion_subtraction.min())  ### 0
		# print(" erosion_subtraction.max()):", erosion_subtraction.max())  ### 255
		optimizer.zero_grad()
		### Predict by the model 
		### And calculate the training losses
		if args.rosta=='RIM':
			predict = model(ori)
			predict_global_tt, predict_local_tt, predict_fusion_tt = predict[0]
			predict_global_ft, predict_local_ft, predict_fusion_ft = predict[1]
			predict_global_bt, predict_local_bt, predict_fusion_bt = predict[1]
			predict_fusion = predict[3]
			
			loss_global_tt = get_crossentropy_loss(3, trimap, predict_global_tt)
			loss_global_bt = get_crossentropy_loss(2, dilation, predict_global_bt)
			loss_global_ft = get_crossentropy_loss(2, erosion, predict_global_ft)

			loss_local_tt = get_alpha_loss(predict_local_tt, mask				 , trimap, args) + get_laplacian_loss(predict_local_tt, mask				, trimap, args)
			loss_local_bt = get_alpha_loss(predict_local_bt, dilation_subtraction, trimap, args) + get_laplacian_loss(predict_local_bt, dilation_subtraction, trimap, args)
			loss_local_ft = get_alpha_loss(predict_local_ft, erosion_subtraction , trimap, args) + get_laplacian_loss(predict_local_ft, erosion_subtraction , trimap, args)

			loss_final_tt = get_alpha_loss_whole_img(predict_fusion_tt, mask, args) + get_laplacian_loss_whole_img(predict_fusion_tt, mask, args) + get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion_tt, args)
			loss_final_bt = get_alpha_loss_whole_img(predict_fusion_bt, mask, args) + get_laplacian_loss_whole_img(predict_fusion_bt, mask, args) + get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion_bt, args)
			loss_final_ft = get_alpha_loss_whole_img(predict_fusion_ft, mask, args) + get_laplacian_loss_whole_img(predict_fusion_ft, mask, args) + get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion_ft, args)

			loss_final    = get_alpha_loss_whole_img(predict_fusion   , mask, args) + get_laplacian_loss_whole_img(predict_fusion   , mask, args) + get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion   , args)

			loss_tt = loss_global_tt+loss_local_tt+loss_final_tt
			loss_bt = loss_global_bt+loss_local_bt+loss_final_bt
			loss_ft = loss_global_ft+loss_local_ft+loss_final_ft
			loss = loss_tt+loss_bt+loss_ft+loss_final
		else:
			predict_global, predict_local, predict_fusion = model(ori)

			if args.rosta=='TT':
				loss_global =get_crossentropy_loss(3, trimap, predict_global)
				loss_local = get_alpha_loss(predict_local, mask  			   , trimap, args) + get_laplacian_loss(predict_local, mask				  , trimap, args)
			elif args.rosta=='FT':
				loss_global =get_crossentropy_loss(2, dilation, predict_global)
				loss_local = get_alpha_loss(predict_local, erosion_subtraction , trimap, args) + get_laplacian_loss(predict_local, erosion_subtraction, trimap, args)
			else:
				loss_global =get_crossentropy_loss(2, mask, predict_global)
				loss_local = get_alpha_loss(predict_local, dilation_subtraction, trimap, args) + get_laplacian_loss(predict_local, dilation_subtraction, trimap, args)

			loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask, args) + get_laplacian_loss_whole_img(predict_fusion, mask, args)
			loss_fusion_comp  = get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion, args)
			loss = 0.25*loss_global+0.25*loss_local+0.25*loss_fusion_alpha+0.25*loss_fusion_comp
		
		loss.backward()
		optimizer.step()

		if iteration !=  0:
			t1 = time.time()
			num_iter = len(train_loader)
			speed = (t1 - t0) / iteration
			exp_time = format_second(speed * (num_iter * (args.nEpochs - epoch + 1) - iteration))          
			loss_each_epoch.append(loss.item())
			if args.rosta=='RIM':
				print("GFM-RIM-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Loss-TT:{:.5f} Loss-FT:{:.5f} Loss-BT:{:.5f} Loss-final:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.item(), loss_tt.item(), loss_ft.item(), loss_bt.item(), loss_final.item(),speed, exp_time))
			else:
				print("GFM-Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} Global:{:.5f} Local:{:.5f} Fusion-alpha:{:.5f} Fusion-comp:{:.5f} Speed:{:.5f}s/iter {}".format(epoch, args.nEpochs, iteration, num_iter, optimizer.param_groups[0]['lr'], loss.item(), loss_global.item(), loss_local.item(), loss_fusion_alpha.item(), loss_fusion_comp.item(),speed, exp_time))
			
def save_last_checkpoint(args, model, optimizer, epoch):
	### 多存 optimizer_state_dict 和 epoch 讓 model 可以 reload繼續訓練
	print('=====> Saving best model',str(args.epoch))
	create_folder_if_not_exists(args.model_save_dir)
	model_out_path = "{}ckpt_epoch{}.pth".format(args.model_save_dir, args.epoch)
	torch.save({'model_state_dict'    : model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'epoch'				  : epoch} , model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))

def main():
	# args = get_args()
	# args = Kong_args()
	# args = Rebar_args()
	# args = Rebar_args_ksize5()  	### 2025/12/03/星期三, 發現 crop的影響真的很大
	args = Rebar_args_ksize5_fixSize()
	# args = Rebar_args_ksize5_fixSize_CenterCrop()
	# args = Rebar_args_ksize5_HaveSmallSize_CenterCrop()
	now = datetime.datetime.now()
	logging_filename = 'logs/train_logs/'+args.logname+'_'+now.strftime("%Y-%m-%d-%H:%M")+'.log'
	print(f'===> Logging to {logging_filename}') 
	logging.basicConfig(filename=logging_filename, level=logging.INFO)
	args.logging = logging
	logging.info("===============================")
	logging.info(f"===> Loading args\n{args}")
	logging.info("===> Environment init")
	# if not torch.cuda.is_available():
	# 	raise Exception("No GPU and cuda available, please try again")
	
	# Check for flag --fg_generate and --rssn_denoise.
	# if you use --rssn_denoise, you have to use fg_generate=closed_form
	if args.rssn_denoise and args.bg_choice=='hd':
		args.fg_generate = 'closed_form'
		if 'FG_DENOISE_PATH' not in  DATASET_PATHS_DICT['AM2K']['TRAIN'] or 'ORIGINAL_DENOISE_PATH' not in DATASET_PATHS_DICT['BG20K']['TRAIN']:
			raise Exception('No FG_DENOISE_PATH in AM2K or ORIGINAL_DENOISE_PATH in BG20k are found in DATASET_PATHS_DICT , please try training without --rssn_denoise instead.')
		elif (not check_if_folder_exists(DATASET_PATHS_DICT['AM2K']['TRAIN']['FG_DENOISE_PATH'])) or (not check_if_folder_exists(DATASET_PATHS_DICT['BG20K']['TRAIN']['ORIGINAL_DENOISE_PATH'])):
			raise Exception('Either FG_DENOISE_PATH in AM2K or ORIGINAL_DENOISE_PATH in BG20k is not exist, found, please try training without --rssn_denoise instead.')
	else:
		args.rssn_denoise=False

	if args.fg_generate=='closed_form':
		if 'FG_PATH' not in DATASET_PATHS_DICT['AM2K']['TRAIN'] or 'BG_PATH' not in DATASET_PATHS_DICT['AM2K']['TRAIN']:
			raise Exception('No FG or BG generated by closed form are found in DATASET_PATHS_DICT, please try training with --fg_generate=alpha_blending instead.')
		elif (not check_if_folder_exists(DATASET_PATHS_DICT['AM2K']['TRAIN']['FG_PATH'])) or (not check_if_folder_exists(DATASET_PATHS_DICT['AM2K']['TRAIN']['BG_PATH'])):
			raise Exception('Either FG_PATH or BG_PATH in AM2K is not exist, please correct or try training with --fg_generate=alpha_blending instead.')

	args.gpuNums = torch.cuda.device_count()
	logging.info(f'Running with GPUs and the number of GPUs: {args.gpuNums}')
	train_loader = load_dataset(args)
	logging.info('===> Building the model')
	model, start_epoch = load_model(args)
	logging.info('===> Initialize optimizer')
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

	### 自己加入的 可以reload上次的結果繼續訓練
	if(args.load_pretrained_model):
		checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
		model    .load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		print(f"Resuming from epoch {start_epoch}")

	now = datetime.datetime.now()
	# training
	for epoch in range(start_epoch, args.nEpochs + 1):
		# print(f'Train on Epoch: {epoch}')
		train(args, model, optimizer, train_loader, epoch)
		args.epoch = epoch
	### 多存 optimizer_state_dict 和 epoch 讓 model 可以 reload繼續訓練
	save_last_checkpoint(args, model, optimizer, epoch)

if __name__ == "__main__":
	start_time = time.time()
	print("start_time:", time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(start_time)))
	main()
	end_time = time.time()
	cost_time = end_time - start_time
	print("start_time:", time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(start_time)))
	print("end_time  :", time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(end_time)))
	sec  = cost_time % 60
	min  = cost_time // 60
	hour = min // 60
	min  = min % 60
	print(f"cost_time :{int(hour)}:{int(min)}:{sec}")