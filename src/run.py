import os
import argparse
from argparse import Namespace
import time
import json
import pickle

import torch
import numpy as np
import random

from train import train, train_CVAE, train_MVAE
from utils import get_data


def main(args):
	print(f'using device: {args.device}')

	opts = Namespace(
		batch_size = 32,
		mode = 'train',
		lr = 1e-3,
		epochs = 100,
		grad_clip = False,
		mxAlpha = 10,
		mxBeta = 2,
		mxTemp = 5,
		lmbda = 1e-3,
		MMD_sigma = 1000,
		kernel_num = 10,
		matched_IO = False,
		latdim = 105,
		seed = 12
	)

	torch.manual_seed(opts.seed)
	np.random.seed(opts.seed)
	random.seed(opts.seed)

	dataloader, dataloader2, dim, cdim, ptb_targets = get_data(batch_size=opts.batch_size, mode=opts.mode)

	opts.dim = dim
	if opts.latdim is None:
		opts.latdim = cdim
	opts.cdim = cdim

	with open(f'{args.savedir}/config.json', 'w') as f:
		json.dump(opts.__dict__, f, indent=4)

	with open(f'{args.savedir}/ptb_targets.pkl', 'wb') as f:
		pickle.dump(ptb_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(f'{args.savedir}/test_data_single_node.pkl', 'wb') as f:
		pickle.dump(dataloader2, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(f'{args.savedir}/train_data.pkl', 'wb') as f:
		pickle.dump(dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)

	if args.model == 'cmvae':
		train(dataloader, opts, args.device, args.savedir, log=True)
	elif args.model == 'cvae':
		train_CVAE(dataloader, opts, args.device, args.savedir, log=True)
	elif args.model == 'mvae':
		train_MVAE(dataloader, opts, args.device, args.savedir, log=True) 


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parse args')
	parser.add_argument('-s', '--savedir', type=str, default='../result/', help='directory to save the results')
	parser.add_argument('--device', type=str, default=None, help='device to run the training')
	parser.add_argument('--model', type=str, default='cmvae', help='model to run the training')
	args = parser.parse_args()
	
	args.savedir = args.savedir + f'run{int(time.time())}'
	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

	main(args)
