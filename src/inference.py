import torch
import numpy as np
import pickle

from utils import get_data


def evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode='CMVAE'):
	model = model.to(device)
	model.eval()
	
	gt_y = []
	pred_y = []
	c_y = []
	gt_x = []
	for i, X in enumerate(dataloader):

		x = X[0]
		y = X[1]
		c = X[2]
		x = x.to(device)
		c = c.to(device)
		
		if numint == 2:
			idx = torch.nonzero(torch.sum(c, axis=0), as_tuple=True)[0]
			c1 = torch.zeros_like(c).to(device)
			c1[:,idx[0]] = 1
			c2 = torch.zeros_like(c).to(device)
			c2[:,idx[1]] = 1
		
		with torch.no_grad():
			if mode=='CMVAE':
				if numint == 1:
					y_hat, _, _, _, _ = model(x, c, c, num_interv=1, temp=temp)
				else: 
					# ############ debug
					# bc, csz = model.c_encode(c1, temp=temp)
					# bc2, csz2 = model.c_encode(c2, temp=temp)
					# mu, var = model.encode(x)
					# z = model.reparametrize(mu, var)
					# zinterv = z * (1. - bc - bc2) + bc * csz.reshape(-1,1)/2 + bc2 * csz2.reshape(-1,1)/2
					# u = (zinterv) @ torch.inverse(torch.eye(model.z_dim).to(model.device) -  torch.triu((model.G), diagonal=1))   
					# y_hat = model.decode(u)
					y_hat, _, _, _, _ = model(x, c1, c2, num_interv=2, temp=temp)	
			elif mode=='CVAE':
				if numint == 1:
					bc, csz = model.c_encode(c, temp=temp)
					z = torch.DoubleTensor(bc.size()).normal_().to(device)
					u = model.dag(z, bc, csz, bc, csz, num_interv=1)
					y_hat = model.decode(u)
				else:
					bc, csz = model.c_encode(c1, temp=temp)
					bc2, csz2 = model.c_encode(c2, temp=temp)
					z = torch.DoubleTensor(bc.size()).normal_().to(device)
					u = model.dag(z, bc, csz, bc2, csz2, num_interv=2)
					y_hat = model.decode(u)	
			elif mode=='CVAE-obs':
				if numint == 1:
					bc, csz = model.c_encode(c, temp=temp)
					mu, var = model.encode(x)
					mu_z = model.reverse_dag(mu, None, None, None, None, num_interv=0)
					z = model.reparametrize(mu_z, var)
					u = model.dag(z, bc, csz, bc, csz, num_interv=1)
					y_hat = model.decode(u)
				else:
					bc, csz = model.c_encode(c1, temp=temp)
					bc2, csz2 = model.c_encode(c2, temp=temp)
					mu, var = model.encode(x)
					mu_z = model.reverse_dag(mu, None, None, None, None, num_interv=0)
					z = model.reparametrize(mu_z, var)
					u = model.dag(z, bc, csz, bc2, csz2, num_interv=2)
					y_hat = model.decode(u)	
			elif mode=='MVAE':
				if numint == 1:
					y_hat, _, _, _ = model(x, c, c, num_interv=1, temp=temp)
				else:
					y_hat, _, _, _ = model(x, c1, c2, num_interv=2, temp=temp)						

		gt_x.append(x.cpu().numpy())
		gt_y.append(y.numpy())
		pred_y.append(y_hat.detach().cpu().numpy())
		c_y.append(c.cpu().numpy())

	gt_x = np.vstack(gt_x)
	gt_y = np.vstack(gt_y)
	pred_y = np.vstack(pred_y)
	c_y = np.vstack(c_y)

	rmse = np.sqrt(np.mean(((pred_y[:] - gt_y[:])**2)) / np.mean(((gt_y[:])**2)))
	signerr = (np.sum(np.sum((np.sign(pred_y) != np.sign(gt_y)))) / gt_y.size)

	return rmse, signerr, gt_y, pred_y, c_y, gt_x


def evaluate_single_leftout(model, path_to_dataloder, device, mode, temp=1):
	with open(f'{path_to_dataloder}/test_data_single_node.pkl', 'rb') as f:
		dataloader = pickle.load(f)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=1, mode=mode)


def evaluate_double(model, path_to_ptbtargets, device, mode, temp=1):
	with open(f'{path_to_ptbtargets}/ptb_targets.pkl', 'rb') as f:
		ptb_targets = pickle.load(f)
	dataloader, _, _, _ = get_data(mode='test', perturb_targets=ptb_targets)

	return evaluate_generated_samples(model, dataloader, device, temp, numint=2, mode=mode)
