import pickle
import numpy as np
import random
from itertools import combinations

np.random.seed(1234)
random.seed(1234)

dim = 10
latdim = 5 
Gp = np.diag(np.ones(latdim - 1), 1) # True graph
sz = 5 # size of intervention
nz = np.diag(np.ones(latdim)) # size of noise
nonlinear = 1
samples = 1024

# generate data
tform = np.eye(latdim, dim)

# single node
x = np.zeros((2*samples*latdim,dim))
xc = np.zeros((2*samples*latdim,dim))
c = ['' for _ in range(2*samples*latdim)]

for j in range(latdim):
	# no intervention
	latentz = np.matmul(np.random.randn(2*samples,latdim), nz)
	latent = np.zeros((2*samples,latdim))
	for i in range(latdim):
		if i == 0:
			latent[:,i] = latentz[:,i]
		else:
			if nonlinear:
				latent[:,i] = latentz[:,i] + np.maximum(latent[:,:i],0) @ (Gp)[:i,i]
			else:
				latent[:,i] = latentz[:,i] + latent[:,:i] @ (Gp)[:i,i]

	# intervention
	latentz_new =  np.matmul(np.random.randn(2*samples,latdim), nz)
	latentc = np.zeros((2*samples,latdim))
	intervSz = np.zeros((2*samples,1))
	for i in range(latdim):
		if i == (j%latdim):
			intervSz[:,0] = sz*np.ones((2*samples)) 
			if nonlinear:
				latentc[:,i] = (latentz_new[:,i] + np.maximum(latentc[:,:i],0) @ (Gp)[:i,i]) + intervSz[:,0]
			else:
				latentc[:,i] = (latentz_new[:,i] + latentc[:,:i] @ (Gp)[:i,i]) + intervSz[:,0]
		else:
			if nonlinear:
				latentc[:,i] = latentz_new[:,i] + np.maximum(latentc[:,:i], 0) @ (Gp)[:i,i]
			else:
				latentc[:,i] = latentz_new[:,i] + latentc[:,:i] @ (Gp)[:i,i]
    

	x[(j)*2*samples:(j+1)*2*samples,:] = latent @ tform
	xc[(j)*2*samples:(j+1)*2*samples,:] = latentc @ tform
	c[(j)*2*samples:(j+1)*2*samples] = [str(j) for _ in range(2*samples)]


# double node
x_d = np.zeros((samples*latdim*(latdim-1)//2,dim))
xc_d = np.zeros((samples*latdim*(latdim-1)//2,dim))
c_d = ['' for _ in range(samples*latdim*(latdim-1)//2)]

for enum,(j,jj) in enumerate(combinations(range(latdim),2)):
	# no intervention
	latentz = np.matmul(np.random.randn(samples,latdim), nz)
	latent = np.zeros((samples,latdim))
	for i in range(latdim):
		if i == 0:
			latent[:,i] = latentz[:,i]
		else:
			if nonlinear:
				latent[:,i] = latentz[:,i] + np.maximum(latent[:,:i], 0) @ (Gp)[:i,i]
			else:
				latent[:,i] = latentz[:,i] + latent[:,:i] @ (Gp)[:i,i]

	# intervention
	latentz_new = np.matmul(np.random.randn(samples,latdim), nz)
	latentc = np.zeros((samples,latdim))
	intervSz = np.zeros((samples,1))
	for i in range(latdim):
		if i == (j%latdim) or i==(jj%latdim):
			intervSz[:,0] = sz*np.ones((samples)) 
			if nonlinear:
				latentc[:,i] = (latentz_new[:,i] + np.maximum(latentc[:,:i], 0) @ (Gp)[:i,i]) + intervSz[:,0]
			else:
				latentc[:,i] = (latentz_new[:,i] + latentc[:,:i] @ (Gp)[:i,i]) + intervSz[:,0]
		else:
			if nonlinear:
				latentc[:,i] = latentz_new[:,i] + np.maximum(latentc[:,:i], 0) @ (Gp)[:i,i]
			else:
				latentc[:,i] = latentz_new[:,i] + latentc[:,:i] @ (Gp)[:i,i]
    


	x_d[(enum)*samples:(enum+1)*samples,:] = latent @ tform
	xc_d[(enum)*samples:(enum+1)*samples,:] = latentc @ tform
	c_d[(enum)*samples:(enum+1)*samples] = [str(j)+','+str(jj) for _ in range(samples)]

dataset = {}
dataset['ptb_targets'] = [str(i) for i in range(latdim)]
dataset['single'] = {'X':x, 'Xc':xc, 'ptbs':c}
dataset['double'] = {'X':x_d, 'Xc':xc_d, 'ptbs':c_d}
dataset['nonlinear'] = nonlinear

with open(f'./data_{nonlinear}.pkl', 'wb') as f:
	pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)