import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from copy import deepcopy
import numpy as np
import os

from model import CMVAE, CVAE, MVAE, CMVAE_simu
from utils import MMD_loss


# fit CMVAE to data
def train(
    dataloader,
    opts,
    device,
    savedir,
    log,
    simu=False,
    order=None,
    nonlinear=False,
    ):

    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    if simu:
        cmvae = CMVAE_simu(
            dim = opts.dim,
            z_dim = opts.latdim,
            c_dim = opts.cdim,
            nonlinear=nonlinear,
            order=order,
            device = device
        ) 
    else:      
        cmvae = CMVAE(
            dim = opts.dim,
            z_dim = opts.latdim,
            c_dim = opts.cdim,
            device = device
        )
    cmvae.double()
    cmvae.to(device)

    optimizer = torch.optim.Adam(params=cmvae.parameters(), lr=opts.lr)

    cmvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))

    ## Loss parameters
    beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
    beta_schedule[:10] = 0
    beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
    alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
    alpha_schedule[:] = opts.mxAlpha
    alpha_schedule[:5] = 0
    alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
    alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

    ## Softmax temperature 
    temp_schedule = torch.ones(opts.epochs)
    temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)

    min_train_loss = np.inf
    best_model = deepcopy(cmvae)
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        L1Av = 0
        for (i, X) in enumerate(dataloader):
            x = X[0]
            y = X[1]
            c = X[2]
            
            if cmvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                
            optimizer.zero_grad()
            y_hat, x_recon, z_mu, z_var, G = cmvae(x, c, c, num_interv=1, temp=temp_schedule[n])
            mmd_loss, recon_loss, kl_loss, L1 = loss_function(y_hat, y, x_recon, x, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss + opts.lmbda*L1
            loss.backward()
            if opts.grad_clip:
                for param in cmvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()
            L1Av += L1.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})

        if simu:
            if n % 10 == 0:
                print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct)+','+'L1='+str(L1Av/ct))
        else:
            print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})

        if (mmdAv + reconAv + klAv + L1Av)/ct < min_train_loss:
            min_train_loss = (mmdAv + reconAv + klAv + L1Av)/ct 
            best_model = deepcopy(cmvae)
            torch.save(best_model, os.path.join(savedir, 'best_model.pt'))

    last_model = deepcopy(cmvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))


# fit CVAE baseline to data
def train_CVAE(
    dataloader,
    opts,
    device,
    savedir,
    log
    ):

    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    cvae = CVAE(
        dim = opts.dim,
        z_dim = opts.latdim,
        c_dim = opts.cdim,
        device = device
    )
    cvae.double()
    cvae.to(device)

    optimizer = torch.optim.Adam(params=cvae.parameters(), lr=opts.lr)

    cvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))

    ## Loss parameters
    beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
    beta_schedule[:10] = 0
    beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
    alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
    alpha_schedule[:] = opts.mxAlpha
    alpha_schedule[:5] = 0
    alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
    alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

    ## Softmax temperature 
    temp_schedule = torch.ones(opts.epochs)
    temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)

    min_train_loss = np.inf
    best_model = deepcopy(cvae)
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        reconAv = 0
        i_reconAv = 0
        klAv = 0
        for (i, X) in tqdm(enumerate(dataloader)):
            x = X[0]
            y = X[1]
            c = X[2]
            
            if cvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                
            optimizer.zero_grad()

            y_recon, z_mu, z_var, G = cvae(y, c, c, num_interv=torch.sum(c[0]), temp=temp_schedule[n])
            _, i_recon_loss, kl_loss, L1 = loss_function(None, None, y_recon, y, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss = alpha_schedule[n]*i_recon_loss + beta_schedule[n]*kl_loss/2 + opts.lmbda*L1/2

            x_recon, z_mu, z_var, G = cvae(x, None, None, 0, temp=temp_schedule[n])
            _, recon_loss, kl_loss, L1 = loss_function(None, None, x_recon, x, z_mu, z_var, G, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss += recon_loss + beta_schedule[n]*kl_loss/2 + opts.lmbda*L1/2
            
            loss.backward()
            if opts.grad_clip:
                for param in cvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            i_reconAv += i_recon_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'i_recon_loss':i_recon_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})

        print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'I_MSE='+str(i_reconAv/ct )+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg i_recon_loss': i_reconAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})

        if (i_reconAv+reconAv + klAv)/ct < min_train_loss:
            min_train_loss = (i_reconAv + reconAv + klAv)/ct 
            best_model = deepcopy(cvae)
            torch.save(best_model, os.path.join(savedir, 'best_model.pt'))

    last_model = deepcopy(cvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))


# fit MVAE to data
def train_MVAE(
    dataloader,
    opts,
    device,
    savedir,
    log
    ):

    if log:
        wandb.init(project='cmvae', name=savedir.split('/')[-1])  

    mvae = MVAE(
        dim = opts.dim,
        z_dim = opts.latdim,
        c_dim = opts.cdim,
        device = device
    )
    mvae.double()
    mvae.to(device)

    optimizer = torch.optim.Adam(params=mvae.parameters(), lr=opts.lr)

    mvae.train()
    print("Training for {} epochs...".format(str(opts.epochs)))

    ## Loss parameters
    beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
    beta_schedule[:10] = 0
    beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10) 
    alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
    alpha_schedule[:] = opts.mxAlpha
    alpha_schedule[:5] = 0
    alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
    alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

    ## Softmax temperature 
    temp_schedule = torch.ones(opts.epochs)
    temp_schedule[5:] = torch.linspace(1, opts.mxTemp, opts.epochs-5)

    min_train_loss = np.inf
    best_model = deepcopy(mvae)
    for n in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        for (i, X) in tqdm(enumerate(dataloader)):
            x = X[0]
            y = X[1]
            c = X[2]
            
            if mvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c = c.to(device)
                
            optimizer.zero_grad()
            y_hat, x_recon, z_mu, z_var = mvae(x, c, c, num_interv=1, temp=temp_schedule[n])
            mmd_loss, recon_loss, kl_loss, _ = loss_function(y_hat, y, x_recon, x, z_mu, z_var, None, opts.MMD_sigma, opts.kernel_num, opts.matched_IO)
            loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss
            loss.backward()
            if opts.grad_clip:
                for param in mvae.parameters():
                    if param.grad is not None:
                        param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

            ct += 1
            lossAv += loss.detach().cpu().numpy()
            mmdAv += mmd_loss.detach().cpu().numpy()
            reconAv += recon_loss.detach().cpu().numpy()
            klAv += kl_loss.detach().cpu().numpy()

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})

        print('Epoch '+str(n)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})

        if (mmdAv + reconAv + klAv)/ct < min_train_loss:
            min_train_loss = (mmdAv + reconAv + klAv)/ct 
            best_model = deepcopy(mvae)
            torch.save(best_model, os.path.join(savedir, 'best_model.pt'))

    last_model = deepcopy(mvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))



# loss function definition
def loss_function(y_hat, y, x_recon, x, mu, var, G, MMD_sigma, kernel_num, matched_IO=False):

    if not matched_IO:
        matching_function_interv = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num) # MMD Distance since we don't have paired data
    else:
        matching_function_interv = nn.MSELoss() # MSE if there is matched interv/observ samples
    matching_function_recon = nn.MSELoss() # reconstruction

    if y_hat is None:
        MMD = 0
    else:
        MMD = matching_function_interv(y_hat, y)
    MSE = matching_function_recon(x_recon, x)
    logvar = torch.log(var)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)/x.shape[0]
    if G is None:
        L1 = 0
    else:
        L1 = torch.norm(torch.triu(G,diagonal=1),1)  # L1 norm for sparse G
    return MMD, MSE, KLD, L1



