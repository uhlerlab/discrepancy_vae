import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


# VAE model with causal layer and mmd loss
# "dim" specifies the sample dimension; "c_dim" specifies the dimension of the intervention encoding.
#  "z_dim" specifies the dimension of the latent space.
class CMVAE(nn.Module):
    def __init__(self, dim, z_dim, c_dim, device=None):

        super(CMVAE, self).__init__()

        if device is None:
            self.cuda = False
            self.device = 'cpu'
        else:
            self.device = device
            self.cuda = True

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # encoder
        hids = 128
        self.fc1 = nn.Linear(self.dim,hids)
        weights_init(self.fc1)
        
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)
        
        # DAG matrix G (upper triangular, z_dim x z_dim). 
        # encoded as a dense matrix, where only upper triangular parts will be used
        self.G = torch.nn.Parameter(torch.normal(0,.1,size = (self.z_dim,self.z_dim)))
        
        # C encoder
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.c_dim))

        # decoder
        self.d1 = nn.Linear(self.z_dim,hids)
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)
        
        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h)) 

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.DoubleTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.DoubleTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, u):        
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))
    
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h)*temp)
        s = c @ self.c_shift
        return h, s
    
    # Causal DAG "layer"
    # bc is a softmax vector encoding the target of the intervetnion
    # csz encodes the strength of the intervention
    def dag(self, z, bc, csz, bc2, csz2, num_interv = 1):
        if num_interv == 0:
            u = (z) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu((self.G), diagonal=1))
        else:
            if num_interv == 1: # 1 - bc
                zinterv = z * (1.) + bc * csz.reshape(-1,1)
            else: # 1. - bc - bc2
                zinterv = z * (1.) + bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
            
            u = (zinterv) @ torch.inverse(torch.eye(self.z_dim).to(self.device) -  torch.triu((self.G), diagonal=1))     
        return u

    def forward(self, x, c, c2, num_interv = 1, temp = 1):
        assert num_interv in [0,1,2], "support single- or double-node interventions only"

        # decode an interventional sample from an observational sample    
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        
        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # create the reconstruction of observational sample
        u_recon = self.dag(z, bc*0, csz*0, bc*0, csz*0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return y_hat, x_recon, mu, var, self.G


# Baseline models
# conditional VAE with causal layer but no mmd loss
class CVAE(nn.Module):
    def __init__(self, dim, z_dim, c_dim, device=None):

        super(CVAE, self).__init__()

        if device is None:
            self.cuda = False
            self.device = 'cpu'
        else:
            self.device = device
            self.cuda = True

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # encoder
        hids = 128
        self.fc1 = nn.Linear(self.dim,hids)
        weights_init(self.fc1)
        
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)
        
        # DAG matrix G (upper triangular, z_dim x z_dim). 
        # encoded as a dense matrix, where only upper triangular parts will be used
        self.G = torch.nn.Parameter(torch.normal(0,.1,size = (self.z_dim,self.z_dim)))
        
        # C encoder
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.c_dim))

        # decoder
        self.d1 = nn.Linear(self.z_dim,hids)
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)
        
        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h)) 

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.DoubleTensor(std.size()).normal_().to(self.device)
        else:

            eps = torch.DoubleTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, u):        
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))
    
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h)*temp)
        s = c @ self.c_shift
        return h, s
    
    # Causal DAG "layer"
    # bc is a softmax vector encoding the target of the intervetnion
    # csz encodes the strength of the intervention
    def dag(self, z, bc, csz, bc2, csz2, num_interv = 1):
        if num_interv == 0:
            u = (z) @ torch.inverse(torch.eye(self.z_dim).to(self.device) - torch.triu((self.G), diagonal=1))
        else:
            if num_interv == 1:
                zinterv = z * (1.) + bc * csz.reshape(-1,1)
            else:
                zinterv = z * (1.) + bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
            
            u = (zinterv) @ torch.inverse(torch.eye(self.z_dim).to(self.device) -  torch.triu((self.G), diagonal=1))     
        return u

    def reverse_dag(self, mu, bc, csz, bc2, csz2, num_interv = 1):
        mu_z = (mu) @ (torch.eye(self.z_dim).to(self.device) - torch.triu((self.G), diagonal=1))
        if num_interv == 1:
            mu_z += bc * csz.reshape(-1,1)
        elif num_interv == 2:
            mu_z += bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
        return mu_z

    def forward(self, x, c, c2, num_interv = 1, temp = 1):
        assert num_interv in [0,1,2], "support single- or double-node interventions only"

        # decode an interventional sample from an observational sample
        if num_interv:    
            bc, csz = self.c_encode(c, temp)       
            bc2, csz2 = self.c_encode(c2, temp)
        else:
            bc = None
            csz = None
            bc2 = None
            csz2 = None
        
        # get mean and variance of exogenous z
        mu, var = self.encode(x)
        mu_z = self.reverse_dag(mu, bc, csz, bc2, csz2, num_interv)

        # sample interventional z
        z = self.reparametrize(mu_z, var)
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
    
        x_recon = self.decode(u)
        
        return x_recon, mu_z, var, self.G


# discrepancy vae without causal layer
class MVAE(nn.Module):
    def __init__(self, dim, z_dim, c_dim, device=None):

        super(MVAE, self).__init__()

        if device is None:
            self.cuda = False
            self.device = 'cpu'
        else:
            self.device = device
            self.cuda = True

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # encoder
        hids = 128
        self.fc1 = nn.Linear(self.dim,hids)
        weights_init(self.fc1)
        
        self.fc_mean = nn.Linear(hids, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(hids, z_dim)
        weights_init(self.fc_var)
        
        # C encoder
        self.c1 = nn.Linear(self.c_dim, hids)
        self.c2 = nn.Linear(hids, self.z_dim)
        self.c_shift = nn.Parameter(torch.ones(self.c_dim))

        # decoder
        self.d1 = nn.Linear(self.z_dim,hids)
        self.d2 = nn.Linear(hids, self.dim)
        weights_init(self.d1)
        weights_init(self.d2)
        
        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.leakyrelu(self.fc1(x))
        return self.fc_mean(h), F.softplus(self.fc_var(h)) 

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.DoubleTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.DoubleTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, u):        
        h = self.leakyrelu(self.d1(u))
        return self.leakyrelu(self.d2(h))
    
    def c_encode(self, c, temp=1):
        h = self.leakyrelu(self.c1(c))
        h = self.sftmx(self.c2(h)*temp)
        s = c @ self.c_shift
        return h, s
    
    # latent layer, direct additive instead of using a DAG
    def latent(self, z, bc, csz, bc2, csz2, num_interv = 1):
        if num_interv == 0:
            u = z
        else:
            if num_interv == 1:
                zinterv = z * (1.) + bc * csz.reshape(-1,1)
            else:
                zinterv = z * (1.) + bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1)
            
            u = zinterv   
        return u

    def forward(self, x, c, c2, num_interv = 1, temp = 1):
        assert num_interv in [0,1,2], "support single- or double-node interventions only"

        # decode an interventional sample from an observational sample    
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        
        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        u = self.latent(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # create the reconstruction of observational sample
        u_recon = self.latent(z, bc*0, csz*0, bc*0, csz*0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return y_hat, x_recon, mu, var


# VAE for simulation
class CMVAE_simu(nn.Module):
    def __init__(self, dim, z_dim, c_dim, nonlinear, order, device=None):

        super(CMVAE_simu, self).__init__()

        if device is None:
            self.cuda = False
            self.device = 'cpu'
        else:
            self.device = device
            self.cuda = True

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim
        self.nonlinear = nonlinear

        # encoder
        self.fc_mean = nn.Linear(self.dim, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(self.dim, z_dim)
        weights_init(self.fc_var)
        
        # DAG matrix G (upper triangular, z_dim x z_dim). 
        # encoded as a dense matrix, where only upper triangular parts will be used
        self.G = torch.nn.Parameter(torch.normal(0,.1,size = (self.z_dim,self.z_dim)))
        
        # C encoder
        # self.c1 = nn.Linear(self.c_dim, self.z_dim)
        # weights_init(self.c1)
        self.c_shift = nn.Parameter(torch.ones(self.c_dim))
        self.order = order

        # decoder
        self.d1 = nn.Linear(self.z_dim, self.dim)
        weights_init(self.d1)
        
        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sftmx = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        return self.fc_mean(x), F.softplus(self.fc_var(x)) 

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.DoubleTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.DoubleTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, u): 
        return self.d1(u)
    
    def c_encode(self, c, temp=1):
        # h = self.sftmx(self.c1(c)*temp)
        s = c @ self.c_shift
        # return h, s ( torch.ones_like(c[:,0]))
        return c[:,self.order], s
    
    # Causal DAG "layer"
    # bc is a softmax vector encoding the target of the intervetnion
    # csz encodes the strength of the intervention
    def dag(self, z, bc, csz, bc2, csz2, num_interv = 1):
        if num_interv == 0:
            zinterv = z
        else:
            if num_interv == 1:
                zinterv = z * (1.)+ (bc * csz.reshape(-1,1))
            else: 
                zinterv = z * (1.) + (bc * csz.reshape(-1,1) + bc2 * csz2.reshape(-1,1))
        if self.nonlinear:
            u = torch.zeros_like(zinterv)
            for i in range(self.z_dim):
                u[:,i] = zinterv[:,i].clone() + self.relu(u[:,:i].clone()) @ (torch.triu((self.G), diagonal=1))[:i,i]
        else:
            u = (zinterv) @ torch.inverse(torch.eye(self.z_dim).to(self.device) -  torch.triu((self.G), diagonal=1))     
        return u

    def forward(self, x, c, c2, num_interv = 1, temp = 1):
        assert num_interv in [0,1,2], "support single- or double-node interventions only"

        # decode an interventional sample from an observational sample    
        bc, csz = self.c_encode(c, temp)       
        bc2, csz2 = self.c_encode(c2, temp)
        
        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        u = self.dag(z, bc, csz, bc2, csz2, num_interv)
    
        y_hat = self.decode(u)
        
        # create the reconstruction of observational sample
        u_recon = self.dag(z, bc*0, csz*0, bc*0, csz*0, num_interv=0)
        x_recon = self.decode(u_recon)
        
        return y_hat, x_recon, mu, var, self.G



def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        truncated_normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


def truncated_normal_(tensor, mean=0, std=0.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
