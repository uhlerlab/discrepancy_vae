import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc


# read the norman dataset.
# map the target genes of each cell to a binary vector, using a target gene list "perturb_targets".
# "perturb_type" specifies whether the returned object contains single trarget-gene samples, double target-gene samples, or both.
class SCDataset(Dataset):
    def __init__(self, datafile='/home/jzhang/discrepancy_vae/identifiable_causal_vae/data/datasets/Norman2019_raw.h5ad', perturb_type='single', perturb_targets=None):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double', 'both'], 'perturb_type not supported!'

        adata = sc.read_h5ad(datafile)

        if perturb_targets is None:
            ptb_targets = list(set().union(*[set(i.split(',')) for i in adata.obs['guide_ids'].value_counts().index]))
            ptb_targets.remove('')
        else:
            ptb_targets = perturb_targets
        self.ptb_targets = ptb_targets

        if perturb_type == 'single':
            ptb_adata = adata[(~adata.obs['guide_ids'].str.contains(',')) & (adata.obs['guide_ids']!='')].copy()
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
            del ptb_adata
        elif perturb_type == 'double':
            ptb_adata = adata[adata.obs['guide_ids'].str.contains(',')].copy()
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
            del ptb_adata     
        else:
            ptb_adata = adata[adata.obs['guide_ids']!=''].copy() 
            self.ptb_samples = ptb_adata.X
            self.ptb_names = ptb_adata.obs['guide_ids'].values
            self.ptb_ids = map_ptb_features(ptb_targets, ptb_adata.obs['guide_ids'].values)
            del ptb_adata

        self.ctrl_samples = adata[adata.obs['guide_ids']==''].X.copy()
        self.rand_ctrl_samples = self.ctrl_samples[
            np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
            ]
        del adata

    def __getitem__(self, item):
        x = torch.from_numpy(self.rand_ctrl_samples[item].toarray().flatten()).double()
        y = torch.from_numpy(self.ptb_samples[item].toarray().flatten()).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        return x, y, c
    
    def __len__(self):
        return self.ptb_samples.shape[0]


# read simulation dataset
class SimuDataset(Dataset):
    def __init__(self, datafile='/home/jzhang/discrepancy_vae/identifiable_causal_vae/data/simulation/data_1.pkl', perturb_type='single', perturb_targets=None):
        super(Dataset, self).__init__()
        assert perturb_type in ['single', 'double'], 'perturb_type not supported!'

        with open(datafile, 'rb') as f:
            dataset = pickle.load(f)

        if perturb_targets is None:
            ptb_targets = dataset['ptb_targets']
        else:
            ptb_targets = perturb_targets
        self.ptb_targets = ptb_targets

        
        ptb_data = dataset[perturb_type]
        self.ctrl_samples = ptb_data['X']
        self.ptb_samples = ptb_data['Xc']
        self.ptb_names = np.array(ptb_data['ptbs'])
        self.ptb_ids = map_ptb_features(ptb_targets, ptb_data['ptbs'])
        del ptb_data 

        self.nonlinear = dataset['nonlinear']
        del dataset

    def __getitem__(self, item):
        x = torch.from_numpy(self.ctrl_samples[item].flatten()).double()
        y = torch.from_numpy(self.ptb_samples[item].flatten()).double()
        c = torch.from_numpy(self.ptb_ids[item]).double()
        return x, y, c
    
    def __len__(self):
        return self.ptb_samples.shape[0]



def map_ptb_features(all_ptb_targets, ptb_ids):
    ptb_features = []
    for id in ptb_ids:
        feature = np.zeros(all_ptb_targets.__len__())
        feature[[all_ptb_targets.index(i) for i in id.split(',')]] = 1
        ptb_features.append(feature)
    return np.vstack(ptb_features)



