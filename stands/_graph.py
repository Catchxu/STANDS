import dgl
import torch
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from PIL import Image
from typing import Optional, List
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors


class Build_graph:
    def __init__(self, adata: ad.AnnData, image: Optional[np.ndarray],
                 position: np.ndarray, n_neighbors: int = 4,
                 patch_size: int = 48, train_mode: bool = True):
        self.adata = adata
        self.image = image
        self.position = position
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        u, v = self.get_edge()
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['gene'] = self.get_gene()
        if self.image is not None:
            self.g.ndata['patch'] = self.get_patch()

    def get_edge(self):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        nbrs = nbrs.fit(self.position)
        _, indices = nbrs.kneighbors(self.position)
        u = indices[:, 0].repeat(self.n_neighbors)
        v = indices[:, 1:].flatten()
        return u, v

    def get_patch(self):
        if not isinstance(self.image[0, 0, 0], np.uint8):
            self.image = np.uint8(self.image * 255)
            
        img = Image.fromarray(self.image)
        r = np.ceil(self.patch_size/2).astype(int)

        trans = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180)
        ])
 
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        p_list = []
        for i in range(len(self.position)):
            x, y = self.position[i, :]
            p = img.crop((x - r, y - r, x + r, y + r))
            if self.train_mode:
                p = trans(p)
            p = preprocess(p)
            p_list.append(p.reshape(3, 2*r, 2*r))
        return torch.stack(p_list)

    def get_gene(self):
        return torch.Tensor(self.adata.X)

    def pack(self):
        self.dim = self.g.ndata['gene'].shape[1]
        self.use_image = True if self.image is not None else False
        g_pack = {
            'adata': self.adata,
            'graph': self.g,
            'gene_dim': self.dim,
            'patch_size': self.patch_size,
            'use_image': self.use_image
        }
        return g_pack


class Build_multi_graph:
    def __init__(self, adata: List[ad.AnnData], image: Optional[List[np.ndarray]],
                 position: List[np.ndarray], n_neighbors: int = 4,
                 patch_size: int = 48, train_mode: bool = True):
        warnings.filterwarnings("ignore")
        self.adata = adata
        self.adata_raw = adata
        self.image = image
        self.position = position
        self.n_dataset = len(adata)
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.train_mode = train_mode

        self.batch = self.get_batch()
        u, v = self.get_edge()
        self.g = dgl.to_bidirected(dgl.graph((u, v)))
        self.g = dgl.add_self_loop(self.g)

        self.g.ndata['batch'] = self.batch
        self.g.ndata['gene'] = self.get_gene()
        if self.image is not None:
            self.g.ndata['patch'] = self.get_patch()

    def get_batch(self):
        adata = []
        for i in range(self.n_dataset):
            a = self.adata[i]
            a.obs['batch'] = i
            adata.append(a)
        self.adata = ad.concat(adata, merge='same')
        self.adata.obs_names_make_unique(join=',')
        batch = np.array(pd.get_dummies(self.adata.obs['batch']), dtype=np.float32)
        return torch.Tensor(batch)

    def get_edge(self):
        self.adata.obs['idx'] = range(self.adata.n_obs)
        u_list, v_list = [], []
        for i in range(self.n_dataset):
            adata = self.adata[self.adata.obs['batch'] == i]
            position = self.position[i]
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1)
            nbrs = nbrs.fit(position)
            _, indices = nbrs.kneighbors(position)
            u = adata.obs['idx'][indices[:, 0].repeat(self.n_neighbors)]
            v = adata.obs['idx'][indices[:, 1:].flatten()]
            u_list = u_list + u.tolist()
            v_list = v_list + v.tolist()
        return u_list, v_list

    def get_patch(self):
        p_list = []
        for i in range(self.n_dataset):
            img = self.image[i]
            if not isinstance(img[0, 0, 0], np.uint8):
                img = np.uint8(img * 255)
            img = Image.fromarray(img)

            position = self.position[i]
            r = np.ceil(self.patch_size/2).astype(int)

            trans = transforms.Compose([
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180)
            ])
            
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            for i in range(len(position)):
                x, y = position[i, :]
                p = img.crop((x - r, y - r, x + r, y + r))
                if self.train_mode:
                    p = trans(p)
                p = preprocess(p)
                p_list.append(p.reshape(3, 2*r, 2*r))
        return torch.stack(p_list)
    
    def get_gene(self):
        return torch.Tensor(self.adata.X)

    def pack(self):
        self.dim = self.g.ndata['gene'].shape[1]
        self.use_image = True if self.image is not None else False
        g_pack = {
            'adata': self.adata_raw,
            'graph': self.g,
            'gene_dim': self.dim,
            'patch_size': self.patch_size,
            'data_n': self.n_dataset,
            'use_image': self.use_image
        }
        return g_pack