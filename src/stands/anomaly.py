import os
import dgl
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict, Union, Any
from sklearn.preprocessing import LabelEncoder

from .model import GeneratorAD, Discriminator
from ._utils import select_device, seed_everything


class AnomalyDetect:
    def __init__(self, 
                 n_epochs: int = 10, 
                 batch_size: int = 128,
                 learning_rate: float = 3e-4, 
                 GPU: Union[bool, str] = True,
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.device = select_device(GPU)

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        self.weight = weight

    def fit(self, ref: Dict[str, Any], only_ST: bool = False, weight_dir: Optional[str] = None):
        '''Train STANDS on reference graph'''
        tqdm.write('Begin to train the model on reference datasets...')

        # dataset provides subgraph for training
        ref_g = ref['graph']
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            ref_g, ref_g.nodes(), self.sampler, batch_size=self.batch_size, 
            shuffle=True, drop_last=True, num_workers=0, device=self.device
        )

        self.only_ST = only_ST
        self.init_model(ref, weight_dir)

        self.G.train()
        self.D.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, _, blocks in self.dataset:
                    self.UpdateD(blocks)
                    self.UpdateG(blocks)
                
                # Update learning rate for G and D
                self.D_sch.step()
                self.G_sch.step()

                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
        
        tqdm.write('Training has been finished.')

    def init_model(self, ref, weight_dir):
        self.G = GeneratorAD(ref['gene_dim'], ref['patch_size'], self.only_ST).to(self.device)
        self.D = Discriminator(self.G.z_dim, self.G.z_dim, self.only_ST).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_G, T_max=self.n_epochs)
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_D, T_max=self.n_epochs)

        self.L1 = nn.L1Loss().to(self.device)

        self.init_weight(weight_dir)

    @torch.no_grad()
    def init_weight(self, weight_dir):
        '''Initial stage for pretrained weights and memory block'''
        self.G.extract.load_weight(weight_dir)

        # Initial the memory block with the normal embeddings
        sum_t = self.G.Memory.mem_dim/self.batch_size
        t = 0
        while t < sum_t:
            for _, _, blocks in self.dataset:
                if self.only_ST:
                    real_g = blocks[0].srcdata['gene']
                    z, _ = self.G.STforward(blocks, real_g)
                else:
                    real_g = blocks[0].srcdata['gene']
                    real_p = blocks[1].srcdata['patch']
                    z, _, _ = self.G(blocks, real_g, real_p)

                self.G.Memory.update_mem(z)
                t += 1
    
    def UpdateD(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        if self.only_ST:
            # generate fake data
            real_z_g, fake_g = self.G.STforward(blocks, blocks[0].srcdata['gene'])
                    
            # get real data from blocks
            real_g = blocks[1].dstdata['gene']

            d1 = torch.mean(self.D.geneforward(real_g, real_p))
            d2 = torch.mean(self.D(fake_g.detach(), fake_p.detach()))



    def UpdateG(self, blocks):