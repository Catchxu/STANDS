import dgl
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import anndata as ad
from tqdm import tqdm
from typing import Optional, Dict, Union, Any

from .model import Discriminator, KinPair, GeneratorBC, GeneratorAD
from ._utils import select_device, seed_everything, calculate_gradient_penalty


class FindPairs:
    def __init__(self, 
                 n_epochs: int = 1000, 
                 learning_rate: float = 2e-4,
                 GPU: Union[bool, str] = True, 
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):

        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.device = select_device(GPU)

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        self.weight = weight

    def fit(self, generator: GeneratorAD,  raw: Dict[str, Any]):
        '''Find Kin Pairs'''
        tqdm.write('Begin to find Kin Pairs between datasets...')

        raw_g = raw['graph']
        ref_g, tgt_g = self.split(raw_g)
        self.G = generator.to(self.device)

        # freeze self.G weight
        for param in self.G.parameters():
            param.requires_grad = False

        self.init_model(raw, ref_g.num_nodes(), tgt_g.num_nodes())

        self.M.train()
        self.D.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                # generate embeddings
                z_ref = self.G.extract.GeneEncoder(ref_g, ref_g.ndata['gene'])
                z_tgt = self.G.extract.GeneEncoder(tgt_g, tgt_g.ndata['gene'])

                self.UpdateD(z_ref, z_tgt)
                self.UpdateM(z_ref, z_tgt)

                # update learning rate for G and D
                self.D_sch.step()
                self.M_sch.step()
                t.set_postfix(G_Loss = self.M_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        self.M.eval()
        with torch.no_grad():
            z_ref = self.G.extract.GeneEncoder(ref_g, ref_g.ndata['gene'])
            z_tgt = self.G.extract.GeneEncoder(tgt_g, tgt_g.ndata['gene'])
            _, _, m = self.M(z_ref, z_tgt)
            pair_id = list(ref_g.nodes().cpu().numpy()[m.argmax(axis=1)])
            ref_g = dgl.node_subgraph(ref_g, pair_id)
            tgt_g.ndata['ref_gene'] = ref_g.ndata['gene']

        tqdm.write('Kin Pairs have been found.\n')
        return ref_g, tgt_g

    def split(self, graph: dgl.DGLGraph):
        '''Split the integrated graph to reference and target graph'''
        idx = torch.argmax(graph.ndata['batch'], -1).numpy()
        ref_id = list(np.where(idx == 0)[0])
        tgt_id = list(np.where(idx != 0)[0])
        ref_g = dgl.node_subgraph(graph, ref_id).to(self.device)
        tgt_g = dgl.node_subgraph(graph, tgt_id).to(self.device)
        return ref_g, tgt_g

    def init_model(self, raw, n_ref, n_tgt):
        self.M = KinPair(n_ref, n_tgt).to(self.device)
        self.D = Discriminator(raw['gene_dim'], raw['patch_size'], only_ST=True).to(self.device)

        self.opt_M = optim.Adam(self.M.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.M_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_M, T_max=self.n_epochs)
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_D, T_max=self.n_epochs)

        self.Loss = nn.MSELoss().to(self.device)

        # freeze the encoder weights
        for _, value in self.G.named_parameters():
            value.requires_grad = False

    def UpdateD(self, z_ref, z_tgt):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        z_ref, z_tgt, _ = self.M(z_ref, z_tgt)
        d1 = torch.mean(self.D.Zforward(z_ref.detach()))
        d2 = torch.mean(self.D.Zforward(z_tgt.detach()))
        gp = calculate_gradient_penalty(self.D, z_ref.detach(), z_tgt.detach(), Zforward=True)

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()

    def UpdateM(self, z_ref, z_tgt):
        '''Updating mapping matrix'''
        self.opt_M.zero_grad()

        # reconstruct z_tgt with z_ref
        fake_z_tgt, z_tgt, _ = self.M(z_ref, z_tgt)
        d = self.D.Zforward(fake_z_tgt)

        Loss_rec = self.Loss(z_tgt, fake_z_tgt)
        Loss_adv = -torch.mean(d)

        # store generator loss for printing training information and backward
        self.M_loss = self.weight['w_rec']*Loss_rec + self.weight['w_adv']*Loss_adv
        self.M_loss.backward()
        self.opt_M.step()




class BatchAlign:
    def __init__(self, 
                 n_epochs: int = 10, 
                 batch_size: int = 128,
                 learning_rate: float = 3e-4, 
                 n_dis: int = 2,
                 GPU: Union[bool, str] = True, 
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_dis = n_dis
        self.device = select_device(GPU)

        self.seed = random_state
        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        self.weight = weight

    def fit(self, raw: Dict[str, Any], generator: GeneratorAD, **alignerkwargs):
        '''Remove batch effects'''
        adatas = raw['adata']
        adata_ref = adatas[0]
        adata_tgt = ad.concat(adatas[1:])

        # find Kin Pairs
        Aligner = FindPairs(random_state=self.seed, **alignerkwargs)
        _, tgt_g = Aligner.fit(generator, raw)

        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler, batch_size=self.batch_size, 
            shuffle=True, drop_last=False, num_workers=0, device=self.device
        )

        self.init_model(raw, generator)

        tqdm.write('Begin to correct spatial transcriptomics datasets...')
        self.G.train()
        self.D.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, _, blocks in self.dataset:

                    # Update discriminator for n_dis times
                    for _ in range(self.n_dis):
                        self.UpdateD(blocks)

                    # Update generator for one time
                    self.UpdateG(blocks)

                # Update learning rate for G and D
                self.D_sch.step()
                self.G_sch.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
        
        self.dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler, batch_size=self.batch_size, 
            shuffle=False, drop_last=False, num_workers=0, device=self.device
        )

        self.G.eval()
        corrected = []
        with torch.no_grad():
            for _, _, blocks in self.dataset:
                fake_g = self.G.STforward(
                    blocks, blocks[0].srcdata['gene'], blocks[-1].dstdata['batch']
                )
                corrected.append(fake_g.cpu().detach())

        corrected = torch.cat(corrected, dim=0).numpy()
        adata_tgt.X = corrected
        adata = ad.concat([adata_ref, adata_tgt])
        tqdm.write('Datasets have been corrected.\n')
        return adata

    def init_model(self, raw: Dict[str, Any], generator: GeneratorAD):
        z_dim = generator.extract.z_dim
        self.G = GeneratorBC(generator.extract, raw['data_n'], z_dim).to(self.device)
        self.D = Discriminator(raw['gene_dim'], raw['patch_size']).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_G, T_max=self.n_epochs)
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt_D, T_max=self.n_epochs)

        self.L1 = nn.L1Loss().to(self.device)

    def UpdateD(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        # generate fake data
        batchid = blocks[-1].dstdata['batch']
        fake_g = self.G.STforward(blocks, blocks[0].srcdata['gene'], batchid)

        # get real data from blocks
        real_g = blocks[1].dstdata['gene']

        d1 = torch.mean(self.D.SCforward(real_g))
        d2 = torch.mean(self.D.SCforward(fake_g.detach()))
        gp = calculate_gradient_penalty(self.D, real_g, fake_g.detach())         

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()

    def UpdateG(self, blocks):
        '''Updating generator'''
        self.opt_G.zero_grad()

        # generate fake data
        batchid = blocks[-1].dstdata['batch']
        fake_g = self.G.STforward(blocks, blocks[0].srcdata['gene'], batchid)

        # get real data from blocks
        real_g = blocks[1].dstdata['gene']

        # discriminator provides feedback
        d = self.D.SCforward(fake_g)

        Loss_rec = self.L1(real_g, fake_g)
        Loss_adv = - torch.mean(d)

        # store generator loss for printing training information and backward
        self.G_loss = self.weight['w_rec']*Loss_rec + self.weight['w_adv']*Loss_adv
        self.G_loss.backward()
        self.opt_G.step()