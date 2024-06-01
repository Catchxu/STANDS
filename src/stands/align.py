import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Union, Any

from .model import Discriminator, KinPair
from ._utils import select_device, seed_everything, calculate_gradient_penalty


class FindPairs:
    def __init__(self, 
                 n_epochs: int = 2000, 
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

    def fit(self, generator: nn.Module,  raw: Dict[str, Any]):
        '''Find kin pairs'''
        tqdm.write('Begin to find kin pairs between datasets...')

        raw_g = raw['graph']
        ref_g, tgt_g = self.split(raw_g)
        self.G = generator

        self.init_model(raw, ref_g.num_nodes(), tgt_g.num_nodes())

        self.M.train()
        self.D.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                # generate embeddings
                z_ref = self.G.extract.encode(ref_g, ref_g[0].srcdata['gene'])
                z_tgt = self.G.extract.encode(tgt_g, tgt_g[0].srcdata['gene'])

                self.UpdateD(z_ref, z_tgt)
                self.UpdateM(z_ref, z_tgt)

                # update learning rate for G and D
                self.D_sch.step()
                self.M_sch.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)
        
        self.G.eval()
        with torch.no_grad():
            _, _, m = self.G(ref_g, tgt_g)
            pair_id = list(ref_g.nodes().cpu().numpy()[m.argmax(axis=1)])
            ref_g = dgl.node_subgraph(ref_g, pair_id)
            tgt_g.ndata['ref_gene'] = ref_g.ndata['gene']

        tqdm.write('Kin pairs have been found.\n')
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
        self.D = Discriminator(raw['gene_dim'], raw['patch_size']).to(self.device)

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
        gp = calculate_gradient_penalty(self.D, z_ref.detach(), z_tgt.detach())

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()

    def UpdateM(self, z_ref, z_tgt):
        '''Updating mapping matrix'''
        self.opt_M.zero_grad()

        # reconstruct z_ref with z_tgt
        fake_z_ref, z_tgt, _ = self.M(z_ref, z_tgt)
        d = self.D.Zforward(fake_z_ref)

        Loss_rec = self.Loss(z_ref, fake_z_ref)
        Loss_adv = -torch.mean(d)

        # store generator loss for printing training information and backward
        self.M_loss = self.weight['w_rec']*Loss_rec + self.weight['w_adv']*Loss_adv
        self.M_loss.backward()
        self.opt_M.step()

