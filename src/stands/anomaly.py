import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict, Union, Any

from .model import GeneratorAD, Discriminator
from .model import GMMWithPrior
from ._utils import select_device, seed_everything, calculate_gradient_penalty


class AnomalyDetect:
    def __init__(self, 
                 n_epochs: int = 10, 
                 batch_size: int = 128,
                 learning_rate: float = 2e-4,
                 n_dis: int = 3,
                 GPU: Union[bool, str] = True,
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_dis = n_dis
        self.device = select_device(GPU)

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            weight = {'w_rec': 50, 'w_adv': 1, 'w_gp': 10}
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
        
        tqdm.write('Training has been finished.')
    
    @torch.no_grad()
    def predict(self, tgt: Dict[str, Any], run_gmm: bool = True):
        '''Detect anomalous spots on target graph'''

        tgt_g = tgt['graph']
        dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler, batch_size=self.batch_size, 
            shuffle=False, drop_last=False, num_workers=0, device=self.device
        )

        self.G.eval()
        self.D.eval()
        tqdm.write('Detect anomalous spots on target dataset...')

        ref_score = self.score(self.dataset)
        tgt_score = self.score(dataset)

        tqdm.write('Anomalous spots have been detected.\n')

        if run_gmm:
            gmm = GMMWithPrior(ref_score)
            threshold = gmm.fit(tgt_score=tgt_score)
            tgt_label = [1 if s >= threshold else 0 for s in tgt_score]
            return tgt_score, tgt_label
        else:
            return tgt_score

    def init_model(self, ref, weight_dir):
        self.G = GeneratorAD(ref['gene_dim'], ref['patch_size'], self.only_ST).to(self.device)
        self.D = Discriminator(ref['gene_dim'], ref['patch_size'], self.only_ST).to(self.device)

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
                    z, _, _ = self.G.Fullforward(blocks, real_g, real_p)

                self.G.Memory.update_mem(z)
                t += 1
    
    def UpdateD(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        if self.only_ST:
            # generate fake data
            _, fake_g = self.G.STforward(blocks, blocks[0].srcdata['gene'])
                    
            # get real data from blocks
            real_g = blocks[1].dstdata['gene']

            d1 = torch.mean(self.D.SCforward(real_g))
            d2 = torch.mean(self.D.SCforward(fake_g.detach()))
            gp = calculate_gradient_penalty(self.D, real_g, fake_g.detach())

        else:
            _, fake_g, fake_p = self.G.Fullforward(
                blocks, blocks[0].srcdata['gene'], blocks[1].srcdata['patch']
            )

            # get real data from blocks
            real_g = blocks[1].dstdata['gene']
            real_p = blocks[1].dstdata['patch']

            d1 = torch.mean(self.D.Fullforward(real_g, real_p))
            d2 = torch.mean(self.D.Fullforward(fake_g.detach(), fake_p.detach()))
            gp = calculate_gradient_penalty(
                self.D, real_g, fake_g.detach(), real_p, fake_p.detach()
            )            

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()

    def UpdateG(self, blocks):
        '''Updating generator'''
        self.opt_G.zero_grad()

        if self.only_ST:
            # generate fake data
            z, fake_g = self.G.STforward(blocks, blocks[0].srcdata['gene'])
                    
            # get real data from blocks
            real_g = blocks[1].dstdata['gene']

            # discriminator provides feedback
            d = self.D.SCforward(fake_g)

            Loss_rec = self.L1(real_g, fake_g)
            Loss_adv = - torch.mean(d)

        else:
            z, fake_g, fake_p = self.G.Fullforward(
                blocks, blocks[0].srcdata['gene'], blocks[1].srcdata['patch']
            )

            # get real data from blocks
            real_g = blocks[1].dstdata['gene']
            real_p = blocks[1].dstdata['patch']

            # discriminator provides feedback
            d = self.D.Fullforward(fake_g, fake_p)

            Loss_rec = (self.L1(real_g, fake_g)+self.L1(real_p, fake_p))/2
            Loss_adv = - torch.mean(d)
        
        # store generator loss for printing training information and backward
        self.G_loss = self.weight['w_rec'] * Loss_rec + self.weight['w_adv'] * Loss_adv
        self.G_loss.backward()
        self.opt_G.step()

        # updating memory block with generated embeddings, fake_z
        self.G.Memory.update_mem(z)

    def score(self, dataset):
        # calucate anomaly score
        dis = []
        for _, _, blocks in dataset:
            if self.only_ST:
                # generate fake data
                _, fake_g = self.G.STforward(blocks, blocks[0].srcdata['gene'])
                d = self.D.SCforward(blocks[1].dstdata['gene'])
                d_hat = self.D.SCforward(fake_g.detach())

            else:
                _, fake_g, fake_p = self.G.Fullforward(
                    blocks, blocks[0].srcdata['gene'], blocks[1].srcdata['patch']
                )
                d = self.D.Fullforward(blocks[1].dstdata['gene'], blocks[1].dstdata['patch'])
                d_hat = self.D.Fullforward(fake_g.detach(), fake_p.detach())
            
            cos_sim = F.cosine_similarity(d, d_hat, dim=1)
            dis.append(cos_sim.cpu().detach())

        # Normalize anomaly scores
        dis = torch.cat(dis, dim=0).numpy()
        score = (dis.max() - dis)/(dis.max() - dis.min())

        score = list(score.reshape(-1))
        return score