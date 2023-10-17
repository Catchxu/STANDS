import os
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict
from torch.nn import functional as F

from .model import Generator_AD, Generator_Pair, Generator_BC
from .model import Discriminator
from ._utils import seed_everything, calculate_gradient_penalty


class ADNet:
    def __init__(self, n_epochs: int = 10, batch_size: int = 128,
                 learning_rate: float = 2e-5, mem_dim: int = 1024,
                 shrink_thres: float = 0.01, temperature: float = 1,
                 n_critic: int = 2, GPU: bool = True,
                 random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train STAND.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres
        self.tem = temperature
        self.n_critic = n_critic

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def fit(self, ref_g: dgl.DGLGraph, weight_dir: Optional[str] = None, save: bool = True, **kwargs):
        '''Fine-tune STand on reference graph'''
        tqdm.write('Begin to fine-tune the model on normal spots...')

        # dataset provides subgraph for training
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            ref_g, ref_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=4, device=self.device)

        self.in_dim = ref_g.ndata['gene'].shape[1]
        self.patch_size = ref_g.ndata['patch'].shape[2]
        self.D = Discriminator(self.patch_size, self.in_dim).to(self.device)
        self.G = Generator_AD(self.patch_size, self.in_dim, thres=self.shrink_thres,
                              mem_dim=self.mem_dim, tem=self.tem, **kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr*self.n_critic, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_epochs)
        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_G,
                                                          T_max = self.n_epochs)

        self.L1 = nn.L1Loss().to(self.device)

        self.prepare(weight_dir)

        self.D.train()
        self.G.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, _, blocks in self.dataset:

                    # Update discriminator for 'n_critic' times
                    for _ in range(self.n_critic):
                        self.update_D(blocks)

                    # Update generator for one time
                    self.update_G(blocks)

                # Update learning rate for G and D
                self.D_sch.step()
                self.G_sch.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        tqdm.write('Fine-tuning has been finished.')

        if save:
            if 'patch' in ref_g.ndata.keys():
                save_module = ['GeneEncoder', 'GeneDecoder',
                               'ImageEncoder', 'ImageDecoder',
                               'Fusion']
            else:
                save_module = ['GeneEncoder', 'GeneDecoder']

            weight_dir = os.path.dirname(__file__) + '/temp.pth'
            self.G.save_weights(weight_dir, save_module)  # save the trained STNet weights
    
    @torch.no_grad()
    def predict(self, tgt_g: dgl.DGLGraph):
        '''Detect anomalous spots on target graph'''
        if (self.G is None or self.D is None):
            raise RuntimeError('Please fine-tune the model first.')

        dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=4, device=self.device)

        self.G.eval()
        self.D.eval()
        tqdm.write('Detect anomalous spots on test dataset...')
    
        # calucate anomaly score
        dis = []
        for _, _, blocks in dataset:
            # get real data from blocks
            real_g = blocks[0].srcdata['gene']
            real_p = blocks[1].srcdata['patch']
            _, fake_g, fake_p = self.G(blocks, real_g, real_p)

            d = self.D(fake_g, fake_p)
            dis.append(d.cpu().detach())

        # Normalize anomaly scores
        dis = torch.mean(torch.cat(dis, dim=0), dim=1).numpy()
        # dis = torch.max(torch.cat(dis, dim=0), dim=1).values.numpy()
        score = (dis.max() - dis)/(dis.max() - dis.min())

        tqdm.write('Anomalous spots have been detected.\n')
        return list(score.reshape(-1))

    @torch.no_grad()
    def prepare(self, weight_dir: Optional[str]):
        '''Prepare stage for pretrained weights and memory block'''
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')

        # Load the pre-trained weights for Encoder and Decoder
        model_dict = self.G.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.G.load_state_dict(model_dict)

        # Initial the memory block with the normal embeddings
        sum_t = self.mem_dim/self.batch_size
        t = 0
        while t < sum_t:
            for _, _, blocks in self.dataset:
                real_g = blocks[0].srcdata['gene']
                real_p = blocks[1].srcdata['patch']
                z, _, _ = self.G(blocks, real_g, real_p)
                self.G.Memory.update_mem(z)
                t += 1
    
    def update_D(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        # generate fake data
        _, fake_g, fake_p = self.G(blocks,
                                   blocks[0].srcdata['gene'],
                                   blocks[1].srcdata['patch'])

        # get real data from blocks
        real_g = blocks[1].dstdata['gene']
        real_p = blocks[1].dstdata['patch']

        d1 = torch.mean(self.D(real_g, real_p))
        d2 = torch.mean(self.D(fake_g.detach(), fake_p.detach()))
        gp = calculate_gradient_penalty(self.D, real_g, fake_g.detach(), real_p, fake_p.detach())

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_scaler.scale(self.D_loss).backward()

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()

    def update_G(self, blocks):
        '''Updating generator'''
        self.opt_G.zero_grad()

        # get real data from blocks
        real_g = blocks[0].srcdata['gene']
        real_p = blocks[1].srcdata['patch']
        real_z, fake_g, fake_p = self.G(blocks, real_g, real_p)

        # discriminator provides feedback
        d = self.D(fake_g, fake_p)

        Loss_rec = (self.L1(blocks[-1].dstdata['gene'], fake_g) + 
                    self.L1(blocks[-1].dstdata['patch'], fake_p))/2
        Loss_adv = -torch.mean(d)

        # store generator loss for printing training information and backward
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.G_scaler.scale(self.G_loss).backward()
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()

        # updating memory block with generated embeddings, fake_z
        self.G.Memory.update_mem(real_z)


class AlignNet:
    def __init__(self, n_epochs: int = 500, learning_rate: float = 2e-4,
                 GPU: bool = True, random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train STAND.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        
        self.n_epochs = n_epochs
        self.lr = learning_rate

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight
    
    def fit(self, raw_g: dgl.DGLGraph, weight_dir: Optional[str] = None, **kwargs):
        '''Find spot pairs '''
        tqdm.write('Begin to find spot pairs between datasets...')

        ref_g, tgt_g = self.unpack(raw_g)

        self.in_dim = ref_g.ndata['gene'].shape[1]
        self.patch_size = ref_g.ndata['patch'].shape[2]
        self.D = Discriminator(self.patch_size, 256).to(self.device)
        self.G = Generator_Pair(self.patch_size, self.in_dim,
                                ref_g.num_nodes(), tgt_g.num_nodes(), **kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_epochs)
        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_G,
                                                          T_max = self.n_epochs)

        self.Loss = nn.MSELoss().to(self.device)

        self.prepare(weight_dir)

        self.D.train()
        self.G.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                self.update_D(ref_g, tgt_g)
                self.update_G(ref_g, tgt_g)

                # Update learning rate for G and D
                self.D_sch.step()
                self.G_sch.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        self.G.eval()
        with torch.no_grad():
            _, _, m = self.G(ref_g, tgt_g)
            pair_id = list(ref_g.nodes().cpu().numpy()[m.argmax(axis=1)])
            ref_g = dgl.node_subgraph(ref_g, pair_id)
            tgt_g.ndata['ref_gene'] = ref_g.ndata['gene']

        tqdm.write('Spot pairs have been found.')
        return ref_g, tgt_g, m.shape[1]  # m.shape[1], the number of batches

    def unpack(self, graph):
        '''Unpack the integrated graph to reference and target graph'''
        idx = torch.argmax(graph.ndata['batch'], -1).numpy()
        ref_id = list(np.where(idx == 0)[0])
        tgt_id = list(np.where(idx != 0)[0])
        ref_g = dgl.node_subgraph(graph, ref_id).to(self.device)
        tgt_g = dgl.node_subgraph(graph, tgt_id).to(self.device)
        return ref_g, tgt_g

    def prepare(self, weight_dir: Optional[str]):
        '''Prepare stage for pretrained weights'''
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            if os.path.exists(os.path.dirname(__file__) + '/temp.pth'):
                pre_weights = torch.load(os.path.dirname(__file__) + '/temp.pth')
            else:
                pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')

        # Load the pre-trained weights for Encoder and Decoder
        model_dict = self.G.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.G.load_state_dict(model_dict)

        # freeze the encoder weights
        for _, value in self.G.GeneEncoder.named_parameters():
            value.requires_grad = False
        for _, value in self.G.ImageEncoder.named_parameters():
            value.requires_grad = False

    def update_D(self, ref_g, tgt_g):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        z_ref, z_tgt, _ = self.G(ref_g, tgt_g)
        d1 = torch.mean(self.D(z_ref.detach()))
        d2 = torch.mean(self.D(z_tgt.detach()))
        gp = calculate_gradient_penalty(self.D, z_ref.detach(), z_tgt.detach())

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_scaler.scale(self.D_loss).backward()

        self.D_scaler.step(self.opt_D)
        self.D_scaler.update()
    
    def update_G(self, ref_g, tgt_g):
        '''Updating generator'''
        self.opt_G.zero_grad()

        z_ref, z_tgt, _ = self.G(ref_g, tgt_g)
        d = self.D(z_tgt.detach())

        Loss_rec = self.Loss(z_ref, z_tgt)
        Loss_adv = -torch.mean(d)

        # store generator loss for printing training information and backward
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)

        self.G_scaler.scale(self.G_loss).backward()
        self.G_scaler.step(self.opt_G)
        self.G_scaler.update()


class BCNet:
    def __init__(self, n_epochs: int = 10, batch_size: int = 128,
                 learning_rate: float = 2e-5, n_critic: int = 2,
                 GPU: bool = True, random_state: Optional[int] = None,
                 weight: Optional[Dict[str, float]] = None):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train STAND.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_critic = n_critic

        if random_state is not None:
            self.seed = random_state
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight
    
    def fit(self, raw_g: dgl.DGLGraph, weight_dir: Optional[str] = None,
            n_epochs: int = 500, learning_rate: float = 2e-4, **kwargs):
        '''Remove batch effects'''
        Aligner = AlignNet(n_epochs, learning_rate, random_state=self.seed)
        ref_g, tgt_g, data_n = Aligner.fit(raw_g)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=True,
            drop_last=False, num_workers=4, device=self.device)
        
        self.in_dim = ref_g.ndata['gene'].shape[1]
        self.patch_size = ref_g.ndata['patch'].shape[2]
        self.D = Discriminator(self.patch_size, 256).to(self.device)
        self.G = Generator_BC(data_n, self.patch_size, self.in_dim, **kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.D_scaler = torch.cuda.amp.GradScaler()
        self.G_scaler = torch.cuda.amp.GradScaler()
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_epochs)
        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_G,
                                                          T_max = self.n_epochs)
        
        self.Loss = nn.MSELoss().to(self.device)

        self.prepare(weight_dir)

        tqdm.write('Begin to correct spatial transcriptomics datasets...')
        self.D.train()
        self.G.train()
        with tqdm(total=self.n_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                for _, _, blocks in self.dataset:

                    # Update discriminator for n_critic times
                    for _ in range(self.n_critic):
                        self.update_D(blocks)

                    # Update generator for one time
                    self.update_G(blocks)

                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        

        
        self.dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=4, device=self.device)
    
    def prepare(self, weight_dir):
        '''Prepare stage for pretrained weights'''
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')

        # Load the pre-trained weights for Encoder and Decoder
        model_dict = self.G.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.G.load_state_dict(model_dict)
        




        






        



        