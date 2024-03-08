import os
import dgl
import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder

from .model import GeneratorAD, GeneratorPair, GeneratorBC
from .model import Discriminator, GMMWithPrior, Cluster
from ._utils import seed_everything, calculate_gradient_penalty


class AnomalyDetection:
    def __init__(self, n_epochs: int = 10, batch_size: int = 128,
                 learning_rate: float = 2e-5, mem_dim: int = 1024,
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
        self.n_critic = n_critic

        if random_state is not None:
            seed_everything(random_state)

        if weight is None:
            self.weight = {'w_rec': 30, 'w_adv': 1, 'w_gp': 10}
        else:
            self.weight = weight

    def fit(self, ref: Dict[str, Any], weight_dir: Optional[str] = None,
            save: bool = False, **kwargs):
        '''Fine-tune STand on reference graph'''
        tqdm.write('Begin to fine-tune the model on reference datasets...')

        # dataset provides subgraph for training
        ref_g = ref['graph']
        self.use_img = ref['use_image']
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            ref_g, ref_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=0, device=self.device)

        self.D = Discriminator(ref['patch_size'], ref['gene_dim']).to(self.device)
        self.G = GeneratorAD(ref['patch_size'], ref['gene_dim'], mem_dim=self.mem_dim, 
                             use_image=self.use_img, **kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr*self.n_critic, betas=(0.5, 0.999))
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
                        self.UpdateD(blocks)

                    # Update generator for one time
                    self.UpdateG(blocks)

                # Update learning rate for G and D
                self.D_sch.step()
                self.G_sch.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        tqdm.write('Fine-tuning has been finished.')

        # if save:
        #     if 'patch' in ref_g.ndata.keys():
        #         save_module = ['GeneEncoder', 'GeneDecoder',
        #                        'ImageEncoder', 'ImageDecoder',
        #                        'Fusion']
        #     else:
        #         save_module = ['GeneEncoder', 'GeneDecoder']

        #     weight_dir = os.path.dirname(__file__) + '/temp.pth'
        #     self.G.save_weights(weight_dir, save_module)  # save the trained STNet weights
    
    @torch.no_grad()
    def predict(self, tgt: Dict[str, Any]):
        '''Detect anomalous spots on target graph'''
        if (self.G is None or self.D is None):
            raise RuntimeError('Please fine-tune the model first.')
        
        tgt_g = tgt['graph']

        dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0, device=self.device)

        self.G.eval()
        self.D.eval()
        tqdm.write('Detect anomalous spots on target dataset...')
    
        ref_score = self.score(self.dataset)
        tgt_score = self.score(dataset)
        gmm = GMMWithPrior(ref_score, tol=0.00001)
        threshold = gmm.fit(tgt_score=tgt_score)
        tgt_label = [1 if s >= threshold else 0 for s in tgt_score]

        tqdm.write('Anomalous spots have been detected.\n')
        return tgt_score, tgt_label

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
                real_p = blocks[1].srcdata['patch'] if self.use_img else None
                z, _, _ = self.G(blocks, real_g, real_p)
                self.G.Memory.update_mem(z)
                t += 1

    def UpdateD(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        # generate fake data
        _, fake_g, fake_p = self.G(blocks, blocks[0].srcdata['gene'],
                                   blocks[1].srcdata['patch'] if self.use_img else None)

        # get real data from blocks
        real_g = blocks[1].dstdata['gene']
        real_p = blocks[1].dstdata['patch'] if self.use_img else None

        if self.use_img:
            d1 = torch.mean(self.D(real_g, real_p))
            d2 = torch.mean(self.D(fake_g.detach(), fake_p.detach()))
            gp = calculate_gradient_penalty(self.D, real_g, fake_g.detach(),
                                            real_p, fake_p.detach())
        else:
            d1 = torch.mean(self.D(real_g))
            d2 = torch.mean(self.D(fake_g.detach()))
            gp = calculate_gradient_penalty(self.D, real_g, fake_g.detach())

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()

    def UpdateG(self, blocks):
        '''Updating generator'''
        self.opt_G.zero_grad()

        # get real data from blocks
        real_g = blocks[0].srcdata['gene']
        real_p = blocks[1].srcdata['patch'] if self.use_img else None
        real_z, fake_g, fake_p = self.G(blocks, real_g, real_p)

        # discriminator provides feedback
        d = self.D(fake_g, fake_p)

        if self.use_img:
            Loss_rec = (self.L1(blocks[-1].dstdata['gene'], fake_g) + 
                        self.L1(blocks[-1].dstdata['patch'], fake_p))/2
        else:
            Loss_rec = self.L1(blocks[-1].dstdata['gene'], fake_g)
        Loss_adv = -torch.mean(d)

        # store generator loss for printing training information and backward
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)
        self.G_loss.backward()
        self.opt_G.step()

        # updating memory block with generated embeddings, fake_z
        self.G.Memory.update_mem(real_z)
    
    def score(self, dataset):
        # calucate anomaly score
        dis = []
        for _, _, blocks in dataset:
            # get real data from blocks
            real_g = blocks[0].srcdata['gene']
            real_p = blocks[1].srcdata['patch'] if self.use_img else None
            _, fake_g, fake_p = self.G(blocks, real_g, real_p)

            d = self.D(fake_g, fake_p)
            dis.append(d.cpu().detach())

        # Normalize anomaly scores
        dis = torch.mean(torch.cat(dis, dim=0), dim=1).numpy()
        # dis = torch.max(torch.cat(dis, dim=0), dim=1).values.numpy()
        score = (dis.max() - dis)/(dis.max() - dis.min())

        score = list(score.reshape(-1))
        return score




class KinPair:
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
    
    def fit(self, raw: Dict[str, Any], weight_dir: Optional[str] = None, **kwargs):
        '''Find spot pairs '''
        tqdm.write('Begin to find spot pairs between datasets...')

        raw_g = raw['graph']
        ref_g, tgt_g = self.split(raw_g)

        self.D = Discriminator(raw['patch_size'], 256).to(self.device)
        self.G = GeneratorPair(raw['patch_size'], raw['gene_dim'],
                               ref_g.num_nodes(), tgt_g.num_nodes(), **kwargs).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
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

        tqdm.write('Spot pairs have been found.\n')
        return ref_g, tgt_g

    def split(self, graph: dgl.DGLGraph):
        '''Split the integrated graph to reference and target graph'''
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
        self.D_loss.backward()
        self.opt_D.step()
    
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
        self.G_loss.backward()
        self.opt_G.step()




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
    
    def fit(self, raw: Dict[str, Any], weight_dir: Optional[str] = None, **kwargs):
        '''Remove batch effects'''
        adatas = raw['adata']
        adata_ref = adatas[0]
        adata_tgt = ad.concat(adatas[1:])
        
        Aligner = KinPair(random_state=self.seed, **kwargs)
        _, tgt_g = Aligner.fit(raw)

        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=True,
            drop_last=False, num_workers=0, device=self.device)

        self.D = Discriminator(raw['patch_size'], raw['gene_dim']).to(self.device)
        self.G = GeneratorBC(raw['data_n'], raw['patch_size'],
                             raw['gene_dim']).to(self.device)

        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.D_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_D,
                                                          T_max = self.n_epochs)
        self.G_sch = optim.lr_scheduler.CosineAnnealingLR(optimizer = self.opt_G,
                                                          T_max = self.n_epochs)
        
        self.L1 = nn.L1Loss().to(self.device)

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

                # Update learning rate for G and D
                self.D_sch.step()
                self.G_sch.step()
                t.set_postfix(G_Loss = self.G_loss.item(),
                              D_Loss = self.D_loss.item())
                t.update(1)

        self.dataset = dgl.dataloading.DataLoader(
            tgt_g, tgt_g.nodes(), self.sampler,
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0, device=self.device)

        self.G.eval()
        corrected = []
        with torch.no_grad():
            for _, _, blocks in self.dataset:
                fake_g = self.G(blocks, blocks[0].srcdata['gene'],
                                blocks[-1].dstdata['batch'])
                corrected.append(fake_g.cpu().detach())

        corrected = torch.cat(corrected, dim=0).numpy()
        adata_tgt.X = corrected
        adata = ad.concat([adata_ref, adata_tgt])
        tqdm.write('Datasets have been corrected.\n')
        return adata

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

    def update_D(self, blocks):
        '''Updating discriminator'''
        self.opt_D.zero_grad()

        # generate fake data
        fake_g = self.G(blocks, blocks[0].srcdata['gene'], blocks[-1].dstdata['batch'])

        # get real data from blocks
        real_g = blocks[-1].dstdata['ref_gene']

        d1 = torch.mean(self.D(real_g))
        d2 = torch.mean(self.D(fake_g.detach()))
        gp = calculate_gradient_penalty(self.D, real_g, fake_g.detach())

        # store discriminator loss for printing training information
        self.D_loss = - d1 + d2 + gp * self.weight['w_gp']
        self.D_loss.backward()
        self.opt_D.step()

    def update_G(self, blocks):
        '''Updating generator'''
        self.opt_G.zero_grad()

        # generate fake data
        fake_g = self.G(blocks, blocks[0].srcdata['gene'], blocks[-1].dstdata['batch'])

        # discriminator provides feedback
        d = self.D(fake_g)

        Loss_rec = self.L1(blocks[-1].dstdata['ref_gene'], fake_g)
        Loss_adv = -torch.mean(d)

        # store generator loss for printing training information and backward
        self.G_loss = (self.weight['w_rec']*Loss_rec +
                       self.weight['w_adv']*Loss_adv)
        self.G_loss.backward()
        self.opt_G.step()




class SubNet:
    def __init__(self, generator: nn.Module, n_subtypes: int = 2,
                 n_epochs: int = 10, update_interval=3,
                 learning_rate: float = 1e-4, GPU: bool = True, 
                 random_state: Optional[int] = None
            ):
        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train STAND.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.n_epochs = n_epochs
        self.interval = update_interval
        self.lr = learning_rate
        self.n_subtypes = n_subtypes

        if random_state is not None:
            self.seed = random_state
            seed_everything(random_state)

        self.G = generator.to(self.device)

    def pretrain(self, tgt: Dict[str, Any], type_key: str,
                 pretrain_epochs: int = 100):
        graph = tgt['graph'].to(self.device)
        label_encoder = LabelEncoder()
        df = tgt['adata'].obs
        df['type_encoded'] = label_encoder.fit_transform(df[type_key])
        node_type = pd.get_dummies(df['type_encoded'], prefix='category')
        node_type = torch.FloatTensor(node_type.values).to(self.device)
        graph.ndata['type'] = node_type

        res_g, res_p = self.gen_res(self, graph)
        graph.ndata['res_gene'] = res_g
        graph.ndata['res_patch'] = res_p        
        self.C = Cluster(self.G, tgt['use_image'])

        opt = optim.Adam(self.C.parameters(), lr=self.lr, betas=(0.5, 0.999))
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer = self.opt, T_max = pretrain_epochs)
        Loss = nn.CrossEntropyLoss().to(self.device)

        self.C.train()
        with tqdm(total=pretrain_epochs) as t:
            for _ in range(self.n_epochs):
                t.set_description(f'Pretrain Epochs')
                
                opt.zero_grad()
                pred = self.C.pretrain(
                    [graph, graph], graph.ndata['gene'], graph.ndata['res_gene'],
                    graph.ndata['patch'], graph.ndata['res_patch']
                )
                loss = Loss(pred, graph.ndata['type'])
                loss.backward()
                opt.step()
                sch.step()

                t.set_postfix(Loss =  loss.item())
                t.update(1)
        
        return self.C

    @torch.no_grad()
    def gen_res(self, graph: dgl.DGLGraph):
        '''Generate reconstructed data'''
        self.G.eval()
        _, fake_g, fake_p = self.G(
            [graph, graph], graph.ndata['gene'], graph.ndata['patch']
            )
        res_g = graph.ndata['gene'] - fake_g.detach()
        res_p = graph.ndata['patch'] - fake_p.detach()
        return res_g, res_p








