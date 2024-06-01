import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from tqdm import tqdm
from sklearn.cluster import KMeans

from .backbone.layer import TFBlock, CrossTFBlock
from .generator import GeneratorAD
from ..configs import ClusterConfigs


class Cluster(nn.Module):
    def __init__(self, generator: GeneratorAD, n_subtypes):
        super().__init__()

        self.G = generator
        self.z_dim = self.G.z_dim
        self.subtypes = n_subtypes

        configs = ClusterConfigs(self.z_dim)
        self.alpha = configs.alpha
        self.KMeans_n_init = configs.KMeans_n_init
        self.learning_rate = configs.learning_rate
        self.n_epochs = configs.n_epochs
        self.update_interval = configs.update_interval
        self.weight_decay = configs.weight_decay

        if configs.cross_attn:
            self.fusion = CrossTFBlock(**configs.TFBlock)
        else:
            self.fusion = TFBlock(**configs.TFBlock)

        self.mu = Parameter(torch.Tensor(self.subtypes, self.z_dim))

        # classifer for supervised pre-training
        self.classifer = nn.Linear(self.z_dim, n_subtypes)

    def fullforward(self, g_block, res_g, res_p):
        res_g, res_p = self.G.extract.encode(g_block, res_g, res_p)
        res_z = torch.cat([res_g, res_p], dim=1)
        return res_z

    def STforward(self, g_block, res_g):
        res_z = self.G.extract.encode(g_block, res_g)
        return res_z

    def SCforward(self, res_g):
        res_z = self.G.extract.encode(res_g)
        return res_z

    def forward(self, z, res_z):
        x = self.fusion(z, res_z)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        self.mu_update(x, q)
        return x, q
    
    def pretrain(self, z, res_z):
        x = self.fusion(z, res_z)
        return self.classifer(x)

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def mu_init(self, feat):
        kmeans = KMeans(self.subtypes, n_init=self.KMeans_n_init)
        y_pred = kmeans.fit_predict(feat)
        feat = pd.DataFrame(feat, index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def mu_update(self, feat, q):
        y_pred = torch.argmax(q, axis=1).cpu().numpy()
        feat = pd.DataFrame(feat.cpu().detach().numpy(), index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name='Group')
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby('Group').mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def fit(self, z, res_z):
        # z and res_z are obtained by fullforward, SCforward, or STforward
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.weight_decay)

        self.mu_init(self.fusion(z, res_z).cpu().detach().numpy())

        self.train()
        with tqdm(total=self.n_epochs) as t:
            for epoch in range(self.n_epochs):
                t.set_description(f'Train Epochs')

                if epoch % self.update_interval == 0:
                    _, q = self.forward(z, res_z)
                    p = self.target_distribution(q).data

                optimizer.zero_grad()
                _, q = self.forward(z, res_z)
                loss = self.loss_function(p, q)
                loss.backward()
                optimizer.step()
            
                t.set_postfix(Loss = loss.item())
                t.update(1)

        with torch.no_grad():
            self.eval()
            new_z, q = self.forward(new_z)
            return q