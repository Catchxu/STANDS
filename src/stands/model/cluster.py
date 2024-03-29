import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans

from ._block import TFBlock


class Cluster(nn.Module):
    def __init__(self, generator, use_image: bool = True,
                 z_dim=256, n_subtypes=2, alpha=1, **kwargs):
        super().__init__()
        self.subtypes = n_subtypes
        self.alpha = alpha
        self.z_dim = z_dim * 2 if use_image else z_dim
        self.Fusion = TFBlock(self.z_dim, num_heads=2)
        self.G = generator

        self.mu = Parameter(torch.Tensor(self.subtypes, self.z_dim*2))

        # classifer for supervised pre-training
        self.classifer = nn.Linear(self.z_dim*2, n_subtypes)

    def forward(self, g_block, feat_g, res_g, feat_p=None, res_p=None):
        z, _, _ = self.G(g_block, feat_g, feat_p)
        res_z, _, _ = self.G(g_block, res_g, res_p)
        x = self.Fusion(z, res_z)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        self.mu_update(x, q)
        return x, q
    
    def pretrain(self, g_block, feat_g, res_g, feat_p=None, res_p=None):
        z, _, _ = self.G(g_block, feat_g, feat_p)
        res_z, _, _ = self.G(g_block, res_g, res_p)
        x = self.Fusion(z, res_z)
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
        kmeans = KMeans(self.subtypes, n_init=20)
        y_pred = kmeans.fit_predict(feat)
        feat = pd.DataFrame(feat, index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def mu_update(self, feat, q):
        y_pred = torch.argmax(q, axis=1).cpu().numpy()
        feat = pd.DataFrame(feat.cpu().detach().numpy(), index=np.arange(0, feat.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, feat.shape[0]), name="Group")
        Mergefeat = pd.concat([feat, Group], axis=1)
        centroid = np.asarray(Mergefeat.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(centroid))

    def fit(self, g, z_x, res_x, z_img, res_img, learning_rate=1e-4, n_epochs=10,
            update_interval=3, weight_decay=1e-4, verbose=True, log_interval=1):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        res_x = self.Gene_net.encode(g, feat=res_x)
        res_img = self.Image_net.encode(g, feat=res_img)
        new_z = torch.cat([z_x, z_img, res_x, res_img], dim=1)
        self.mu_init(self.Trans(new_z).cpu().detach().numpy())

        self.train()
        for epoch in range(n_epochs):
            if epoch % update_interval == 0:
                _, q = self.forward(new_z)
                p = self.target_distribution(q).data

            optimizer.zero_grad()
            _, q = self.forward(new_z)
            loss = self.loss_function(p, q)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ((epoch+1) % log_interval == 0) and (verbose):
                txt = 'Train Epoch: [{:^4}/{:^4}({:^3.0f}%)]    Loss: {:.6f}'
                txt = txt.format(epoch+1, n_epochs, 100.*(epoch+1)/n_epochs, loss)
                print(txt)

        with torch.no_grad():
            self.eval()
            new_z, q = self.forward(new_z)
            return new_z, q