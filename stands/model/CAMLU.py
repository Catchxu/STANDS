import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from typing import Optional
from .._utils import seed_everything


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear(x)
        return x
    

class AENet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.AE = nn.Sequential(
            LinearBlock(in_dim, 256),
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            LinearBlock(64, 128),
            LinearBlock(128, 256),
            nn.Linear(256, in_dim)
        )
    
    def forward(self, x):
        return self.AE(x)


class CAMLU:
    def __init__(self, in_dim=3000, epochs=100, batch_size=32, learning_rate=0.001,
                 GPU=True, random_state: Optional[int] = None,):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if GPU:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                print("GPU isn't available, and use CPU to train STAND.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if random_state is not None:
            seed_everything(random_state)
        
        self.net = AENet(in_dim).to(self.device)
    
    def detect(self, ref: ad.AnnData, tgt: ad.AnnData, return_score=True):
        self.Dataset = DataLoader(ref.X, self.batch_size, True, pin_memory=True)
        self.Loss = nn.MSELoss().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), self.learning_rate)

        self.net.train()
        with tqdm(total=self.epochs) as t:
            for _ in range(self.epochs):
                t.set_description(f'Train Epochs')

                for _, data in enumerate(self.Dataset):
                    data = data.to(self.device)
                    recon = self.net(data)
                    loss = self.Loss(recon, data)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                t.set_postfix(Loss = loss.item())
                t.update(1)
        
        self.Dataset = DataLoader(tgt.X, self.batch_size, False, pin_memory=True)
        self.net.eval()

        error = []
        for _, data in enumerate(self.Dataset):
            data = data.to(self.device)
            recon = self.net(data)
            e = torch.abs(recon - data).cpu().detach()
            error.append(e)
        
        error = torch.cat(error, dim=0).numpy()
        error = ad.AnnData(X=error)
        sc.pp.filter_genes(error, min_cells=10)
        sc.pp.highly_variable_genes(error, n_top_genes=500, subset=True)

        if return_score:
            score = error.X.mean(axis = 1)
            score = (score - score.min()) / (score.max() - score.min())
            return score
        else:
            return error