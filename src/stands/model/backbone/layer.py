import torch.nn as nn
from dgl.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, nhead, norm: bool = True,
                 act: bool = True, dropout: bool = True):
        super().__init__()
        self.layer = GATv2Conv(in_dim, out_dim, num_heads=nhead)
        self.fc = nn.Sequential(
            nn.Linear(nhead*out_dim, out_dim),
            nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

    def forward(self, g, feat):
        feat = self.layer(g, feat)
        z = self.fc(feat.flatten(1))
        return z


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm: bool = True,
                 act: bool = True, dropout: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x