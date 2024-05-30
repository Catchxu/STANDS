import torch.nn as nn
import torch.nn.functional as F

from .layer import GAT, LinearBlock


class GATEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256], nheads=[4, 1]):
        super().__init__()
        self.GAT1 = GAT(in_dim, out_dim[0], nheads[0])
        self.GAT2 = GAT(out_dim[0], out_dim[1], nheads[1],
                        False, False, False)

    def forward(self, g_block, feat):
        feat = self.GAT1(g_block[0], feat)
        z = self.GAT2(g_block[1], feat)
        return z




class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256]):
        super().__init__()
        self.fc = nn.Sequential(
            LinearBlock(in_dim, out_dim[0]),
            LinearBlock(out_dim[0], out_dim[1]),
        )
        self.act = nn.ReLU()

    def forward(self, feat):
        z = self.fc(feat)
        return self.act(z)




class MLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256]):
        super().__init__()
        self.fc = nn.Sequential(
            LinearBlock(out_dim[1], out_dim[0]),
            LinearBlock(out_dim[0], in_dim, False, False, False),
        )
        self.act = nn.ReLU()
    
    def forward(self, z):
        feat = self.fc(z)
        return self.act(feat)