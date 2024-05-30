import torch
import torch.nn as nn

from .backbone import CriticNet
from ..configs import DisConfigs


class Discriminator(nn.Module):
    def __init__(self, g_dim, p_dim=None, only_ST=False, only_SC=False):
        super().__init__()
        assert only_ST and only_SC == False

        self.discriminator = CriticNet(DisConfigs(g_dim, p_dim, only_ST, only_SC))
    
    def fullforward(self, z_g, z_p):
        z = torch.cat([z_g, z_p], dim=1)
        return self.discriminator(z)

    def geneforward(self, z_g):
        return self.discriminator(z_g)

