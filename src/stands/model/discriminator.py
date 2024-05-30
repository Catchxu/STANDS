import torch
import torch.nn as nn

from .backbone import CriticNet, ExtractorDis, ExtractorDisOnlySC
from ..configs import DisSCConfigs, DisFullConfigs, DisConfigs


class Discriminator(nn.Module):
    def __init__(self, gene_dim, patch_size, only_ST=False, only_SC=False):
        super().__init__()
        assert only_ST and only_SC == False

        if only_ST or only_SC:
            configs = DisSCConfigs(gene_dim)
            self.extract = ExtractorDisOnlySC(configs)
        
        else:
            configs = DisFullConfigs(gene_dim, patch_size)
            self.extract = ExtractorDis(configs)

        self.discriminator = CriticNet(DisConfigs(configs.z_dim, only_ST, only_SC))
    
    def fullforward(self, feat_g, feat_p):
        z_g, z_p = self.extract.encode(feat_g, feat_p)
        z = torch.cat([z_g, z_p], dim=1)
        return self.discriminator(z)

    def SCforward(self, feat_g):
        z_g = self.extract.encode(feat_g)
        return self.discriminator(z_g)

