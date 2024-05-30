import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Extractor, ExtractorOnlyST, ExtractorOnlySC
from .backbone import MemoryBlock, StyleBlock
from ..configs import SCConfigs, STConfigs, FullConfigs, MBConfigs


class GeneratorAD(nn.Module):
    def __init__(self, gene_dim, out_dim=[512, 256], patch_size=None,
                 cross_attn=True, only_ST=False, only_SC=False):
        super().__init__()
        assert only_ST and only_SC == False

        if only_ST:
            self.extract = ExtractorOnlyST(STConfigs(gene_dim, out_dim))
            self.Memory = MemoryBlock(out_dim[-1], **MBConfigs().MBBlock)
        
        elif only_SC:
            self.extract = ExtractorOnlySC(SCConfigs(gene_dim, out_dim))
            self.Memory = MemoryBlock(out_dim[-1], **MBConfigs().MBBlock)
        
        else:
            paras = {
                'gene_dim': gene_dim,
                'out_dim': out_dim,
                'patch_size': patch_size,
                'cross_attn': cross_attn
            }
            self.extract = Extractor(FullConfigs(**paras))
            self.Memory = MemoryBlock(out_dim[-1]*2, **MBConfigs().MBBlock)
    
    def fullforward(self, g_block, feat_g, feat_p):
        z_g, z_p = self.extract.encode(g_block, feat_g, feat_p)
        z_g, z_p = self.extract.fusion(z_g, z_p)
        z = torch.concat([z_g, z_p], dim=-1)
        mem_z = self.Memory(z)
        z_g, z_p = torch.chunk(mem_z, 2, dim = -1)
        feat_g, feat_p = self.extract.decode(z_g, z_p)
        return z, feat_g, feat_p
    
    def STforward(self, g_block, feat_g):
        z = self.extract.encode(g_block, feat_g)
        mem_z = self.Memory(z)
        feat_g = self.extract.decode(mem_z)
        return z, feat_g

    def SCforward(self, feat_g):
        z = self.extract.encode(feat_g)
        mem_z = self.Memory(z)
        feat_g = self.extract.decode(mem_z)
        return z, feat_g




class GeneratorPair(STNet):
    def __init__(self, patch_size, in_dim, n_ref, n_tgt, **kwargs):
        super().__init__(patch_size, in_dim, **kwargs)
        self.mapping = nn.Parameter(torch.Tensor(n_tgt, n_ref))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mapping.size(1))
        self.mapping.data.uniform_(-stdv, stdv)

    def forward(self, ref, tgt):
        z_ref, _ = self.encode([ref, ref], ref.ndata['gene'])
        z_tgt, _ = self.encode([tgt, tgt], tgt.ndata['gene'])
        z_ref = torch.mm(F.relu(self.mapping), z_ref)
        return z_ref, z_tgt, F.relu(self.mapping).detach().cpu().numpy()


class GeneratorBC(STNet):
    def __init__(self, data_n, patch_size, in_dim, out_dim=[512, 256], **kwargs):
        super().__init__(patch_size, in_dim, out_dim, **kwargs)
        self.Style = StyleBlock(data_n, out_dim[-1])

    def forward(self, g_block, feat_g, batchid):
        z = self.GeneEncoder(g_block, feat_g)
        z = self.Style(z, batchid)
        feat_g = self.GeneDecoder(z)
        return feat_g
