import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .backbone import GATEncoder, MLPEncoder, MLPDecoder
from .backbone import ResNetEncoder, ResNetDecoder
from ._block import MemoryBlock, TFBlock, StyleBlock




class STNet(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim=[512, 256], 
                 z_dim=256, use_image=True):
        super().__init__()
        self.GeneEncoder = GeneEncoder(in_dim, out_dim, nheads=[4, 1])
        self.GeneDecoder = GeneDecoder(in_dim, out_dim)

        self.ImageEncoder = ImageEncoder(patch_size, z_dim=z_dim)
        self.ImageDecoder = ImageDecoder(patch_size, z_dim=z_dim)

        self.Fusion = TFBlock(d_model=z_dim)

        self.use_image = use_image

    def encode(self, g_block, feat_g, feat_p=None):
        z_g = self.GeneEncoder(g_block, feat_g)
        if self.use_image:
            z_p = self.ImageEncoder(g_block[1], feat_p)
            return z_g, z_p

        else:
            return z_g, None

    def decode(self, z_g, z_p=None):
        feat_g = self.GeneDecoder(z_g)
        if self.use_image:
            feat_p = self.ImageDecoder(z_p)
            return feat_g, feat_p
        else:
            return feat_g, None

    def pretrain(self, g_block, feat_g, feat_p):
        z_g, z_p = self.encode(g_block, feat_g, feat_p)
        z_g, z_p = self.Fusion(z_g, z_p)
        feat_g, feat_p = self.decode(z_g, z_p)
        return feat_g, feat_p


class GeneratorAD(STNet):
    def __init__(self, patch_size, in_dim, out_dim=[512, 256], z_dim=256, 
                 use_image=True, mem_dim=1024, thres=0.01, tem=1, **kwargs):
        super().__init__(patch_size, in_dim, out_dim, z_dim, use_image, **kwargs)
        if use_image:
            self.Memory = MemoryBlock(mem_dim, 2*z_dim, thres, tem)
        else:
            self.Memory = MemoryBlock(mem_dim, z_dim, thres, tem)

    def forward(self, g_block, feat_g, feat_p=None):
        z_g, z_p = self.encode(g_block, feat_g, feat_p)

        if self.use_image:
            z_g, z_p = self.Fusion(z_g, z_p)
            z = torch.concat([z_g, z_p], dim=-1)
            mem_z = self.Memory(z)
            z_g, z_p = torch.chunk(mem_z, 2, dim = -1)
        else:
            z = z_g
            z_g = self.Memory(z_g)

        feat_g, feat_p = self.decode(z_g, z_p)
        return z, feat_g, feat_p


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
