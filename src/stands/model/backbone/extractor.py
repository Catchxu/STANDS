import torch.nn as nn
from .gene import GATEncoder, MLPEncoder, MLPDecoder
from .image import ResNetEncoder, ResNetDecoder
from .layer import TFBlock, CrossTFBlock


class Extractor(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        z_dim = config['out_dim'][-1]
        self.GeneEncoder = GATEncoder(config['gene_dim'], config['out_dim'], **config['GATEncoder'])
        self.GeneDecoder = MLPDecoder(config['gene_dim'], config['out_dim'])
        self.ImageEncoder = ResNetEncoder(config['patch_size'], z_dim=z_dim, **config['ImageEncoder'])
        self.ImageDecoder = ResNetDecoder(config['patch_size'], z_dim=z_dim, **config['ImageDecoder'])

        if config['cross_attn']:
            self.fusion = CrossTFBlock(z_dim, z_dim, **config['TFBlock'])
        else:
            self.fusion = TFBlock(z_dim, z_dim, **config['TFBlock'])
    
    def encode(self, g_block, feat_g, feat_p):
        z_g = self.GeneEncoder(g_block, feat_g)
        z_p = self.ImageEncoder(g_block[1], feat_p)
        return z_g, z_p

    def decode(self, z_g, z_p):
        feat_g = self.GeneDecoder(z_g)
        feat_p = self.ImageDecoder(z_p)
        return feat_g, feat_p

    def pretrain(self, g_block, feat_g, feat_p):
        z_g, z_p = self.encode(g_block, feat_g, feat_p)
        z_g, z_p = self.fusion(z_g, z_p)
        feat_g, feat_p = self.decode(z_g, z_p)
        return feat_g, feat_p




class ExtractorOnlyST(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        z_dim = config['out_dim'][-1]
        self.GeneEncoder = GATEncoder(config['gene_dim'], config['out_dim'], **config['GATEncoder'])
        self.GeneDecoder = MLPDecoder(config['gene_dim'], config['out_dim'])
    
    def encode(self, g_block, feat_g):
        z_g = self.GeneEncoder(g_block, feat_g)
        return z_g

    def decode(self, z_g):
        feat_g = self.GeneDecoder(z_g)
        return feat_g
    
    def pretrain(self, g_block, feat_g):
        return self.decode(self.encode(g_block, feat_g))




class ExtractorOnlySC(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.GeneEncoder = MLPEncoder(config['gene_dim'], config['out_dim'])
        self.GeneDecoder = MLPDecoder(config['gene_dim'], config['out_dim'])
    
    def encode(self, feat_g):
        z_g = self.GeneEncoder(feat_g)
        return z_g

    def decode(self, z_g):
        feat_g = self.GeneDecoder(z_g)
        return feat_g
    
    def pretrain(self, feat_g):
        return self.decode(self.encode(feat_g))