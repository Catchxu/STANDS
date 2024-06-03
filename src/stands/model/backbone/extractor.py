import os
import torch
import torch.nn as nn
from typing import Optional

from .gene import GATEncoder, MLPEncoder, MLPDecoder
from .image import ResNetEncoder, ResNetDecoder
from .layer import TFBlock, CrossTFBlock


class Extractor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        z_dim = configs.out_dim[-1]
        self.z_dim = z_dim*2
        self.g_dim = z_dim
        self.p_dim = z_dim
        self.GeneEncoder = GATEncoder(configs.gene_dim, configs.out_dim, **configs.GATEncoder)
        self.GeneDecoder = MLPDecoder(configs.gene_dim, configs.out_dim)
        self.ImageEncoder = ResNetEncoder(configs.patch_size, z_dim=z_dim, **configs.ImageEncoder)
        self.ImageDecoder = ResNetDecoder(configs.patch_size, z_dim=z_dim, **configs.ImageDecoder)

        if configs.cross_attn:
            self.fusion = CrossTFBlock(z_dim, z_dim, **configs.TFBlock)
        else:
            self.fusion = TFBlock(z_dim, z_dim, **configs.TFBlock)

        # Version ID
        self.only_SC = False
        self.only_ST = False

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
    
    def load_weight(self, weight_dir: Optional[str]):
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')
        
        # load the pre-trained weights for extractor
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)




class ExtractorOnlyST(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.z_dim = configs.out_dim[-1]
        self.GeneEncoder = GATEncoder(configs.gene_dim, configs.out_dim, **configs.GATEncoder)
        self.GeneDecoder = MLPDecoder(configs.gene_dim, configs.out_dim)

        # Version ID
        self.only_SC = False
        self.only_ST = True
    
    def encode(self, g_block, feat_g):
        z_g = self.GeneEncoder(g_block, feat_g)
        return z_g

    def decode(self, z_g):
        feat_g = self.GeneDecoder(z_g)
        return feat_g
    
    def pretrain(self, g_block, feat_g):
        return self.decode(self.encode(g_block, feat_g))

    def load_weight(self, weight_dir: Optional[str]):
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')
        
        # Load the pre-trained weights for extractor
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)




class ExtractorOnlySC(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.z_dim = configs.out_dim[-1]
        self.GeneEncoder = MLPEncoder(configs.gene_dim, configs.out_dim)
        self.GeneDecoder = MLPDecoder(configs.gene_dim, configs.out_dim)

        # Version ID
        self.only_SC = True
        self.only_ST = False
    
    def encode(self, feat_g):
        z_g = self.GeneEncoder(feat_g)
        return z_g

    def decode(self, z_g):
        feat_g = self.GeneDecoder(z_g)
        return feat_g
    
    def pretrain(self, feat_g):
        return self.decode(self.encode(feat_g))

    def load_weight(self, weight_dir: Optional[str]):
        if weight_dir:
            pre_weights = torch.load(weight_dir)
        else:
            pre_weights = torch.load(os.path.dirname(__file__) + '/model.pth')
        
        # Load the pre-trained weights for extractor
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pre_weights.items()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)




class ExtractorDis(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.z_dim = z_dim = configs.out_dim[-1]
        self.GeneEncoder = MLPEncoder(configs.gene_dim, configs.out_dim)
        self.ImageEncoder = ResNetEncoder(configs.patch_size, z_dim=z_dim, **configs.ImageEncoder)

    def encode(self, feat_g, feat_p):
        z_g = self.GeneEncoder(feat_g)
        z_p = self.ImageEncoder.woGAT_forward(feat_p)
        return z_g, z_p




class ExtractorDisOnlySC(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.z_dim = configs.out_dim[-1]
        self.GeneEncoder = MLPEncoder(configs.gene_dim, configs.out_dim)
    
    def encode(self, feat_g):
        z_g = self.GeneEncoder(feat_g)
        return z_g