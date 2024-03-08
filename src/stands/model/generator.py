import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dgl.nn import GATv2Conv

from ._block import MemoryBlock, TFBlock, StyleBlock


class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, nhead, norm: bool = True,
                 act: bool = True, dropout: bool = True):
        super().__init__()
        self.layer = GATv2Conv(in_dim, out_dim, num_heads=nhead)
        self.fc = nn.Sequential(
            nn.Linear(nhead*out_dim, out_dim),
            nn.InstanceNorm1d(out_dim) if norm else nn.Identity(),
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
            nn.InstanceNorm1d(out_dim) if norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity(),
            nn.Dropout(0.3) if dropout else nn.Identity(),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class GeneEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256], nheads=[4, 1]):
        super().__init__()
        self.GAT1 = GAT(in_dim, out_dim[0], nheads[0])
        self.GAT2 = GAT(out_dim[0], out_dim[1], nheads[1],
                        False, False, False)

    def forward(self, g_block, feat):
        feat = self.GAT1(g_block[0], feat)
        z = self.GAT2(g_block[1], feat)
        return z


class GeneDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=[512, 256]):
        super().__init__()
        self.fc = nn.Sequential(
            LinearBlock(out_dim[1], out_dim[0]),
            LinearBlock(out_dim[0], in_dim, False, False, False),
        )
    
    def forward(self, z):
        feat = self.fc(z)
        return F.relu(feat)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ImageEncoder(nn.Module):
    def __init__(self, patch_size, n_ResidualBlock=8, n_levels=2,
                 input_channels=3, z_dim=256, MultiResSkips=True):
        super().__init__()

        self.max_filters = 2**(n_levels+3)
        self.n_levels = n_levels
        self.MultiResSkips = MultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_1 = 2**(i + 3)
            n_filters_2 = 2**(i + 4)
            ks = 2**(n_levels - i)

            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                              for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                nn.Sequential(
                    nn.Conv2d(n_filters_1, n_filters_2,
                              kernel_size=(2, 2), stride=(2, 2), padding=0),
                    nn.BatchNorm2d(n_filters_2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if MultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                  kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        nn.BatchNorm2d(self.max_filters),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.z_dim = z_dim
        self.img_latent_dim = patch_size // (2**n_levels)
        self.feat_dim = self.z_dim*self.img_latent_dim**2
        self.GAT = GAT(self.feat_dim, self.z_dim, 4)

    def forward(self, g, feat):
        feat = self.input_conv(feat)
        skips = []

        for i in range(self.n_levels):
            feat = self.res_blk_list[i](feat)
            if self.MultiResSkips:
                skips.append(self.multi_res_skip_list[i](feat))
            feat = self.conv_list[i](feat)

        if self.MultiResSkips:
            feat = sum([feat] + skips)

        feat = self.output_conv(feat)
        z = self.GAT(g, feat.flatten(1))
        return z


class ImageDecoder(nn.Module):
    def __init__(self, patch_size, n_ResidualBlock=8, n_levels=2,
                 z_dim=256, output_channels=3, MultiResSkips=True):
        super().__init__()

        self.max_filters = 2**(n_levels+3)
        self.n_levels = n_levels
        self.MultiResSkips = MultiResSkips

        self.conv_list = nn.ModuleList()
        self.res_blk_list = nn.ModuleList()
        self.multi_res_skip_list = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=z_dim, out_channels=self.max_filters,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(self.max_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for i in range(n_levels):
            n_filters_0 = 2**(self.n_levels - i + 3)
            n_filters_1 = 2**(self.n_levels - i + 2)
            ks = 2 ** (i + 1)

            self.res_blk_list.append(
                nn.Sequential(*[ResidualBlock(n_filters_1, n_filters_1)
                              for _ in range(n_ResidualBlock)])
            )

            self.conv_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(n_filters_0, n_filters_1,
                                       kernel_size=(2, 2), stride=(2, 2), padding=0),
                    nn.BatchNorm2d(n_filters_1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if MultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(in_channels=self.max_filters, out_channels=n_filters_1,
                                           kernel_size=(ks, ks), stride=(ks, ks), padding=0),
                        nn.BatchNorm2d(n_filters_1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = nn.Conv2d(in_channels=n_filters_1, out_channels=output_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.z_dim = z_dim
        self.img_latent_dim = patch_size // (2**n_levels)
        self.feat_dim = self.z_dim*self.img_latent_dim**2
        self.fc = nn.Linear(z_dim, self.feat_dim)

    def forward(self, z):
        z = self.fc(z).view(-1, self.z_dim, self.img_latent_dim, self.img_latent_dim)
        z = z_top = self.input_conv(z)
        for i in range(self.n_levels):
            z = self.conv_list[i](z)
            z = self.res_blk_list[i](z)
            if self.MultiResSkips:
                z += self.multi_res_skip_list[i](z_top)

        x = self.output_conv(z)
        return torch.tanh(x)




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

    def save_weights(self, weight_dir: str, save_module: List[str]):
        state_dict = self.state_dict()
        save_state = {}

        for param in state_dict:
            for module in save_module:
                if module in param:
                    save_state.update({param: state_dict[param]})

        torch.save(save_state, weight_dir)


class GeneratorAD(STNet):
    def __init__(self, patch_size, in_dim, z_dim=256, use_image=True,
                 mem_dim=1024, thres=0.01, tem=1, **kwargs):
        super().__init__(patch_size, in_dim, z_dim, use_image, **kwargs)
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
