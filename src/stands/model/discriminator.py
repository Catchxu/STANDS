import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as SNorm


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act: bool = True):
        super().__init__()
        self.linear = nn.Sequential(
            SNorm(nn.Linear(in_dim, out_dim)),
            nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.linear(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.residual_block = nn.Sequential(
            SNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            SNorm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class ResNet(nn.Module):
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
            SNorm(nn.Conv2d(in_channels=input_channels, out_channels=8,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)),
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
                    SNorm(nn.Conv2d(n_filters_1, n_filters_2,
                              kernel_size=(2, 2), stride=(2, 2), padding=0)),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

            if MultiResSkips:
                self.multi_res_skip_list.append(
                    nn.Sequential(
                        SNorm(nn.Conv2d(in_channels=n_filters_1, out_channels=self.max_filters,
                                  kernel_size=(ks, ks), stride=(ks, ks), padding=0)),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )

        self.output_conv = nn.Conv2d(in_channels=self.max_filters, out_channels=z_dim,
                                     kernel_size=(3, 3), stride=(1, 1), padding=1)
        
        self.z_dim = z_dim
        self.img_latent_dim = patch_size // (2**n_levels)
        self.feat_dim = self.z_dim*self.img_latent_dim**2
        self.fc = nn.Linear(self.feat_dim, self.z_dim)

    def forward(self, feat):
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
        z = self.fc(feat.flatten(1))
        return z

    
class CriticNet(nn.Module):
    def __init__(self, in_dim=256, out_dim=64):
        super().__init__()
        
        self.initial1 = nn.Sequential(
            SNorm(nn.Linear(in_dim, in_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            SNorm(nn.Linear(in_dim, out_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.initial2 = nn.Sequential(
            SNorm(nn.Linear(in_dim*2, in_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            SNorm(nn.Linear(in_dim, out_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.FCBlock = nn.Sequential(
            SNorm(nn.Linear(out_dim, out_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            SNorm(nn.Linear(out_dim, out_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, dis_g, dis_p=None):
        if dis_p is None:
            z = self.initial1(dis_g)
        else:
            x = torch.cat([dis_g, dis_p], dim=1)
            z = self.initial2(x)
        return z + self.FCBlock(z)


class Discriminator(nn.Module):
    def __init__(self, patch_size, in_dim, out_dim=[512, 256], z_dim=256):
        super().__init__()
        self.gene_dis = nn.Sequential(
            LinearBlock(in_dim, out_dim[0]),
            LinearBlock(out_dim[0], out_dim[1])
        )
        self.image_dis = ResNet(patch_size, z_dim=z_dim)

        self.critic = CriticNet(in_dim=z_dim, out_dim=16)

    def forward(self, feat_g, feat_p=None):
        dis_g = self.gene_dis(feat_g)
        if feat_p is None:
            dis = self.critic(dis_g)
        else:
            dis_p = self.image_dis(feat_p)
            dis = self.critic(dis_g, dis_p)
        return dis

