import torch
import torch.nn as nn

from .layer import GAT


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




class ResNetEncoder(nn.Module):
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
            nn.BatchNorm2d(8),
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




class ResNetDecoder(nn.Module):
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