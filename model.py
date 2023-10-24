import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channel_img, feature_d):
        super(Discriminator, self).__init__()
        self.critic = nn.Sequential(
            self._block(channel_img, feature_d, 5, 2, 1),
            self._block(feature_d, feature_d * 2, 4, 2, 1),
            self._block(feature_d * 2, feature_d * 4, 4, 2, 1),
            self._block(feature_d * 4, feature_d * 8, 4, 2, 1),
            nn.Conv2d(feature_d * 8, 1, 4, 2, 1)
        )

    def forward(self, x):
        return self.critic(x)

    @staticmethod
    def _block(in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )


class Generator(nn.Module):
    def __init__(self, z_dim, channel_img, feature_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, feature_g * 16, 6, 2, 1),
            self._block(feature_g * 16, feature_g * 8, 4, 2, 1),
            self._block(feature_g * 8, feature_g * 4, 4, 2, 1),
            self._block(feature_g * 4, feature_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_g * 2, channel_img, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

    def _block(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.2)
    return None
