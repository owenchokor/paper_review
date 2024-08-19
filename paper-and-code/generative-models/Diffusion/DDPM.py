import torch
import torch.nn as nn

class Upsample(nn.Module):
    def init(
            self, dim, dim_out, factor = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1) #1d conv
        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared i, h, w)
        nn.init.kaiming_uniform_(conv_weight)