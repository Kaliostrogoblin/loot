import torch
from torch import nn

from .conv_coord import AddCoords


class Loot(nn.Module):
    def __init__(self, n_stations, use_radius=True):
        super(Loot, self).__init__()
        self.coord_conv = AddCoords(radius_channel=use_radius)
        self.n_stations = n_stations
        self.in_channels = n_stations + 2
        if use_radius:
          self.in_channels += 1
          
        # layers
        self.cnn1 = self.simple_conv_block(self.in_channels, 32, 9)
        self.cnn2 = self.simple_conv_block(32, 32, 9)
        self.cnn3 = self.simple_conv_block(32, 64, 9)
        self.cnn4 = self.simple_conv_block(64, 64, 9)
        self.out_probs = nn.Conv2d(64, 1, 1)
        self.out_shifts = nn.Conv2d(64, (self.n_stations-1)*2, 1)
        
        
    def forward(self, inputs):
        x = self.coord_conv(inputs)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        # residual connection
        logits = self.out_probs(x)+inputs[:, :1, :, :]
        shifts = self.out_shifts(x)
        outputs = torch.cat([logits, shifts], 1)
        return outputs
        
        
    def simple_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels))