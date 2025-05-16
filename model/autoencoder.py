import torch
import torch.nn as nn

def group_norm(channels):
    if channels >= 32:
        return nn.GroupNorm(8, channels)
    elif channels >= 16:
        return nn.GroupNorm(4, channels)
    else:
        return nn.GroupNorm(2, channels)

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=True, activation=True, dropout=None):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if norm:
        layers.append(group_norm(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    if dropout is not None:
        layers.append(nn.Dropout3d(dropout))
    return nn.Sequential(*layers)

def deconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0, norm=True, activation=True):
    layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
    if norm:
        layers.append(group_norm(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class Conv3DAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 176, 208, 160), output_channels=1, bottleneck_dim=64):
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(1, 32, kernel_size=4, stride=2, padding=1, dropout=0.1),
            conv_block(32, 24, kernel_size=4, stride=2, padding=1, dropout=0.1),
            conv_block(24, 16, kernel_size=4, stride=2, padding=1),
            conv_block(16, 8, kernel_size=4, stride=2, padding=1),
        )

        # # Determine shape after encoder
        # with torch.no_grad():
        #     dummy = torch.zeros((1, *input_shape))
        #     conv_out = self.encoder_conv(dummy)
        #     self.conv_shape = conv_out.shape[1:]
        #     self.flat_dim = conv_out.numel() // conv_out.shape[0]

        # # Bottleneck
        # self.encoder_fc = nn.Linear(self.flat_dim, bottleneck_dim)
        # self.decoder_fc = nn.Linear(bottleneck_dim, self.flat_dim)

        # Decoder
        self.decoder_conv = nn.Sequential(
            deconv_block(8, 16, kernel_size=4, padding=1, output_padding=0),
            deconv_block(16, 24, kernel_size=4, padding=1, output_padding=0),
            deconv_block(24, 32, kernel_size=4, padding=1, output_padding=0),
            deconv_block(32, output_channels, kernel_size=4, padding=1, output_padding=0, norm=False, activation=False),
        )

    def forward(self, x):
        z = self.encoder_conv(x)
        # x = torch.flatten(x, start_dim=1)
        # z = self.encoder_fc(x)

        # x = self.decoder_fc(z)
        # x = x.view(-1, *self.conv_shape)
        x_out = self.decoder_conv(z)

        return z, x_out


class Conv3DAutoencoder2c(nn.Module):
    def __init__(self, input_shape=(1, 176, 208, 160), bottleneck_dim=64):
        super().__init__()
        self.input_shape = input_shape

        # Encoder: double convs per level, half filters
        self.encoder_conv = nn.Sequential(
            # Level 1
            # conv_block(1, 12, kernel_size=3, stride=1, padding=1, dropout=True),
            conv_block(1, 12, kernel_size=4, stride=2, padding=1, dropout=True),
            # Level 2
            # conv_block(12, 12, kernel_size=3, stride=1, padding=1, dropout=True),
            conv_block(12, 12, kernel_size=4, stride=2, padding=1, dropout=True),
            # Level 3
            # conv_block(12, 8, kernel_size=3, stride=1, padding=1),
            conv_block(8, 8, kernel_size=4, stride=2, padding=1),
            # Level 4
            # conv_block(8, 4, kernel_size=3, stride=1, padding=1),
            conv_block(4, 4, kernel_size=4, stride=2, padding=1),
        )

        # Determine shape after encoder
        with torch.no_grad():
            dummy = torch.zeros((1, *input_shape))
            conv_out = self.encoder_conv(dummy)
            self.conv_shape = conv_out.shape[1:]
            self.flat_dim = conv_out.numel() // conv_out.shape[0]

        # Bottleneck
        self.encoder_fc = nn.Linear(self.flat_dim, bottleneck_dim)
        self.decoder_fc = nn.Linear(bottleneck_dim, self.flat_dim)

        # Decoder: symmetric double blocks
        self.decoder_conv = nn.Sequential(
            # Level 4
            deconv_block(4, 8, kernel_size=4, padding=1, output_padding=0),
            conv_block(8, 8, kernel_size=3, stride=1, padding=1),
            # Level 3
            deconv_block(8, 12, kernel_size=4, padding=1, output_padding=0),
            conv_block(12, 12, kernel_size=3, stride=1, padding=1),
            # Level 2
            deconv_block(12, 12, kernel_size=4, padding=1, output_padding=0),
            conv_block(12, 12, kernel_size=3, stride=1, padding=1),
            # Level 1 (final)
            deconv_block(12, 12, kernel_size=4, padding=1, output_padding=0),
            conv_block(12, 1, kernel_size=3, stride=1, padding=1, norm=False, activation=False)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        z = self.encoder_fc(x)

        # Decoder
        x = self.decoder_fc(z)
        x = x.view(-1, *self.conv_shape)
        x_out = self.decoder_conv(x)

        return z, x_out


class Conv3DSparseAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 64, 80, 48), bottleneck_dim=64):
        super().__init__()
        self.input_shape = input_shape
        # self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(1, 64, kernel_size=4, stride=2, padding=1, dropout=True), # -> [16, 80, 88, 104]

            conv_block(64, 32, kernel_size=4, stride=2, padding=1, dropout=True), # -> [32, 40, 44, 52]

            conv_block(32, 16, kernel_size=4, stride=2, padding=1), # -> [32, 20, 22, 26]

            conv_block(16, 8, kernel_size=4, stride=2, padding=1), # -> [64, 10, 11, 13]

        )

        # Determine shape after encoder
        with torch.no_grad():
            dummy = torch.zeros((1, *input_shape))
            conv_out = self.encoder_conv(dummy)
            self.conv_shape = conv_out.shape[1:]
            self.flat_dim = conv_out.numel() // conv_out.shape[0]

        # Bottleneck
        self.encoder_fc = nn.Linear(self.flat_dim, bottleneck_dim)
        self.decoder_fc = nn.Linear(bottleneck_dim, self.flat_dim)

        # Decoder
        self.decoder_conv = nn.Sequential(

            deconv_block(8, 16, kernel_size=4, padding=1, output_padding=0),

            deconv_block(16, 32, kernel_size=4, padding=1, output_padding=0),

            deconv_block(32, 64, kernel_size=4, padding=1, output_padding=0),

            deconv_block(64, 1, kernel_size=4, padding=1, output_padding=0, norm=False, activation=False),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        z = self.encoder_fc(x)

        x = self.decoder_fc(z)
        x = x.view(-1, *self.conv_shape)
        x_out = self.decoder_conv(x)

        return z, x_out
