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
    def __init__(self, input_shape=(1, 176, 208, 160), bottleneck_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.bottleneck_dim = bottleneck_dim
        input_channels = output_channels = input_shape[0]

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(input_channels, 32, kernel_size=4, stride=2, padding=1, dropout=0.1),
            conv_block(32, 24, kernel_size=4, stride=2, padding=1, dropout=0.1),
            conv_block(24, 16, kernel_size=4, stride=2, padding=1),
            conv_block(16, 8, kernel_size=4, stride=2, padding=1),
        )

        if bottleneck_dim is not None:
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
            deconv_block(16, 24, kernel_size=4, padding=1, output_padding=0),
            deconv_block(24, 32, kernel_size=4, padding=1, output_padding=0),
            deconv_block(32, output_channels, kernel_size=4, padding=1, output_padding=0, norm=False, activation=False),
        )

    def forward(self, x):
        x = z = self.encoder_conv(x)

        if self.bottleneck_dim is not None:
            x = torch.flatten(x, start_dim=1)
            z = self.encoder_fc(x)
            x = self.decoder_fc(z)
            x = x.view(-1, *self.conv_shape)

        x_out = self.decoder_conv(x)

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
        self.bottleneck_dim = bottleneck_dim
        input_channels = output_channels = input_shape[0]

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(input_channels, 64, kernel_size=4, stride=2, padding=1, dropout=True), # -> [16, 80, 88, 104]

            conv_block(64, 32, kernel_size=4, stride=2, padding=1, dropout=True), # -> [32, 40, 44, 52]

            conv_block(32, 16, kernel_size=4, stride=2, padding=1), # -> [32, 20, 22, 26]

            # conv_block(16, 8, kernel_size=4, stride=2, padding=1), # -> [64, 10, 11, 13]

        )

        # Determine shape after encoder
        if bottleneck_dim is not None:
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

            # deconv_block(8, 16, kernel_size=4, padding=1, output_padding=0),

            deconv_block(16, 32, kernel_size=4, padding=1, output_padding=0),

            deconv_block(32, 64, kernel_size=4, padding=1, output_padding=0),

            deconv_block(64, output_channels, kernel_size=4, padding=1, output_padding=0, norm=False, activation=False),
        )

    def forward(self, x):
        x = z = self.encoder_conv(x)

        if self.bottleneck_dim is not None:
            x = torch.flatten(x, start_dim=1)
            z = self.encoder_fc(x)
            x = self.decoder_fc(z)
            x = x.view(-1, *self.conv_shape)

        x_out = self.decoder_conv(x)

        return z, x_out


class ResConv3DAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 176, 208, 160), bottleneck_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.bottleneck_dim = bottleneck_dim
        input_channels = output_channels = input_shape[0]

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(input_channels, 32, kernel_size=4, stride=2, padding=1, dropout=0.1),
            conv_block(32, 24, kernel_size=4, stride=2, padding=1, dropout=0.1),
            conv_block(24, 16, kernel_size=4, stride=2, padding=1),
            conv_block(16, 8, kernel_size=4, stride=2, padding=1),
        )

        if bottleneck_dim is not None:
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
            deconv_block(16, 24, kernel_size=4, padding=1, output_padding=0),
            deconv_block(24, 32, kernel_size=4, padding=1, output_padding=0),
            deconv_block(32, output_channels, kernel_size=4, padding=1, output_padding=0, norm=False, activation=False),
        )

    def forward(self, x):
        x = z = self.encoder_conv(x)

        if self.bottleneck_dim is not None:
            x = torch.flatten(x, start_dim=1)
            z = self.encoder_fc(x)
            x = self.decoder_fc(z)
            x = x.view(-1, *self.conv_shape)

        x_out = self.decoder_conv(x)

        return z, x_out

import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual double conv block from earlier
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 norm=True, activation=True, dropout=None):
        super(ResidualConvBlock, self).__init__()
        self.use_projection = in_channels != out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not norm)
        self.norm1 = nn.GroupNorm(8, out_channels) if norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not norm)
        self.norm2 = nn.GroupNorm(8, out_channels) if norm else nn.Identity()

        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1) if self.use_projection else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True) if activation else nn.Identity()
        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        identity = self.projection(x)
        out = self.relu1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.final_relu(out + identity)
        return self.dropout(out)

# Autoencoder
class ResConv3DAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 176, 208, 160), bottleneck_dim=64):
        super().__init__()
        self.input_shape = input_shape
        self.bottleneck_dim = bottleneck_dim
        input_channels = output_channels = input_shape[0]

        # Encoder
        self.enc1 = nn.Sequential(
            ResidualConvBlock(input_channels, 32, norm=False),
            nn.MaxPool3d(2)
        )
        self.enc2 = nn.Sequential(
            ResidualConvBlock(32, 64, dropout=0.1),
            nn.MaxPool3d(2)
        )
        self.enc3 = nn.Sequential(
            ResidualConvBlock(64, 128),
            nn.MaxPool3d(2)
        )
        self.enc4 = nn.Sequential(
            ResidualConvBlock(128, 16),
            # nn.MaxPool3d(2)
        )

        # Dummy pass to determine flattened shape
        if bottleneck_dim is not None:
            with torch.no_grad():
                dummy = torch.zeros((1, *input_shape))
                x = self.enc1(dummy)
                x = self.enc2(x)
                x = self.enc3(x)
                x = self.enc4(x)
                self.conv_shape = x.shape[1:]
                self.flat_dim = x.numel() // x.shape[0]

            self.encoder_fc = nn.Linear(self.flat_dim, bottleneck_dim)
            self.decoder_fc = nn.Linear(bottleneck_dim, self.flat_dim)

        # Decoder
        # self.dec1 = nn.Sequential(
        #     nn.ConvTranspose3d(8, 16, kernel_size=2, stride=2),
        #     ResidualConvBlock(16, 16)
        # )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2),
            ResidualConvBlock(16, 128)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            ResidualConvBlock(128, 64)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            ResidualConvBlock(64, 32),
            conv_block(32, 1, kernel_size=3, stride=1, padding=1, norm=False, activation=False)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        if self.bottleneck_dim is not None:
            x_flat = torch.flatten(x, start_dim=1)
            z = self.encoder_fc(x_flat)
            x = self.decoder_fc(z)
            x = x.view(-1, *self.conv_shape)
        else:
            z = x

        # x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        return z, x
