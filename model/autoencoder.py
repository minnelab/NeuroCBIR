import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=True, activation=True, dropout=False):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if norm:
        layers.append(nn.InstanceNorm3d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    if dropout:
        layers.append(nn.Dropout3d(0.1))
    return nn.Sequential(*layers)

def deconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0, norm=True, activation=True):
    layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
    if norm:
        layers.append(nn.InstanceNorm3d(out_channels))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class Conv3DAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 160, 176, 208), bottleneck_dim=1024):
        super().__init__()
        self.input_shape = input_shape
        # self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            conv_block(1, 8, dropout=True, padding=1),
            nn.MaxPool3d(2),                    # -> [16, 80, 88, 104]

            # conv_block(16, 16, padding=1),
            conv_block(8, 16, padding=1, dropout=True),
            nn.MaxPool3d(2),                    # -> [32, 40, 44, 52]

            # conv_block(32, 32, padding=1),
            conv_block(16, 32, padding=1),
            nn.MaxPool3d(2),                    # -> [32, 20, 22, 26]

            # conv_block(32, 32, padding=1),
            conv_block(32, 64, padding=1),
            nn.MaxPool3d(2),                    # -> [64, 10, 11, 13]
        
            # conv_block(64, 64, padding=1),
            conv_block(64, 64, padding=1),
            nn.MaxPool3d(2, padding=(0,1,1)),                    # -> [64, 5, 6, 7]
            
            conv_block(64, 64, padding=1),
            nn.MaxPool3d(2, padding=(1,0,1)),                    # -> [64, 3, 3, 4]
        )

        # Determine shape after encoder
        # with torch.no_grad():
        #     dummy = torch.zeros((1, *input_shape))
        #     conv_out = self.encoder_conv(dummy)
        #     self.conv_shape = conv_out.shape[1:]
        #     self.flat_dim = conv_out.numel() // conv_out.shape[0]

        # Bottleneck
        # self.encoder_fc = nn.Linear(self.flat_dim, bottleneck_dim)
        # self.decoder_fc = nn.Linear(bottleneck_dim, self.flat_dim)

        # Decoder
        self.decoder_conv = nn.Sequential(
            deconv_block(64, 64, padding=1, output_padding=(0, 1, 0)),
            # conv_block(64, 64),

            deconv_block(64, 64, padding=1, output_padding=(1, 0, 0)),
            # conv_block(64, 64),

            deconv_block(64, 32, padding=1, output_padding=1),
            # conv_block(64, 64),

            deconv_block(32, 16, padding=1, output_padding=1),
            # conv_block(32, 32),

            deconv_block(16, 8, padding=1, output_padding=1),
            # conv_block(32, 32),

            deconv_block(8, 8, padding=1, output_padding=1),
            conv_block(8, 1, activation=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder_conv(x)
        # x = torch.flatten(x, start_dim=1)
        # z = self.encoder_fc(x)

        # x = self.decoder_fc(z)
        # x = x.view(-1, *self.conv_shape)
        x_out = self.decoder_conv(z)

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

            deconv_block(64, 1, kernel_size=4, padding=1, output_padding=0),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        z = self.encoder_fc(x)

        x = self.decoder_fc(z)
        x = x.view(-1, *self.conv_shape)
        x_out = self.decoder_conv(x)

        return z, x_out
