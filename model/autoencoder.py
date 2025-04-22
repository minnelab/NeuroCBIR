import torch
import torch.nn as nn

class Conv3DAutoencoder(nn.Module):
    def __init__(self, input_shape=(1, 160, 176, 208), bottleneck_dim=1024):
        super().__init__()
        self.input_shape = input_shape
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),  # -> [16, 80, 88, 104]
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 40, 44, 52]
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),  # -> [32, 20, 22, 26]
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # -> [64, 10, 11, 13]
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1),  # -> [64, 5, 6, 7]
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
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

        # Decoder (use correct output_padding to match original shape)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 0, 0)),  # [64, 10, 11, 13]
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 20, 22, 26]
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [32, 40, 44, 52]
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 80, 88, 104]
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [1, 160, 176, 208]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        z = self.encoder_fc(x)

        x = self.decoder_fc(z)
        x = x.view(-1, *self.conv_shape)
        x_out = self.decoder_conv(x)

        return z, x_out
