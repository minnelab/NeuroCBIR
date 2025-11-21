# import torch
# import torch.nn as nn

# def group_norm(channels):
#     if channels >= 32:
#         return nn.GroupNorm(8, channels)
#     elif channels >= 16:
#         return nn.GroupNorm(4, channels)
#     else:
#         return nn.GroupNorm(2, channels)

# def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=True, activation=True, dropout=None):
#     layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
#     if norm:
#         layers.append(group_norm(out_channels))
#     if activation:
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#     if dropout is not None:
#         layers.append(nn.Dropout3d(dropout))
#     return nn.Sequential(*layers)

# def deconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0, norm=True, activation=True):
#     layers = [nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
#     if norm:
#         layers.append(group_norm(out_channels))
#     if activation:
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#     return nn.Sequential(*layers)

# class Conv3DAutoencoder(nn.Module):
#     def __init__(self, input_shape=(1, 176, 208, 160), bottleneck_dim=64, n_filters=[32, 32, 32, 32]):
#         super().__init__()
#         self.input_shape = input_shape
#         self.bottleneck_dim = bottleneck_dim
#         input_channels = output_channels = input_shape[0]

#         # Encoder
#         self.encoder_conv = nn.Sequential(
#             conv_block(input_channels, n_filters[0], kernel_size=4, stride=2, padding=1, dropout=0.1),
#             conv_block(n_filters[0], n_filters[1], kernel_size=4, stride=2, padding=1, dropout=0.1),
#             conv_block(n_filters[1], n_filters[2], kernel_size=4, stride=2, padding=1),
#             conv_block(n_filters[2], n_filters[3], kernel_size=4, stride=2, padding=1),
#         )

#         if bottleneck_dim is not None:
#             # Determine shape after encoder
#             with torch.no_grad():
#                 dummy = torch.zeros((1, *input_shape))
#                 conv_out = self.encoder_conv(dummy)
#                 self.conv_shape = conv_out.shape[1:]
#                 self.flat_dim = conv_out.numel() // conv_out.shape[0]

#             # Bottleneck
#             self.encoder_fc = nn.Linear(self.flat_dim, bottleneck_dim)
#             self.decoder_fc = nn.Linear(bottleneck_dim, self.flat_dim)

#         # Decoder
#         self.decoder_conv = nn.Sequential(
#             deconv_block(n_filters[3], n_filters[2], kernel_size=4, padding=1, output_padding=0),
#             deconv_block(n_filters[2], n_filters[1], kernel_size=4, padding=1, output_padding=0),
#             deconv_block(n_filters[1], n_filters[0], kernel_size=4, padding=1, output_padding=0),
#             deconv_block(n_filters[0], output_channels, kernel_size=4, padding=1, output_padding=0, norm=False, activation=False),
#         )

#     def forward(self, x):
#         x = z = self.encoder_conv(x)

#         if self.bottleneck_dim is not None:
#             x = torch.flatten(x, start_dim=1)
#             z = self.encoder_fc(x)
#             x = self.decoder_fc(z)
#             x = x.view(-1, *self.conv_shape)

#         x_out = self.decoder_conv(x)

#         return z, x_out
