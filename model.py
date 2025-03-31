import torch
from torch import nn

device = torch.device("cpu")

ATTRS = {
    "INPUTS_DIMS": [
        (64, 256),
        (32, 512),
        (16, 1024),
    ],
    "OUTPUT_DIMS": (32, 1024),
}

#


class CoordConv2d_onlyY_fixedDims(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CoordConv2d_onlyY_fixedDims, self).__init__()

        self.conv = nn.Conv2d(
            in_channels + 2,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.y_coords = (
            ((torch.arange(1, 1 + dimensions[1]) + 1.0 / 2.0) / (1.0 + dimensions[1]))
            .repeat((1, 1, dimensions[0], 1))
            .to(device)
        )
        self.y_coords_log = ((11.0 + torch.log2(self.y_coords)) / 11.0).to(device)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # Concatenates coordinates channels to input
        x = torch.cat(
            [
                x,
                self.y_coords.expand(x.size(0), 1, width, height),
                self.y_coords_log.expand(x.size(0), 1, width, height),
            ],
            dim=1,
        )

        # Performs convolution
        x = self.conv(x)

        return x


class DeeplySupervizedUnet(nn.Module):

    def convblock_enc(
        self, dims, n_in_channels, n_out_channels, permits_slimming=False
    ):
        if not permits_slimming:
            assert n_in_channels <= n_out_channels
        # https://debuggercafe.com/unet-from-scratch-using-pytorch
        """
        In the original paper implementation, the convolution operations were
        not padded but we are padding them here. This is because, we need the
        output result size to be same as input size.
        """
        biggest_n_channels = max(n_in_channels, n_out_channels)
        conv_op = nn.Sequential(
            CoordConv2d_onlyY_fixedDims(
                dims,
                n_in_channels,
                biggest_n_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(biggest_n_channels),
            nn.Mish(inplace=True),
            CoordConv2d_onlyY_fixedDims(
                dims,
                biggest_n_channels,
                n_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return conv_op

    def convblock_dec(
        self, n_in_channels, n_out_channels, permits_thicken=False, dropout_p=0.0
    ):
        if not permits_thicken:
            assert n_out_channels <= n_in_channels
        # https://debuggercafe.com/unet-from-scratch-using-pytorch
        """
        In the original paper implementation, the convolution operations were
        not padded but we are padding them here. This is because, we need the
        output result size to be same as input size.
        """
        mean_n_channels = (n_in_channels + n_out_channels) // 2
        conv_op = nn.Sequential(
            (nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()),
            nn.Conv2d(
                n_in_channels, mean_n_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.Mish(inplace=True),
            nn.Conv2d(
                mean_n_channels, n_out_channels, kernel_size=3, padding=1, bias=True
            ),
            nn.LeakyReLU(inplace=True),
        )
        return conv_op

    def downsampler(self, n_channels, ratios):
        r_x, r_y = ratios
        assert r_x == int(r_x) and r_y == int(r_y)

        # depthwise-seperable
        conv_op = nn.Sequential(
            (
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=(1, 1 + 2 * (r_y - 1)),
                    padding=(0, r_y - 1),
                    bias=True,
                    groups=n_channels,
                )
                if r_x != 1
                else nn.Identity()
            ),
            (
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=(1 + 2 * (r_x - 1), 1),
                    padding=(r_x - 1, 0),
                    bias=True,
                    groups=n_channels,
                )
                if r_y != 1
                else nn.Identity()
            ),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.MaxPool2d(kernel_size=(r_x, r_y), stride=(r_x, r_y)),
            # nn.LeakyReLU(inplace=True),
        )
        return conv_op

    def upsampler(self, n_channels, ratios):
        r_x, r_y = ratios
        assert r_x == int(r_x) and r_y == int(r_y)
        conv_op = nn.Sequential(
            nn.Upsample(
                scale_factor=ratios,
                mode="bilinear",
                align_corners=False,
            ),
            (
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=(1 + 2 * (r_x - 1), 1),
                    padding=(r_x - 1, 0),
                    bias=True,
                    groups=n_channels,
                )
                if r_x != 1
                else nn.Identity()
            ),
            (
                nn.Conv2d(
                    n_channels,
                    n_channels,
                    kernel_size=(1, 1 + 2 * (r_y - 1)),
                    padding=(0, r_y - 1),
                    bias=True,
                    groups=n_channels,
                )
                if r_y != 1
                else nn.Identity()
            ),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=1,
                bias=True,
            ),
            # nn.LeakyReLU(inplace=True),
        )
        return conv_op

    def exporter(self, n_in_channels, n_intermediate_channels):
        return nn.Sequential(
            nn.Conv2d(
                n_in_channels,
                n_intermediate_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                n_intermediate_channels,
                1,
                kernel_size=1,
                stride=1,
            ),
        )

    def __init__(self):
        INPUTS_PATHS_DIMS = [
            [(64, 256), (32, 256)],
            [(32, 512), (16, 512)],
            [(16, 1024), (16, 512)],
        ]
        N_INPUT_PATHS = len(INPUTS_PATHS_DIMS)
        DIMS_AT_JOIN = (16, 256)
        DIMS_OUTS = [DIMS_AT_JOIN, (32, 512), (32, 1024)]
        DIMS_SMALLEST = (2, 4)
        N_TIER_1_FEATURE_MAPS = 32
        N_TIER_2_FEATURE_MAPS = 64
        N_TIER_3_FEATURE_MAPS = 96
        N_TIER_4_FEATURE_MAPS = 128
        N_TIER_5_FEATURE_MAPS = 256
        N_TIER_6_FEATURE_MAPS = 384
        N_FC_HIDDEN_LAYER_NODES = 512
        N_FC_OUTPUTS = 24
        N_TIER_1_PRE_DIMENSIONALITY_REDUC_FEATURE_MAPS = 16

        super(DeeplySupervizedUnet, self).__init__()

        self.input_1_path = nn.ModuleList(
            [
                self.convblock_enc(INPUTS_PATHS_DIMS[0][0], 1, N_TIER_1_FEATURE_MAPS),
                self.downsampler(N_TIER_1_FEATURE_MAPS, (2, 1)),
                self.convblock_enc(
                    INPUTS_PATHS_DIMS[0][1],
                    N_TIER_1_FEATURE_MAPS,
                    N_TIER_2_FEATURE_MAPS,
                ),
                self.downsampler(N_TIER_2_FEATURE_MAPS, (2, 1)),
            ]
        )
        self.input_2_path = nn.ModuleList(
            [
                self.convblock_enc(INPUTS_PATHS_DIMS[1][0], 1, N_TIER_1_FEATURE_MAPS),
                self.downsampler(N_TIER_1_FEATURE_MAPS, (2, 1)),
                self.convblock_enc(
                    INPUTS_PATHS_DIMS[1][1],
                    N_TIER_1_FEATURE_MAPS,
                    N_TIER_2_FEATURE_MAPS,
                ),
                self.downsampler(N_TIER_2_FEATURE_MAPS, (1, 2)),
            ]
        )
        self.input_3_path = nn.ModuleList(
            [
                self.convblock_enc(INPUTS_PATHS_DIMS[2][0], 1, N_TIER_1_FEATURE_MAPS),
                self.downsampler(N_TIER_1_FEATURE_MAPS, (1, 2)),
                self.convblock_enc(
                    INPUTS_PATHS_DIMS[2][1],
                    N_TIER_1_FEATURE_MAPS,
                    N_TIER_2_FEATURE_MAPS,
                ),
                self.downsampler(N_TIER_2_FEATURE_MAPS, (1, 2)),
            ]
        )
        self.joined_path_encode = nn.ModuleList(
            [
                self.convblock_enc(
                    DIMS_AT_JOIN,
                    N_INPUT_PATHS * N_TIER_2_FEATURE_MAPS,
                    N_TIER_3_FEATURE_MAPS,
                    permits_slimming=True,
                ),
                self.downsampler(N_TIER_3_FEATURE_MAPS, (2, 4)),  # 8 x 64
                self.convblock_enc(
                    (8, 64),
                    N_TIER_3_FEATURE_MAPS,
                    N_TIER_4_FEATURE_MAPS,
                ),
                self.downsampler(N_TIER_4_FEATURE_MAPS, (2, 4)),  # 4 x 16
                self.convblock_enc(
                    (4, 16),
                    N_TIER_4_FEATURE_MAPS,
                    N_TIER_5_FEATURE_MAPS,
                ),
                self.downsampler(N_TIER_5_FEATURE_MAPS, (2, 4)),  # 2 x 4
                self.convblock_enc(
                    DIMS_SMALLEST,
                    N_TIER_5_FEATURE_MAPS,
                    N_TIER_6_FEATURE_MAPS,
                ),
            ]
        )
        self.skip_inp_1_tier_0_scaler = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            self.upsampler(1, (1, 4)),
        )
        self.skip_inp_2_tier_0_scaler = self.upsampler(1, (1, 2))
        self.skip_inp_3_tier_0_scaler = self.upsampler(1, (2, 1))
        self.skip_inp_1_tier_1_scaler = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
            self.upsampler(N_TIER_1_FEATURE_MAPS, (1, 4)),
        )
        self.skip_inp_2_tier_1_scaler = self.upsampler(N_TIER_1_FEATURE_MAPS, (1, 2))
        self.skip_inp_3_tier_1_scaler = self.upsampler(N_TIER_1_FEATURE_MAPS, (2, 1))
        self.skip_inp_1_tier_2_scaler = self.upsampler(N_TIER_2_FEATURE_MAPS, (1, 2))
        self.skip_inp_2_tier_2_scaler = self.upsampler(N_TIER_2_FEATURE_MAPS, (2, 1))
        self.skip_inp_3_tier_2_scaler = self.upsampler(N_TIER_2_FEATURE_MAPS, (2, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),  # default : start_dim=1
            nn.Dropout(p=0.1),
            nn.Linear(
                N_TIER_6_FEATURE_MAPS * DIMS_SMALLEST[0] * DIMS_SMALLEST[1],
                N_FC_HIDDEN_LAYER_NODES,
            ),
            nn.Mish(True),
            nn.Dropout(p=0.05),
            nn.Linear(
                N_FC_HIDDEN_LAYER_NODES,
                N_FC_OUTPUTS,
            ),
            nn.ReLU(True),
            nn.Dropout(p=0.01),
        )  # so the expanding path can factor in hollistic information about the image as a whole and not just local patterns
        self.path_decode = nn.ModuleList(
            [
                self.upsampler(N_TIER_6_FEATURE_MAPS, (2, 4)),  # 4 x 16
                self.convblock_dec(
                    N_FC_OUTPUTS + N_TIER_6_FEATURE_MAPS + N_TIER_5_FEATURE_MAPS,
                    N_TIER_5_FEATURE_MAPS,
                    dropout_p=0.1,
                ),
                self.upsampler(N_TIER_5_FEATURE_MAPS, (2, 4)),  # 8 x 64
                self.convblock_dec(
                    N_FC_OUTPUTS + N_TIER_5_FEATURE_MAPS + N_TIER_4_FEATURE_MAPS,
                    N_TIER_4_FEATURE_MAPS,
                    dropout_p=0.1,
                ),
                self.upsampler(N_TIER_4_FEATURE_MAPS, (2, 4)),  # 16 x 256
                self.convblock_dec(
                    N_FC_OUTPUTS + N_TIER_4_FEATURE_MAPS + N_TIER_3_FEATURE_MAPS,
                    N_TIER_3_FEATURE_MAPS,
                    dropout_p=0.1,
                ),
                self.upsampler(N_TIER_3_FEATURE_MAPS, (2, 2)),  # 32 x 512
                self.convblock_dec(
                    N_FC_OUTPUTS
                    + N_TIER_3_FEATURE_MAPS
                    + N_INPUT_PATHS * N_TIER_2_FEATURE_MAPS,
                    N_TIER_2_FEATURE_MAPS,
                    dropout_p=0.05,
                ),
                self.upsampler(N_TIER_2_FEATURE_MAPS, (1, 2)),  # 32 x 1024
                self.convblock_dec(
                    N_FC_OUTPUTS
                    + N_TIER_2_FEATURE_MAPS
                    + N_INPUT_PATHS * N_TIER_1_FEATURE_MAPS,
                    N_TIER_1_FEATURE_MAPS,
                    dropout_p=0.05,
                ),
                self.exporter(
                    N_FC_OUTPUTS + N_TIER_1_FEATURE_MAPS + N_INPUT_PATHS,
                    N_TIER_1_PRE_DIMENSIONALITY_REDUC_FEATURE_MAPS,
                ),
            ]
        )
        self.deep_superv_out_t4 = self.exporter(
            N_TIER_4_FEATURE_MAPS,
            N_TIER_1_PRE_DIMENSIONALITY_REDUC_FEATURE_MAPS,
        )
        self.deep_superv_out_t3 = self.exporter(
            N_TIER_3_FEATURE_MAPS,
            N_TIER_1_PRE_DIMENSIONALITY_REDUC_FEATURE_MAPS,
        )
        self.deep_superv_out_t2 = self.exporter(
            N_TIER_2_FEATURE_MAPS,
            N_TIER_1_PRE_DIMENSIONALITY_REDUC_FEATURE_MAPS,
        )

    def forward(self, x1, x2, x3, do_deep_supervision=False):
        skip_t_0 = torch.cat(
            [
                self.skip_inp_1_tier_0_scaler(x1),
                self.skip_inp_2_tier_0_scaler(x2),
                self.skip_inp_3_tier_0_scaler(x3),
            ],
            dim=1,
        )

        in_path_1 = self.input_1_path[0](x1)
        skip_1_1 = self.skip_inp_1_tier_1_scaler(in_path_1)
        in_path_1 = self.input_1_path[1](in_path_1)
        in_path_1 = self.input_1_path[2](in_path_1)
        skip_1_2 = self.skip_inp_1_tier_2_scaler(in_path_1)
        in_path_1 = self.input_1_path[3](in_path_1)

        in_path_2 = self.input_2_path[0](x2)
        skip_2_1 = self.skip_inp_2_tier_1_scaler(in_path_2)
        in_path_2 = self.input_2_path[1](in_path_2)
        in_path_2 = self.input_2_path[2](in_path_2)
        skip_2_2 = self.skip_inp_2_tier_2_scaler(in_path_2)
        in_path_2 = self.input_2_path[3](in_path_2)

        in_path_3 = self.input_3_path[0](x3)
        skip_3_1 = self.skip_inp_3_tier_1_scaler(in_path_3)
        in_path_3 = self.input_3_path[1](in_path_3)
        in_path_3 = self.input_3_path[2](in_path_3)
        skip_3_2 = self.skip_inp_3_tier_2_scaler(in_path_3)
        in_path_3 = self.input_3_path[3](in_path_3)

        enc = self.joined_path_encode[0](
            torch.cat([in_path_1, in_path_2, in_path_3], dim=1)
        )
        skip_t_3 = enc
        enc = self.joined_path_encode[1](enc)
        enc = self.joined_path_encode[2](enc)
        skip_t_4 = enc
        enc = self.joined_path_encode[3](enc)
        enc = self.joined_path_encode[4](enc)
        skip_t_5 = enc
        enc = self.joined_path_encode[5](enc)
        enc = self.joined_path_encode[6](enc)

        fc = self.fc(enc)
        fc_delinearized = torch.unsqueeze(torch.unsqueeze(fc, 2), 3)

        dec = self.path_decode[0](enc)
        dec = self.path_decode[1](
            torch.cat([fc_delinearized.repeat((1, 1, 4, 16)), skip_t_5, dec], 1)
        )
        dec = self.path_decode[2](dec)
        dec = self.path_decode[3](
            torch.cat([fc_delinearized.repeat((1, 1, 8, 64)), skip_t_4, dec], 1)
        )
        deep_superv_out_t4 = (
            self.deep_superv_out_t4(dec) if do_deep_supervision else None
        )
        dec = self.path_decode[4](dec)
        dec = self.path_decode[5](
            torch.cat([fc_delinearized.repeat((1, 1, 16, 256)), skip_t_3, dec], 1)
        )
        deep_superv_out_t3 = (
            self.deep_superv_out_t3(dec) if do_deep_supervision else None
        )
        dec = self.path_decode[6](dec)
        dec = self.path_decode[7](
            torch.cat(
                [
                    fc_delinearized.repeat((1, 1, 32, 512)),
                    skip_1_2,
                    skip_2_2,
                    skip_3_2,
                    dec,
                ],
                1,
            )
        )

        deep_superv_out_t2 = (
            self.deep_superv_out_t2(dec) if do_deep_supervision else None
        )
        dec = self.path_decode[8](dec)
        dec = self.path_decode[9](
            torch.cat(
                [
                    fc_delinearized.repeat((1, 1, 32, 1024)),
                    skip_1_1,
                    skip_2_1,
                    skip_3_1,
                    dec,
                ],
                1,
            )
        )
        dec = self.path_decode[10](
            torch.cat([fc_delinearized.repeat((1, 1, 32, 1024)), skip_t_0, dec], 1)
        )

        return (
            (deep_superv_out_t4, deep_superv_out_t3, deep_superv_out_t2, dec)
            if do_deep_supervision
            else dec
        )


#

if __name__ == "__main__":
    from torchinfo import summary

    with torch.no_grad():
        torch_model = DeeplySupervizedUnet().to("cpu")

        batch_size = 32
        summary(
            torch_model,
            [
                (batch_size, 1, *ATTRS["INPUTS_DIMS"][0]),
                (batch_size, 1, *ATTRS["INPUTS_DIMS"][1]),
                (batch_size, 1, *ATTRS["INPUTS_DIMS"][2]),
            ],
        )
