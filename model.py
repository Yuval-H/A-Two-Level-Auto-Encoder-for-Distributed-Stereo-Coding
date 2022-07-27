import torch.nn as nn
import torch

import pytorch_msssim

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)



class DSC_stereo_compression(nn.Module):
    """
    Distributed Source Stereo compression from "A TWO-LEVEL AUTO-ENCODER
    FOR DISTRIBUTED STEREO CODING"

    -- link to paper -- , Yuval Harel and Prof. Shai Avidan

    Args:
        n_ch_comp (int): Number of channels for the compressed representation [ bpp = (4*N)/(32**2) ]

        Optional: N (int): Number of channels (base auto-encoder)
    """

    def __init__(self, N=128, n_ch_comp=8, **kwargs):
        super().__init__()

        self.out_channel_N = N
        self.n_ch_comp = n_ch_comp
        self.g_a = nn.Sequential(
            ResidualBlock(3, 3),
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.g_a22 = nn.Sequential(
            conv3x3(N, 64, stride=1),
            ResidualBlock(64, 64),
            ResidualBlockWithStride(64, 64, stride=2),
            AttentionBlock(64),
            conv3x3(64, 32, stride=1),
            ResidualBlock(32, 32),
            conv3x3(32, n_ch_comp, stride=1),
            AttentionBlock(n_ch_comp),
        )

        self.g_s22 = nn.Sequential(
            AttentionBlock(n_ch_comp),
            conv3x3(n_ch_comp, 32, stride=1),
            ResidualBlock(32, 32),
            conv3x3(32, 64, stride=1),
            ResidualBlock(64, 64),
            ResidualBlockUpsample(64, N, 2),
            ResidualBlock(N, N),
        )

        self.g_z1hat_z2 = nn.Sequential(
            AttentionBlock(2*N),
            ResidualBlock(2*N, 2*N),
            ResidualBlock(2*N, N),
            AttentionBlock(N),
            ResidualBlock(N, N),
        )

        self.g_rec1_im2 = nn.Sequential(
            AttentionBlock(6),
            ResidualBlock(6, 6),
            AttentionBlock(6),
            ResidualBlock(6, 3),
            AttentionBlock(3),
            ResidualBlock(3, 3),
            AttentionBlock(3),
        )

        self.g_rec1_im2_new = nn.Sequential(
            AttentionBlock(6),
            ResidualBlock(6, 3),
            ResidualBlock(3, 3),
            AttentionBlock(3),
            ResidualBlock(3, 3),
        )

    def forward(self, im1, im2):
        quant_noise_feature = torch.zeros(im1.size(0), self.out_channel_N, im1.size(2) // 16,
                                          im1.size(3) // 16).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -8, 8)

        channels = self.n_ch_comp
        quant_noise_feature2 = torch.zeros(im1.size(0), channels, im1.size(2) // 32, im1.size(3) // 32).cuda()
        quant_noise_feature2 = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature2), -8, 8)

        z1 = self.g_a(im1)
        z2 = self.g_a(im2)
        if self.training:
            compressed_z1 = z1 + quant_noise_feature
            compressed_z2 = z2 + quant_noise_feature
        else:
            compressed_z1 = torch.round(z1)
            compressed_z2 = torch.round(z2)

        # further compress z1 (quantize features)
        if self.training:
            z1_down = self.g_a22(z1) + quant_noise_feature2
        else:
            z1_down = torch.round(self.g_a22(z1)/16)*16

        # clamp feature to 4 bits
        # Explanation: features range between -+128, quantize bins of size 16. overall 16 option -> log2(16) = 4 bits
        z1_down = torch.clamp(z1_down, -128, 128)


        z1_hat = self.g_s22(z1_down)

        #####################################################################################
        # Decoder Side,  Using z1_down and im2 (In paper, Z_X and Y)
        #####################################################################################
        # cat z1_hat, z2 -> get z1_hat_hat
        z_cat = torch.cat((z1_hat, z2), 1)

        # Option, for ablation studies: remove SI image/ compressed original image. (replace with black image).
        #z_cat = torch.cat((torch.zeros_like(z1_hat), z2), 1)
        #z_cat = torch.cat((z1_hat, torch.zeros_like(z2)), 1)

        z1_hat_hat = self.g_z1hat_z2(z_cat)

        # recon images
        final_im1_recon = self.g_s(z1_hat_hat)

        im1_hat = self.g_s(compressed_z1)
        im2_hat = self.g_s(compressed_z2)

        # distortion
        useL1 = True
        use_msssim = False
        if useL1:
            loss_l1 = nn.L1Loss()

            loss_base_ae = 0.5 * loss_l1(im1_hat.clamp(0., 1.), im1) + 0.5 * loss_l1(im2_hat.clamp(0., 1.), im2)
            loss_feature_space = loss_l1(z1_hat_hat, torch.round(z1/16)*16)
            loss_final_ae = loss_l1(final_im1_recon.clamp(0., 1.), im1)
        elif use_msssim:
            loss_base_ae = 1 - (0.5*(pytorch_msssim.ms_ssim( final_im1_recon.clamp(0., 1.), im1, data_range=1.0) +
                            pytorch_msssim.ms_ssim(im2_hat.clamp(0., 1.), im2, data_range=1.0)))
            loss_feature_space = 1  # not for training use
            loss_final_ae = 1 - pytorch_msssim.ms_ssim(final_im1_recon.clamp(0., 1.), im1, data_range=1.0)
        else: # mse
            loss_base_ae = 0.5*torch.mean((im1_hat.clamp(0., 1.) - im1).pow(2)) + 0.5*torch.mean((im2_hat.clamp(0., 1.) - im2).pow(2))
            loss_feature_space = torch.mean((z1_hat_hat - z1).pow(2))
            loss_final_ae = torch.mean((final_im1_recon.clamp(0., 1.) - im1).pow(2))

        if self.training:
            return loss_base_ae, loss_final_ae, loss_feature_space, torch.clip(final_im1_recon, 0, 1)
        else:
            return (z1-z1_hat,z1_hat, z2), loss_final_ae, torch.clip(final_im1_recon, 0, 1), z1_down
