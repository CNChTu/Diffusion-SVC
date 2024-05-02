import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# from https://github.com/fishaudio/fish-diffusion/blob/main/fish_diffusion/modules/convnext.py
# 适配了一下，所以不是完全一样的


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer (https://github.com/facebookresearch/ConvNeXt-V2/)
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding"""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        layer_scale_init_value: Optional[float] = 1e-6, version = 'v1'
    ):
        super().__init__()
        self.version = version
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=dilation,
            padding=int(dilation * (7 - 1) / 2),
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        if self.version == 'v2':
            self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        if self.version == 'v1':
            self.gamma = (
                nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
                if layer_scale_init_value is not None and layer_scale_init_value > 0
                else None
            )
        self.diffusion_step_projection = nn.Conv1d(dim, dim, 1)
        self.condition_projection = nn.Conv1d(dim, dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        diffusion_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        x = (
            x
            + self.diffusion_step_projection(diffusion_step)
            + self.condition_projection(condition)
        )

        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.version == 'v2':
            x = self.grn(x)
        x = self.pwconv2(x)
        if self.version == 'v1':
            if self.gamma is not None:
                x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNext(nn.Module):
    def __init__(
        self,
        mel_channels=128,
        dim=512,
        mlp_factor=4,
        condition_dim=256,
        num_layers=20,
        dilation_cycle=4,
        gradient_checkpointing=False,
    ):
        super(ConvNext, self).__init__()

        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        self.diffusion_embedding = nn.Sequential(
            DiffusionEmbedding(dim),
            nn.Linear(dim, dim * mlp_factor),
            nn.GELU(),
            nn.Linear(dim * mlp_factor, dim),
        )
        self.conditioner_projection = nn.Sequential(
            nn.Conv1d(condition_dim, dim * mlp_factor, 1),
            nn.GELU(),
            nn.Conv1d(dim * mlp_factor, dim, 1),
        )

        self.residual_layers = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=dim * mlp_factor,
                    dilation=2 ** (i % dilation_cycle),
                )
                for i in range(num_layers)
            ]
        )
        self.output_projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, mel_channels, kernel_size=1),
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, spec, diffusion_step, cond):
        x = spec
        conditioner = cond
        """

        :param x: [B, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :return:
        """

        # To keep compatibility with DiffSVC, [B, 1, M, T]
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.gelu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)
        condition = self.conditioner_projection(conditioner)

        for layer in self.residual_layers:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, condition, diffusion_step
                )
            else:
                x = layer(x, condition, diffusion_step)

        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x
