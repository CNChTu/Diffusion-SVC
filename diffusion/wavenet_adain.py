import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Mish
from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerConfig


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
 
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
 
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class AdaIN(nn.Module):
    def __init__(self, in_channels, style_dim, memory_efficient=True):
        super().__init__()
        self.silu = MemoryEfficientSwish() if memory_efficient else nn.SiLU()
        self.style = nn.Linear(style_dim, in_channels * 2)
        nn.init.zeros_(self.style.weight)
        nn.init.zeros_(self.style.bias)

    def forward(self, x, cond):
        style = self.style(self.silu(cond))
        gamma, beta = torch.chunk(style, 2, dim=1)
        return (1 + gamma[:,None,:]) * x + beta[:,None,:]


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation, kernel_size=3,
                 no_t_emb=False):
        super().__init__()
        self.residual_channels = residual_channels
        self.ln = nn.LayerNorm(residual_channels)
        self.adain = AdaIN(residual_channels, residual_channels)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=kernel_size,
            padding=dilation if (kernel_size == 3) else int((kernel_size-1) * dilation / 2),
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        self.no_t_emb = no_t_emb if (no_t_emb is not None) else False

    def forward(self, x, conditioner, diffusion_step):
        conditioner = self.conditioner_projection(conditioner)

        if not self.no_t_emb:
            diffusion_step = self.diffusion_projection(diffusion_step)
            y = self.adain(self.ln(x.transpose(-1, -2)), diffusion_step).transpose(-1, -2)
        else:
            y = x

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256, dilation=1, kernel_size=3,
                 transformer_use=False, transformer_roformer_use=False, transformer_n_layers=2, transformer_n_head=4,
                 no_t_emb=False):
        super().__init__()
        self.no_t_emb = no_t_emb if (no_t_emb is not None) else False
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=n_hidden,
                residual_channels=n_chans,
                dilation=(2 ** (i % dilation)) if (dilation != 1) else 1,
                kernel_size=kernel_size,
                no_t_emb=self.no_t_emb
            )
            for i in range(n_layers)
        ])
        self.transformer_roformer_use = transformer_roformer_use if (transformer_roformer_use is not None) else False
        if transformer_use:
            if transformer_roformer_use:
                self.transformer = RoFormerEncoder(
                    RoFormerConfig(
                        hidden_size=n_chans,
                        max_position_embeddings=4096,
                        num_attention_heads=transformer_n_head,
                        num_hidden_layers=transformer_n_layers,
                        add_cross_attention=False
                    )
                )
            else:
                transformer_layer = nn.TransformerEncoderLayer(
                    d_model=n_chans,
                    nhead=transformer_n_head,
                    dim_feedforward=n_chans * 4,
                    dropout=0.1,
                    activation='gelu',
                    norm_first=True
                )
                self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_n_layers)
        else:
            self.transformer = None

        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = F.relu(x)
        if self.no_t_emb:
            diffusion_step = None
        else:
            diffusion_step = self.diffusion_embedding(diffusion_step)
            diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        if self.transformer is not None:
            if self.transformer_roformer_use:
                x = self.transformer(x.transpose(1, 2))[0].transpose(1, 2)
            else:
                x = self.transformer(x.transpose(1, 2)).transpose(1, 2)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x[:, None, :, :]
