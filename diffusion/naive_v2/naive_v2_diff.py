import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from .model_conformer_naive import ConformerConvModule
import random


# from https://github.com/fishaudio/fish-diffusion/blob/main/fish_diffusion/modules/convnext.py
# 参考了这个


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


class NaiveV2DiffLayer(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_cond: int,
                 num_heads: int = 4,
                 use_norm: bool = False,
                 conv_only: bool = True,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.1,
                 use_mlp=True,
                 expansion_factor=2,
                 kernel_size=31,
                 wavenet_like=False,
                 conv_model_type='mode1',
                 no_t_emb=False,
                 conv_model_activation='SiLU',
                 GLU_type='GLU'
                 ):
        super().__init__()

        self.conformer = ConformerConvModule(
            dim_model,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            dropout=conv_dropout,
            use_norm=use_norm,
            conv_model_type=conv_model_type,
            activation=conv_model_activation,
            GLU_type=GLU_type
        )
        self.norm = nn.LayerNorm(dim_model) # 请务必注意这是个可学习的层，但模型并没有使用到这一层，在很多地方backward都会出现严重问题，但是在该项目中不会，但直接删除该层会导致加载过往的权重失败，待解决

        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留
        if wavenet_like:
            self.wavenet_like_proj = nn.Conv1d(dim_model, 2 * dim_model, 1)
        else:
            self.wavenet_like_proj = None

        self.no_t_emb = no_t_emb if (no_t_emb is not None) else False
        if not self.no_t_emb:
            self.diffusion_step_projection = nn.Conv1d(dim_model, dim_model, 1)
        else:
            self.diffusion_step_projection = None
        self.condition_projection = nn.Conv1d(dim_cond, dim_model, 1)

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                dropout=atten_dropout,
                activation='gelu',
                norm_first=True
            )
        else:
            self.attn = None

    def forward(self, x, condition=None, diffusion_step=None) -> torch.Tensor:
        res_x = x.transpose(1, 2)
        if self.no_t_emb:
            x = x + self.condition_projection(condition)
        else:
            x = x + self.diffusion_step_projection(diffusion_step) + self.condition_projection(condition)
        x = x.transpose(1, 2)

        if self.attn is not None:
            x = self.attn(x)

        x = self.conformer(x)  # (#batch, dim_model, length)

        if self.wavenet_like_proj is not None:
            x = self.wavenet_like_proj(x.transpose(1, 2)).transpose(1, 2)
            x = F.glu(x, dim=-1)
            return ((x + res_x)/math.sqrt(2.0)).transpose(1, 2), res_x.transpose(1, 2)
        else:
            x = x + res_x
            x = x.transpose(1, 2)
            return x  # (#batch, length, dim_model)


class NaiveV2Diff(nn.Module):
    def __init__(
            self,
            mel_channels=128,
            dim=512,
            use_mlp=True,
            mlp_factor=4,
            condition_dim=256,
            num_layers=20,
            expansion_factor=2,
            kernel_size=31,
            conv_only=True,
            wavenet_like=False,
            use_norm=False,
            conv_model_type='mode1',
            conv_dropout=0.0,
            atten_dropout=0.1,
            no_t_emb=False,
            conv_model_activation='SiLU',
            GLU_type='GLU'
    ):
        super(NaiveV2Diff, self).__init__()
        self.no_t_emb = no_t_emb if (no_t_emb is not None) else False
        self.wavenet_like = wavenet_like
        self.mask_cond_ratio = None

        self.input_projection = nn.Conv1d(mel_channels, dim, 1)
        if self.no_t_emb:
            self.diffusion_embedding = None
        else:
            self.diffusion_embedding = nn.Sequential(
                DiffusionEmbedding(dim),
                nn.Linear(dim, dim * mlp_factor),
                nn.GELU(),
                nn.Linear(dim * mlp_factor, dim),
            )
        
        if use_mlp:
            self.conditioner_projection = nn.Sequential(
                nn.Conv1d(condition_dim, dim * mlp_factor, 1),
                nn.GELU(),
                nn.Conv1d(dim * mlp_factor, dim, 1),
            )
        else:
            self.conditioner_projection = nn.Identity()

        self.residual_layers = nn.ModuleList(
            [
                NaiveV2DiffLayer(
                    dim_model=dim,
                    dim_cond=dim if use_mlp else condition_dim,
                    num_heads=8,
                    use_norm=use_norm,
                    conv_only=conv_only,
                    conv_dropout=conv_dropout,
                    atten_dropout=atten_dropout,
                    use_mlp=use_mlp,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    wavenet_like=wavenet_like,
                    conv_model_type=conv_model_type,
                    no_t_emb=self.no_t_emb,
                    conv_model_activation=conv_model_activation,
                    GLU_type=GLU_type
                )
                for i in range(num_layers)
            ]
        )

        if use_mlp:
            _ = nn.Conv1d(dim * mlp_factor, mel_channels, kernel_size=1)
            nn.init.zeros_(_.weight)
            self.output_projection = nn.Sequential(
                nn.Conv1d(dim, dim * mlp_factor, kernel_size=1),
                nn.GELU(),
                _,
            )
        else:
            self.output_projection = nn.Conv1d(dim, mel_channels, kernel_size=1)
            nn.init.zeros_(self.output_projection.weight)

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

        if self.no_t_emb:
            diffusion_step = None
        else:
            diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)
        condition = self.conditioner_projection(conditioner)

        if self.wavenet_like:
            _sk = []
            for layer in self.residual_layers:
                # conditional mask
                if self.mask_cond_ratio is not None:
                    _mask_cond_ratio = random.choice([True, True, False])
                    if _mask_cond_ratio:
                        # 随机从0到mask_cond_ratio中选择一个数
                        _mask_cond_ratio = random.uniform(0, self.mask_cond_ratio)
                    _conditioner = F.dropout(conditioner, _mask_cond_ratio)
                else:
                    _conditioner = conditioner
                # forward
                x, sk = layer(x, _conditioner, diffusion_step)
                _sk.append(sk)
            x = torch.sum(torch.stack(_sk), dim=0) / math.sqrt(len(self.residual_layers))

        else:
            for layer in self.residual_layers:
                # conditional mask
                if self.mask_cond_ratio is not None:
                    _mask_cond_ratio = random.choice([True, True, False])
                    if _mask_cond_ratio:
                        # 随机从0到mask_cond_ratio中选择一个数
                        _mask_cond_ratio = random.uniform(0, self.mask_cond_ratio)
                    _conditioner = F.dropout(conditioner, _mask_cond_ratio)
                else:
                    _conditioner = conditioner
                # forward
                x = layer(x, condition, diffusion_step)

        # MLP and GLU
        x = self.output_projection(x)  # [B, 128, T]

        return x[:, None] if use_4_dim else x
