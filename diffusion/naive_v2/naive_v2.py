import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm
from .model_conformer_naive import ConformerNaiveEncoder


class Unit2MelNaiveV2(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_hidden=None,  # 废弃
            use_speaker_encoder=False,
            speaker_encoder_out_channels=256,
            net_fn=None,
    ):
        super().__init__()
        self.l2reg_loss = 0  # 废弃
        # catch None
        assert net_fn is not None
        self.n_layers = net_fn.n_layers if (net_fn.n_layers is not None) else 6
        self.n_chans = net_fn.n_chans if (net_fn.n_chans is not None) else 256
        self.out_put_norm = net_fn.out_put_norm if (net_fn.out_put_norm is not None) else False
        self.simple_stack = net_fn.simple_stack if (net_fn.simple_stack is not None) else False

        if net_fn.type == 'LYNXNet' or net_fn.type == 'NaiveNet':
            self.expansion_factor = net_fn.expansion_factor if (net_fn.expansion_factor is not None) else 2
            self.kernel_size = net_fn.kernel_size if (net_fn.kernel_size is not None) else 31
            self.conv_model_type = net_fn.conv_model_type if (net_fn.conv_model_type is not None) else 'mode1'
            self.num_heads = net_fn.num_heads if (net_fn.num_heads is not None) else 8
            self.use_norm = net_fn.use_norm if (net_fn.use_norm is not None) else False
            self.conv_only = net_fn.conv_only if (net_fn.conv_only is not None) else True
            self.conv_dropout = net_fn.conv_dropout if (net_fn.conv_dropout is not None) else 0.0
            self.atten_dropout = net_fn.atten_dropout if (net_fn.atten_dropout is not None) else 0.1

            self.decoder = ConformerNaiveEncoder(
                num_layers=self.n_layers,
                num_heads=self.num_heads,
                dim_model=self.n_chans,
                expansion_factor=self.expansion_factor,
                kernel_size=self.kernel_size,
                use_norm=self.use_norm,
                conv_only=self.conv_only,
                conv_dropout=self.conv_dropout,
                atten_dropout=self.atten_dropout,
                conv_model_type=self.conv_model_type,
            )
        else:
            raise ValueError(f'net_fn.type={net_fn.type} is not supported')

        self.f0_embed = nn.Linear(1, self.n_chans)
        self.volume_embed = nn.Linear(1, self.n_chans)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, self.n_chans, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        self.use_speaker_encoder = use_speaker_encoder
        if use_speaker_encoder:
            self.spk_embed = nn.Linear(speaker_encoder_out_channels, self.n_chans, bias=False)
        else:
            if n_spk is not None and n_spk > 1:
                self.spk_embed = nn.Embedding(n_spk, self.n_chans)

        # conv in stack
        if self.simple_stack:
            self.stack = nn.Conv1d(input_channel, self.n_chans, 1)
        else:
            self.stack = nn.Sequential(
                nn.Conv1d(input_channel, self.n_chans, 3, 1, 1),
                nn.GroupNorm(4, self.n_chans),
                nn.LeakyReLU(),
                nn.Conv1d(self.n_chans, self.n_chans, 3, 1, 1))

        # out
        if self.out_put_norm:
            self.norm = nn.LayerNorm(self.n_chans)
            self.n_out = out_dims
            self.dense_out = weight_norm(
                nn.Linear(self.n_chans, self.n_out))
        else:
            self.out_proj = nn.Linear(self.n_chans, out_dims)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                spk_emb=None, spk_emb_dict=None, use_vae=False):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''
        assert use_vae is False
        x = self.stack(units.transpose(1, 2)).transpose(1, 2)
        x = x + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.use_speaker_encoder:
            if spk_mix_dict is not None:
                assert spk_emb_dict is not None
                for k, v in spk_mix_dict.items():
                    spk_id_torch = spk_emb_dict[str(k)]
                    spk_id_torch = np.tile(spk_id_torch, (len(units), 1))
                    spk_id_torch = torch.from_numpy(spk_id_torch).float().to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                x = x + self.spk_embed(spk_emb)
        else:
            if self.n_spk is not None and self.n_spk > 1:
                if spk_mix_dict is not None:
                    for k, v in spk_mix_dict.items():
                        spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                        x = x + v * self.spk_embed(spk_id_torch - 1)
                else:
                    x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)

        x = self.decoder(x)
        if self.out_put_norm:
            x = self.norm(x)
            x = self.dense_out(x)
        else:
            x = self.out_proj(x)
        if not infer:
            x = F.mse_loss(x, gt_spec)
        return x


class Unit2MelNaiveV2ForDiff(nn.Module):
    def __init__(
            self,
            input_channel,
            out_dims=128,
            net_fn=None,
    ):
        super().__init__()
        self.l2reg_loss = 0  # 废弃
        # catch None
        assert net_fn is not None
        self.n_layers = net_fn.n_layers if (net_fn.n_layers is not None) else 6
        self.n_chans = net_fn.n_chans if (net_fn.n_chans is not None) else 256
        self.out_put_norm = net_fn.out_put_norm if (net_fn.out_put_norm is not None) else False
        self.simple_stack = net_fn.simple_stack if (net_fn.simple_stack is not None) else False

        if net_fn.type == 'LYNXNet' or net_fn.type == 'NaiveNet':
            self.expansion_factor = net_fn.expansion_factor if (net_fn.expansion_factor is not None) else 2
            self.kernel_size = net_fn.kernel_size if (net_fn.kernel_size is not None) else 31
            self.conv_model_type = net_fn.conv_model_type if (net_fn.conv_model_type is not None) else 'mode1'
            self.num_heads = net_fn.num_heads if (net_fn.num_heads is not None) else 8
            self.use_norm = net_fn.use_norm if (net_fn.use_norm is not None) else False
            self.conv_only = net_fn.conv_only if (net_fn.conv_only is not None) else True
            self.conv_dropout = net_fn.conv_dropout if (net_fn.conv_dropout is not None) else 0.0
            self.atten_dropout = net_fn.atten_dropout if (net_fn.atten_dropout is not None) else 0.1

            self.decoder = ConformerNaiveEncoder(
                num_layers=self.n_layers,
                num_heads=self.num_heads,
                dim_model=self.n_chans,
                expansion_factor=self.expansion_factor,
                kernel_size=self.kernel_size,
                use_norm=self.use_norm,
                conv_only=self.conv_only,
                conv_dropout=self.conv_dropout,
                atten_dropout=self.atten_dropout,
                conv_model_type=self.conv_model_type,
            )
        else:
            raise ValueError(f'net_fn.type={net_fn.type} is not supported')

        # conv in stack
        if self.simple_stack:
            self.stack = nn.Conv1d(input_channel, self.n_chans, 1)
        else:
            self.stack = nn.Sequential(
                nn.Conv1d(input_channel, self.n_chans, 3, 1, 1),
                nn.GroupNorm(4, self.n_chans),
                nn.LeakyReLU(),
                nn.Conv1d(self.n_chans, self.n_chans, 3, 1, 1))

        # out
        if self.out_put_norm:
            self.norm = nn.LayerNorm(self.n_chans)
            self.n_out = out_dims
            self.dense_out = weight_norm(
                nn.Linear(self.n_chans, self.n_out))
        else:
            self.out_proj = nn.Linear(self.n_chans, out_dims)

    def forward(self, in_x):

        # input proj
        x = self.stack(in_x.transpose(1, 2)).transpose(1, 2)

        # net
        x = self.decoder(x)

        # out prpk
        if self.out_put_norm:
            x = self.norm(x)
            x = self.dense_out(x)
        else:
            x = self.out_proj(x)
        return x

