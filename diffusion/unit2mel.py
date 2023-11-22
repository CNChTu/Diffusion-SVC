import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .diffusion import GaussianDiffusion
from .wavenet import WaveNet
from .convnext import ConvNext
from .vocoder import Vocoder
from .naive.naive import Unit2MelNaive


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_z(stack_tensor, mean_only=False):
    # stack_tensor: [B x N x D x 2]
    # sample z, or mean only
    m = stack_tensor.transpose(-1, 0)[:1].transpose(-1, 0).squeeze(-1)
    logs = stack_tensor.transpose(-1, 0)[1:].transpose(-1, 0).squeeze(-1)
    if mean_only:
        z = m  # mean only
    else:
        z = m + torch.randn_like(m) * torch.exp(logs)  # sample z
    return z  # [B x N x D]


def load_model_vocoder(
        model_path,
        device='cpu',
        loaded_vocoder=None):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    # load vocoder
    if loaded_vocoder is None:
        vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    else:
        vocoder = loaded_vocoder

    # load model
    model = load_svc_model(args=args, vocoder_dimension=vocoder.dimension)

    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args


def load_model_vocoder_from_combo(combo_model_path, device='cpu'):
    read_dict = torch.load(combo_model_path, map_location=torch.device(device))
    # args
    diff_args = DotDict(read_dict["diff_config_dict"])
    naive_args = DotDict(read_dict["naive_config_dict"])
    # vocoder
    vocoder = Vocoder(diff_args.vocoder.type, diff_args.vocoder.ckpt, device=device)

    # diff_model
    print(' [Loading] ' + combo_model_path)
    diff_model = load_svc_model(args=diff_args, vocoder_dimension=vocoder.dimension)
    diff_model.to(device)
    diff_model.load_state_dict(read_dict["diff_model"]['model'])
    diff_model.eval()

    # naive_model
    naive_model = load_svc_model(args=naive_args, vocoder_dimension=vocoder.dimension)
    naive_model.to(device)
    naive_model.load_state_dict(read_dict["naive_model"]['model'])
    naive_model.eval()
    return diff_model, diff_args, naive_model, naive_args, vocoder


def load_svc_model(args, vocoder_dimension):
    if args.model.type == 'Diffusion':
        model = Unit2Mel(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder_dimension,
            args.model.n_layers,
            args.model.n_chans,
            args.model.n_hidden,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels)

    elif args.model.type == 'DiffusionV2':
        model = Unit2MelV2(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder_dimension,
            args.model.n_hidden,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
            z_rate=args.model.z_rate,
            mean_only=args.model.mean_only,
            max_beta=args.model.max_beta,
            spec_min=args.model.spec_min,
            spec_max=args.model.spec_max,
            denoise_fn=args.model.denoise_fn)

    elif args.model.type == 'Naive':
        model = Unit2MelNaive(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder_dimension,
            args.model.n_layers,
            args.model.n_chans,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels)

    elif args.model.type == 'NaiveFS':
        model = Unit2MelNaive(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder_dimension,
            args.model.n_layers,
            args.model.n_chans,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
            use_full_siren=True,
            l2reg_loss=args.model.l2_reg_loss)

    else:
        raise TypeError(" [X] Unknow model")
    return model


class Unit2MelV2(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_hidden=256,
            use_speaker_encoder=False,
            speaker_encoder_out_channels=256,
            z_rate=None,
            mean_only=False,
            max_beta=0.02,
            spec_min=-12,
            spec_max=2,
            denoise_fn=None):
        super().__init__()

        if denoise_fn is None:
            # catch None
            denoise_fn = {'type': 'WaveNet',
                          'wn_layers': 20,
                          'wn_chans': 384,
                          'wn_dilation': 1,
                          'wn_kernel': 3,
                          'wn_tf_use': False,
                          'wn_tf_rf': False,
                          'wn_tf_n_layers': 2,
                          'wn_tf_n_head': 4}
            denoise_fn = DotDict(denoise_fn)

        if denoise_fn.type == 'WaveNet':
            # catch None
            self.wn_layers = denoise_fn.wn_layers if (denoise_fn.wn_layers is not None) else 20
            self.wn_chans = denoise_fn.wn_chans if (denoise_fn.wn_chans is not None) else 384
            self.wn_dilation = denoise_fn.wn_dilation if (denoise_fn.wn_dilation is not None) else 1
            self.wn_kernel = denoise_fn.wn_kernel if (denoise_fn.wn_kernel is not None) else 3
            self.wn_tf_use = denoise_fn.wn_tf_use if (denoise_fn.wn_tf_use is not None) else False
            self.wn_tf_rf = denoise_fn.wn_tf_rf if (denoise_fn.wn_tf_rf is not None) else False
            self.wn_tf_n_layers = denoise_fn.wn_tf_n_layers if (denoise_fn.wn_tf_n_layers is not None) else 2
            self.wn_tf_n_head = denoise_fn.wn_tf_n_head if (denoise_fn.wn_tf_n_head is not None) else 4

            # init wavenet denoiser
            denoiser = WaveNet(out_dims, self.wn_layers, self.wn_chans, n_hidden, self.wn_dilation, self.wn_kernel,
                               self.wn_tf_use, self.wn_tf_rf, self.wn_tf_n_layers, self.wn_tf_n_head)

        elif denoise_fn.type == 'ConvNext':
            # catch None
            self.cn_layers = denoise_fn.cn_layers if (denoise_fn.cn_layers is not None) else 20
            self.cn_chans = denoise_fn.cn_chans if (denoise_fn.cn_chans is not None) else 384
            self.cn_dilation_cycle = denoise_fn.cn_dilation_cycle if (denoise_fn.cn_dilation_cycle is not None) else 4
            self.mlp_factor = denoise_fn.mlp_factor if (denoise_fn.mlp_factor is not None) else 4
            self.gradient_checkpointing = denoise_fn.gradient_checkpointing if (
                    denoise_fn.gradient_checkpointing is not None) else False
            # init convnext denoiser
            denoiser = ConvNext(
                mel_channels=out_dims,
                dim=self.cn_chans,
                mlp_factor=self.mlp_factor,
                condition_dim=n_hidden,
                num_layers=self.cn_layers,
                dilation_cycle=self.cn_dilation_cycle,
                gradient_checkpointing=self.gradient_checkpointing
            )

        else:
            raise TypeError(" [X] Unknow denoise_fn")

        # catch None
        self.z_rate = z_rate
        self.mean_only = mean_only if (mean_only is not None) else False
        self.max_beta = max_beta if (max_beta is not None) else 0.02
        self.spec_min = spec_min if (spec_min is not None) else -12
        self.spec_max = spec_max if (spec_max is not None) else 2

        # init embed
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        self.use_speaker_encoder = use_speaker_encoder
        if use_speaker_encoder:
            self.spk_embed = nn.Linear(speaker_encoder_out_channels, n_hidden, bias=False)
        else:
            if n_spk is not None and n_spk > 1:
                self.spk_embed = nn.Embedding(n_spk, n_hidden)

        # init diffusion
        self.decoder = GaussianDiffusion(
            denoiser,
            out_dims=out_dims,
            max_beta=self.max_beta,
            spec_min=self.spec_min,
            spec_max=self.spec_max)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                spk_emb=None, spk_emb_dict=None, use_vae=False):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        # embed
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
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

        # sample z or mean only
        if use_vae and (gt_spec is not None):
            gt_spec = get_z(gt_spec, mean_only=self.mean_only)
            if (self.z_rate is not None) and (self.z_rate != 0):
                gt_spec = gt_spec * self.z_rate  # scale z

        # diffusion
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step,
                         use_tqdm=use_tqdm)

        if (self.z_rate is not None) and (self.z_rate != 0):
            x = x / self.z_rate  # scale z

        return x


class Unit2Mel(nn.Module):
    # old version
    def __init__(
            self,
            input_channel,
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=20,
            n_chans=384,
            n_hidden=256,
            use_speaker_encoder=False,
            speaker_encoder_out_channels=256):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        self.use_speaker_encoder = use_speaker_encoder
        if use_speaker_encoder:
            self.spk_embed = nn.Linear(speaker_encoder_out_channels, n_hidden, bias=False)
        else:
            if n_spk is not None and n_spk > 1:
                self.spk_embed = nn.Embedding(n_spk, n_hidden)

        # diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden, 1, 3, False),
                                         out_dims=out_dims, max_beta=0.02, spec_min=-12, spec_max=2)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                spk_emb=None, spk_emb_dict=None, use_vae=False):

        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''

        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
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

        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step,
                         use_tqdm=use_tqdm)

        return x
