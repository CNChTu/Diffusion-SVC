import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .naive.naive import Unit2MelNaive
from .unet1d.unet_1d_condition import UNet1DConditionModel
from .mrte_model import MRTE
class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
                    args.model.use_pitch_aug,
                    vocoder_dimension,
                    args.model.n_layers,
                    args.model.block_out_channels,
                    args.model.n_heads,
                    args.model.n_hidden,
                    mrte_layer=args.model.mrte_layer,
                    mrte_hident_size=args.model.mrte_hident_size
                    )

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
        raise ("Unknow model")
    return model


class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=2,
            block_out_channels=(256,384,512,512),
            n_heads=8,
            n_hidden=256,
            mrte_layer=5,
            mrte_hident_size = 512
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.mrte = MRTE(
            out_dims,
            n_hidden,
            mrte_layer,
            mrte_hident_size,
            n_hidden,
            5,
            4,
            2
        )
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        # diffusion
        self.decoder = GaussianDiffusion(UNet1DConditionModel(in_channels=out_dims + n_hidden,
        out_channels=out_dims,
        block_out_channels=block_out_channels,
        norm_num_groups=8,
        # cross_attention_dim = block_out_channels,
        cross_attention_dim = out_dims,
        attention_head_dim = n_heads,
        # only_cross_attention = True,
        layers_per_block = n_layers,
        addition_embed_type='text',
        resnet_time_scale_shift='scale_shift'), out_dims=out_dims)

    def forward(self, units, f0, volume, reference_audio_mel,aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True
                ):

        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.mrte(self.unit_embed(units), reference_audio_mel)
        x += self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)

        # x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        # x = self.mrte(x, reference_audio_mel)
        x = self.decoder(x, gt_spec=gt_spec,reference_mel = reference_audio_mel, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step,
                         use_tqdm=use_tqdm)

        return x
