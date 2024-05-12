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
from diffusion.w2v_conformer.w2v_conformer_wraper import Wav2Vec2ConformerConfig, Wav2Vec2ConformerWrapper
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
        spec_norm = False if args.vocoder.type == "hifi-vaegan" else True

        model = Unit2Mel(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder_dimension,
                    args.model.n_layers,
                    args.model.block_out_channels,
                    args.model.n_heads,
                    args.model.n_hidden,
                    use_speaker_encoder=args.model.use_speaker_encoder,
                    speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
                    is_tts = args.model.is_tts,
                    spec_norm=spec_norm,
                    acoustic_scale=args.data.acoustic_scale
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
            n_spk,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=2,
            block_out_channels=(256,384,512,512),
            n_heads=8,
            n_hidden=256,
            use_speaker_encoder=False,
            speaker_encoder_out_channels=256,
            is_tts: bool = False,
            spec_norm=True,
            acoustic_scale=1.0,
            use_extract_cond = False,
            mrte_layer = 4,
            mrte_hident_size = 256
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.is_tts = is_tts
        if not is_tts:
            self.f0_embed = nn.Linear(1, n_hidden)
            self.volume_embed = nn.Linear(1, n_hidden)
            if use_pitch_aug:
                self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
            else:
                self.aug_shift_embed = None
        else:
            self.aug_shift_embed = None
            self.f0_embed = None
            self.volume_embed = None

        self.n_spk = n_spk
        self.use_speaker_encoder = use_speaker_encoder
        if use_speaker_encoder:
            self.spk_embed = nn.Linear(speaker_encoder_out_channels, n_hidden, bias=False)
        else:
            if n_spk is not None and n_spk > 1:
                self.spk_embed = nn.Embedding(n_spk, n_hidden)
        # diffusion
        self.decoder = GaussianDiffusion(UNet1DConditionModel(in_channels=out_dims + n_hidden,
        out_channels=out_dims,
        block_out_channels=block_out_channels,
        norm_num_groups=8,
        cross_attention_dim = block_out_channels,
        attention_head_dim = n_heads,
        layers_per_block = n_layers,
        resnet_time_scale_shift='scale_shift'), 
        out_dims=out_dims,
        spec_norm=spec_norm,
        acoustic_scale=acoustic_scale)
        if use_extract_cond:
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
        else:
            self.mrte = nn.Identity()
        # config = Wav2Vec2ConformerConfig(
        #     hidden_size=n_hidden,
        #     num_hidden_layers=n_layers,
        #     num_attention_heads=n_heads,
        #     intermediate_size=n_hidden*4,
        #     hidden_act="gelu",
        #     hidden_dropout=0.1,
        #     activation_dropout=0.1,
        #     attention_dropout=0.1,
        #     feat_proj_dropout=0.0,
        #     feat_quantizer_dropout=0.0,
        #     final_dropout=0.1,
        #     layerdrop=0.1,
        #     initializer_range=0.02,
        #     layer_norm_eps=1e-5,
        #     feat_extract_norm="group",
        #     feat_extract_activation="gelu",
        #     conv_dim=block_out_channels,
        #     conv_stride=(5, 2, 2, 2),
        #     conv_kernel=(10, 3, 3, 3),
        #     conv_bias=False
        # )

        # self.decoder = GaussianDiffusion(Wav2Vec2ConformerWrapper(in_channels=out_dims + n_hidden,
        # out_channels=out_dims,
        # config=config
        # ), 
        # out_dims=out_dims,
        # spec_norm=spec_norm,
        # acoustic_scale=acoustic_scale)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                spk_emb=None, spk_emb_dict=None):

        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        if f0 is None or self.is_tts:
            f0 = 0
        else:
            f0 = self.f0_embed((1 + f0 / 700).log())
        if volume is None or self.is_tts:
            volume = 0
        else:
            volume = self.volume_embed(volume)

        x = self.unit_embed(units) + f0 + volume
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
