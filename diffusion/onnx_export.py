import json
import argparse
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random
from diffusion_onnx import GaussianDiffusion
from wavenet import WaveNet
from convnext import ConvNext
from naive.naive_onnx import Unit2MelNaive
from naive_v2.naive_v2_onnx import Unit2MelNaiveV2
from naive_v2.naive_v2_diff import NaiveV2Diff

parser = argparse.ArgumentParser(description='Onnx Export')
parser.add_argument('--project', type=str, help='Project Name')
args_main = parser.parse_args()

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
def load_model_vocoder(
        model_path,
        device='cpu'):
    pat = model_path
    config_file = model_path + '/config.yaml'
    model_path = model_path + '/model.pt'

    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    if args.model.type == 'Diffusion':
        model = Unit2Mel(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    128,
                    args.model.n_layers,
                    args.model.n_chans,
                    args.model.n_hidden,
                    args.data.encoder_hop_size,
                    args.data.sampling_rate,
                    block_size=args.data.block_size)
    elif args.model.type == 'DiffusionV2':
        model = Unit2MelV2(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            128,
            args.model.n_hidden,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
            z_rate=args.model.z_rate,
            mean_only=args.model.mean_only,
            max_beta=args.model.max_beta,
            spec_min=args.model.spec_min,
            spec_max=args.model.spec_max,
            denoise_fn=args.model.denoise_fn,
            mask_cond_ratio=args.model.mask_cond_ratio,
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size, 
            hop_size=args.data.encoder_hop_size)
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    naive = None
    naive_config_file = pat + '/config_naive.yaml'
    naive_path = pat + '/naive.pt'
    
    if os.path.exists(naive_config_file) and os.path.exists(naive_path):
        with open(naive_config_file, "r") as naive_config:
            naive_args = yaml.safe_load(naive_config)
        naive_args = DotDict(naive_args)

        if naive_args.model.type == 'Naive':
            naive = Unit2MelNaive(
                        naive_args.data.encoder_out_channels, 
                        naive_args.model.n_spk,
                        naive_args.model.use_pitch_aug,
                        128,
                        naive_args.model.n_layers,
                        naive_args.model.n_chans,
                        use_speaker_encoder=naive_args.model.use_speaker_encoder,
                        speaker_encoder_out_channels=naive_args.data.speaker_encoder_out_channels,
                        spec_max=model.decoder.spec_max,
                        spec_min=model.decoder.spec_min)
        elif naive_args.model.type == 'NaiveFS':
            naive = Unit2MelNaive(
                        naive_args.data.encoder_out_channels,
                        naive_args.model.n_spk,
                        naive_args.model.use_pitch_aug,
                        128,
                        naive_args.model.n_layers,
                        naive_args.model.n_chans,
                        use_speaker_encoder=naive_args.model.use_speaker_encoder,
                        speaker_encoder_out_channels=naive_args.data.speaker_encoder_out_channels,
                        spec_max=model.decoder.spec_max,
                        spec_min=model.decoder.spec_min,
                        use_full_siren=True,
                        l2reg_loss=naive_args.model.l2_reg_loss)
        elif naive_args.model.type == 'NaiveV2':
            if naive_args.model.net_fn is None:
                net_fn = {
                    'type': 'LYNXNet',  # LYNXNet是ConformerNaiveEncoder(简称NaiveNet)的别名
                    'n_layers': naive_args.model.n_layers,
                    'n_chans': naive_args.model.n_chans,
                    'out_put_norm': True,
                    'conv_model_type': 'mode1',
                }
                net_fn = DotDict(net_fn)
            else:
                net_fn = naive_args.model.net_fn
            naive = Unit2MelNaiveV2(
                        naive_args.data.encoder_out_channels,
                        naive_args.model.n_spk,
                        naive_args.model.use_pitch_aug,
                        128,
                        use_speaker_encoder=naive_args.model.use_speaker_encoder,
                        speaker_encoder_out_channels=naive_args.data.speaker_encoder_out_channels,
                        net_fn=net_fn,
                        spec_max=model.decoder.spec_max,
                        spec_min=model.decoder.spec_min)
        else:
            raise TypeError(" [X] Unknow model")

        ckpt_naive = torch.load(naive_path, map_location=torch.device(device))
        naive.to(device)
        naive.load_state_dict(ckpt_naive['model'])
        naive.eval()
    return model, args, naive


class Unit2Mel(nn.Module):
    def __init__(self, input_channel, n_spk, use_pitch_aug=False, out_dims=128, n_layers=20, n_chans=384, n_hidden=256, hop_size=320,
            sampling_rate=44100, use_speaker_encoder=False, speaker_encoder_out_channels=256, block_size=512):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.hop_size = hop_size
        self.hubert_channel = input_channel
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
                    n_hidden=n_hidden, out_dims=out_dims, max_beta=0.02, spec_min=-12, spec_max=2)
        self.hidden_size = n_hidden
        
    def forward(self, units, mel2ph, f0, volume, g = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        decoder_inp = F.pad(units, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units.shape[-1]])
        units = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        x = self.unit_embed(units) + self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()) + self.volume_embed(volume.unsqueeze(-1))

        if self.n_spk is not None and self.n_spk > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
            x = x.transpose(1, 2) + g
            return x
        else:
            return x.transpose(1, 2)
        
    def init_spkembed(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        spk_dice_offset = 0
        self.speaker_map = torch.zeros((self.n_spk,1,1,self.hidden_size))

        if self.use_speaker_encoder:
            if spk_mix_dict is not None:
                assert spk_emb_dict is not None
                for k, v in spk_mix_dict.items():
                    spk_id_torch = spk_emb_dict[str(k)]
                    spk_id_torch = np.tile(spk_id_torch, (len(units), 1))
                    spk_id_torch = torch.from_numpy(spk_id_torch).float().to(units.device)
                    self.speaker_map[spk_dice_offset] = self.spk_embed(spk_id_torch)
                    spk_dice_offset = spk_dice_offset + 1
        else:
            if self.n_spk is not None and self.n_spk > 1:
                if spk_mix_dict is not None:
                    for k, v in spk_mix_dict.items():
                        spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                        self.speaker_map[spk_dice_offset] = self.spk_embed(spk_id_torch)
                        spk_dice_offset = spk_dice_offset + 1
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.speaker_map = self.speaker_map.detach()

    def OnnxExport(self, project_name=None, init_noise=None, export_encoder=True, export_denoise=True, export_pred=True, export_after=True):
        n_frames = 100
        hubert = torch.randn((1, n_frames, self.hubert_channel))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        spk_mix = spk_mix.repeat(n_frames, 1)
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        if export_encoder:
            torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"checkpoints/{project_name}/{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1],
                    "spk_mix": [0],
                },
                opset_version=16
            )
        self.decoder.OnnxExport(project_name, init_noise=init_noise, export_denoise=export_denoise, export_pred=export_pred, export_after=export_after)
        
        vec_lay = "layer-12" if self.hubert_channel == 768 else "layer-9"
        spklist = []
        for key in range(self.n_spk):
            spklist.append(f"Speaker_{key}")

        MoeVSConf = {
            "Folder" : f"{project_name}",
            "Name" : f"{project_name}",
            "Type" : "DiffSvc",
            "Rate" : self.sampling_rate,
            "Hop" : self.block_size,
            "Hubert": f"vec-{self.hubert_channel}-{vec_lay}",
            "HiddenSize": self.hubert_channel,
            "Characters": spklist,
            "Diffusion": True,
            "CharaMix": True,
            "Volume": True,
            "V2" : True,
            "Hifigan" : "nsf_hifigan",
            "MelBins" : self.decoder.mel_bins,
            "MaxStep" : self.decoder.k_step
        }

        MoeVSConfJson = json.dumps(MoeVSConf)
        with open(f"checkpoints/{project_name}.json", 'w') as MoeVsConfFile:
            json.dump(MoeVSConf, MoeVsConfFile, indent = 4)

    def ExportOnnx(self, project_name=None):
        n_frames = 100
        hubert = torch.randn((1, n_frames, self.hubert_channel))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        orgouttt = self.orgforward(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        outtt = self.forward(hubert, mel2ph, f0, volume, spk_mix)

        torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1]
                },
                opset_version=16
            )

        condition = torch.randn(1,self.decoder.n_hidden,n_frames)
        noise = torch.randn((1, 1, self.decoder.mel_bins, condition.shape[2]), dtype=torch.float32)
        pndm_speedup = torch.LongTensor([100])
        K_steps = torch.LongTensor([1000])
        self.decoder = torch.jit.script(self.decoder)
        self.decoder(condition, noise, pndm_speedup, K_steps)

        torch.onnx.export(
                self.decoder,
                (condition, noise, pndm_speedup, K_steps),
                f"{project_name}_diffusion.onnx",
                input_names=["condition", "noise", "pndm_speedup", "K_steps"],
                output_names=["mel"],
                dynamic_axes={
                    "condition": [2],
                    "noise": [3],
                },
                opset_version=16
            )


class Unit2MelV2(nn.Module):
    def __init__(self, input_channel, n_spk, use_pitch_aug=False, out_dims=128, n_hidden=256, use_speaker_encoder=False, speaker_encoder_out_channels=256, 
    z_rate=None, mean_only=False, max_beta=0.02, spec_min=-12, spec_max=2, denoise_fn=None, mask_cond_ratio=None, sampling_rate=44100,
    block_size=512, hop_size=320):
        super().__init__()
        if mask_cond_ratio is not None:
            mask_cond_ratio = float(mask_cond_ratio) if (str(mask_cond_ratio) != 'NOTUSE') else None
            if mask_cond_ratio > 0:
                self.mask_cond_ratio = mask_cond_ratio
            else:
                self.mask_cond_ratio = None
        else:
            self.mask_cond_ratio = None

        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.hop_size = hop_size
        self.hubert_channel = input_channel
        self.hidden_size = n_hidden

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
                               self.wn_tf_use, self.wn_tf_rf, self.wn_tf_n_layers, self.wn_tf_n_head, self.dwconv)

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

        elif denoise_fn.type == 'NaiveV2Diff':
            # catch None
            self.cn_layers = denoise_fn.cn_layers if (denoise_fn.cn_layers is not None) else 20
            self.cn_chans = denoise_fn.cn_chans if (denoise_fn.cn_chans is not None) else 384
            self.use_mlp = denoise_fn.use_mlp if (denoise_fn.use_mlp is not None) else True
            self.mlp_factor = denoise_fn.mlp_factor if (denoise_fn.mlp_factor is not None) else 4
            self.expansion_factor = denoise_fn.expansion_factor if (denoise_fn.expansion_factor is not None) else 2
            self.kernel_size = denoise_fn.kernel_size if (denoise_fn.kernel_size is not None) else 31
            self.conv_only = denoise_fn.conv_only if (denoise_fn.conv_only is not None) else True
            self.wavenet_like = denoise_fn.wavenet_like if (denoise_fn.wavenet_like is not None) else False
            self.use_norm = denoise_fn.use_norm if (denoise_fn.use_norm is not None) else False
            self.conv_model_type = denoise_fn.conv_model_type if (denoise_fn.conv_model_type is not None) else 'mode1'
            self.conv_dropout = denoise_fn.conv_dropout if (denoise_fn.conv_dropout is not None) else 0.0
            self.atten_dropout = denoise_fn.atten_dropout if (denoise_fn.atten_dropout is not None) else 0.1
            # init convnext denoiser
            denoiser = NaiveV2Diff(
                mel_channels=out_dims,
                dim=self.cn_chans,
                use_mlp=self.use_mlp,
                mlp_factor=self.mlp_factor,
                condition_dim=n_hidden,
                num_layers=self.cn_layers,
                expansion_factor=self.expansion_factor,
                kernel_size=self.kernel_size,
                conv_only=self.conv_only,
                wavenet_like=self.wavenet_like,
                use_norm=self.use_norm,
                conv_model_type=self.conv_model_type,
                conv_dropout=self.conv_dropout,
                atten_dropout=self.atten_dropout,
            )

        else:
            raise TypeError(" [X] Unknow denoise_fn")
        self.denoise_fn_type = denoise_fn.type

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

    def forward(self, units, mel2ph, f0, volume, g = None):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        # embed
        decoder_inp = F.pad(units, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, units.shape[-1]])
        units = torch.gather(decoder_inp, 1, mel2ph_)  # [B, T, H]

        x = self.unit_embed(units) + self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()) + self.volume_embed(volume.unsqueeze(-1))

        if self.n_spk is not None and self.n_spk > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
            x = x.transpose(1, 2) + g
            return x
        else:
            return x.transpose(1, 2)

    def init_spkembed(self, units, f0, volume, spk_id = None, spk_mix_dict = None, aug_shift = None,
                gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        spk_dice_offset = 0
        self.speaker_map = torch.zeros((self.n_spk,1,1,self.hidden_size))

        if self.use_speaker_encoder:
            if spk_mix_dict is not None:
                assert spk_emb_dict is not None
                for k, v in spk_mix_dict.items():
                    spk_id_torch = spk_emb_dict[str(k)]
                    spk_id_torch = np.tile(spk_id_torch, (len(units), 1))
                    spk_id_torch = torch.from_numpy(spk_id_torch).float().to(units.device)
                    self.speaker_map[spk_dice_offset] = self.spk_embed(spk_id_torch)
                    spk_dice_offset = spk_dice_offset + 1
        else:
            if self.n_spk is not None and self.n_spk > 1:
                if spk_mix_dict is not None:
                    for k, v in spk_mix_dict.items():
                        spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                        self.speaker_map[spk_dice_offset] = self.spk_embed(spk_id_torch)
                        spk_dice_offset = spk_dice_offset + 1
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.speaker_map = self.speaker_map.detach()

    def OnnxExport(self, project_name=None, init_noise=None, export_encoder=True, export_denoise=True, export_pred=True, export_after=True):
        n_frames = 100
        hubert = torch.randn((1, n_frames, self.hubert_channel))
        mel2ph = torch.arange(end=n_frames).unsqueeze(0).long()
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spk_mix = []
        spks = {}
        if self.n_spk is not None and self.n_spk > 1:
            for i in range(self.n_spk):
                spk_mix.append(1.0/float(self.n_spk))
                spks.update({i:1.0/float(self.n_spk)})
        spk_mix = torch.tensor(spk_mix)
        spk_mix = spk_mix.repeat(n_frames, 1)
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)
        if export_encoder:
            torch.onnx.export(
                self,
                (hubert, mel2ph, f0, volume, spk_mix),
                f"checkpoints/{project_name}/{project_name}_encoder.onnx",
                input_names=["hubert", "mel2ph", "f0", "volume", "spk_mix"],
                output_names=["mel_pred"],
                dynamic_axes={
                    "hubert": [1],
                    "f0": [1],
                    "volume": [1],
                    "mel2ph": [1],
                    "spk_mix": [0],
                },
                opset_version=16
            )
        self.decoder.OnnxExport(project_name, init_noise=init_noise, export_denoise=export_denoise, export_pred=export_pred, export_after=export_after)
        
        vec_lay = "layer-12" if self.hubert_channel == 768 else "layer-9"
        spklist = []
        for key in range(self.n_spk):
            spklist.append(f"Speaker_{key}")

        MoeVSConf = {
            "Folder" : f"{project_name}",
            "Name" : f"{project_name}",
            "Type" : "DiffSvc",
            "Rate" : self.sampling_rate,
            "Hop" : self.block_size,
            "Hubert": f"vec-{self.hubert_channel}-{vec_lay}",
            "HiddenSize": self.hubert_channel,
            "Characters": spklist,
            "Diffusion": True,
            "CharaMix": True,
            "Volume": True,
            "V2" : True,
            "Hifigan" : "nsf_hifigan",
            "MelBins" : self.decoder.mel_bins,
            "MaxStep" : self.decoder.k_step
        }

        MoeVSConfJson = json.dumps(MoeVSConf)
        with open(f"checkpoints/{project_name}.json", 'w') as MoeVsConfFile:
            json.dump(MoeVSConf, MoeVsConfFile, indent = 4)



if __name__ == "__main__":
    
    project_name = args_main.project
    
    if project_name is None:
        project_name = "DiffusionV2Test"

    model_path = f'checkpoints/{project_name}'

    model, _, naive = load_model_vocoder(model_path)

    if naive is not None:
        naive.ExportOnnx(project_name)
    # 分开Diffusion导出（需要使用MoeSS/MoeVoiceStudio或者自己编写Pndm/Dpm采样）
    model.OnnxExport(project_name, export_encoder=True, export_denoise=True, export_pred=True, export_after=True)

