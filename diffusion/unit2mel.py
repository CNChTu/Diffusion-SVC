import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random
from .reflow.reflow_1step import RectifiedFlow1Step
from .reflow.reflow import RectifiedFlow
from .diffusion import GaussianDiffusion
from .convnext import ConvNext
from .vocoder import Vocoder
from .naive.naive import Unit2MelNaive
from .naive_v2.naive_v2 import Unit2MelNaiveV2, Unit2MelNaiveV2ForDiff
from .naive_v2.naive_v2_diff import NaiveV2Diff


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_network_from_dot(netdot, out_dims, cond_dims):
    # check type
    if not isinstance(netdot, DotDict):
        assert isinstance(netdot, dict)
        netdot = DotDict(netdot)

    # get network
    if netdot.type == 'WaveNet':
        # catch None
        wn_layers = netdot.wn_layers if (netdot.wn_layers is not None) else 20
        wn_chans = netdot.wn_chans if (netdot.wn_chans is not None) else 384
        wn_dilation = netdot.wn_dilation if (netdot.wn_dilation is not None) else 1
        wn_kernel = netdot.wn_kernel if (netdot.wn_kernel is not None) else 3
        wn_tf_use = netdot.wn_tf_use if (netdot.wn_tf_use is not None) else False
        wn_tf_rf = netdot.wn_tf_rf if (netdot.wn_tf_rf is not None) else False
        wn_tf_n_layers = netdot.wn_tf_n_layers if (netdot.wn_tf_n_layers is not None) else 2
        wn_tf_n_head = netdot.wn_tf_n_head if (netdot.wn_tf_n_head is not None) else 4
        no_t_emb = netdot.no_t_emb if (netdot.no_t_emb is not None) else False

        # init wavenet denoiser
        from .wavenet import WaveNet
        denoiser = WaveNet(out_dims, wn_layers, wn_chans, cond_dims, wn_dilation, wn_kernel,
                           wn_tf_use, wn_tf_rf, wn_tf_n_layers, wn_tf_n_head, no_t_emb)
    elif netdot.type == 'WaveNetAdain':
        # catch None
        wn_layers = netdot.wn_layers if (netdot.wn_layers is not None) else 20
        wn_chans = netdot.wn_chans if (netdot.wn_chans is not None) else 384
        wn_dilation = netdot.wn_dilation if (netdot.wn_dilation is not None) else 1
        wn_kernel = netdot.wn_kernel if (netdot.wn_kernel is not None) else 3
        wn_tf_use = netdot.wn_tf_use if (netdot.wn_tf_use is not None) else False
        wn_tf_rf = netdot.wn_tf_rf if (netdot.wn_tf_rf is not None) else False
        wn_tf_n_layers = netdot.wn_tf_n_layers if (netdot.wn_tf_n_layers is not None) else 2
        wn_tf_n_head = netdot.wn_tf_n_head if (netdot.wn_tf_n_head is not None) else 4
        no_t_emb = netdot.no_t_emb if (netdot.no_t_emb is not None) else False

        # init wavenet denoiser
        from .wavenet_adain import WaveNet
        denoiser = WaveNet(out_dims, wn_layers, wn_chans, cond_dims, wn_dilation, wn_kernel,
                           wn_tf_use, wn_tf_rf, wn_tf_n_layers, wn_tf_n_head, no_t_emb)
    elif netdot.type == 'ConvNext':
        # catch None
        cn_layers = netdot.cn_layers if (netdot.cn_layers is not None) else 20
        cn_chans = netdot.cn_chans if (netdot.cn_chans is not None) else 384
        cn_dilation_cycle = netdot.cn_dilation_cycle if (netdot.cn_dilation_cycle is not None) else 4
        mlp_factor = netdot.mlp_factor if (netdot.mlp_factor is not None) else 4
        gradient_checkpointing = netdot.gradient_checkpointing if (
                netdot.gradient_checkpointing is not None) else False
        # init convnext denoiser
        denoiser = ConvNext(
            mel_channels=out_dims,
            dim=cn_chans,
            mlp_factor=mlp_factor,
            condition_dim=cond_dims,
            num_layers=cn_layers,
            dilation_cycle=cn_dilation_cycle,
            gradient_checkpointing=gradient_checkpointing
        )

    elif (netdot.type == 'NaiveV2Diff') or (netdot.type == 'LYNXNetDiff'):
        # catch None
        cn_layers = netdot.cn_layers if (netdot.cn_layers is not None) else 20
        cn_chans = netdot.cn_chans if (netdot.cn_chans is not None) else 384
        use_mlp = netdot.use_mlp if (netdot.use_mlp is not None) else True
        mlp_factor = netdot.mlp_factor if (netdot.mlp_factor is not None) else 4
        expansion_factor = netdot.expansion_factor if (netdot.expansion_factor is not None) else 2
        kernel_size = netdot.kernel_size if (netdot.kernel_size is not None) else 31
        conv_only = netdot.conv_only if (netdot.conv_only is not None) else True
        wavenet_like = netdot.wavenet_like if (netdot.wavenet_like is not None) else False
        use_norm = netdot.use_norm if (netdot.use_norm is not None) else False
        conv_model_type = netdot.conv_model_type if (netdot.conv_model_type is not None) else 'mode1'
        conv_dropout = netdot.conv_dropout if (netdot.conv_dropout is not None) else 0.0
        atten_dropout = netdot.atten_dropout if (netdot.atten_dropout is not None) else 0.1
        no_t_emb = netdot.no_t_emb if (netdot.no_t_emb is not None) else False
        conv_model_activation = netdot.conv_model_activation if (netdot.conv_model_activation is not None) else 'SiLU'
        GLU_type = netdot.GLU_type if (netdot.GLU_type is not None) else 'GLU'
        fix_free_norm = netdot.fix_free_norm if (netdot.fix_free_norm is not None) else False
        channel_norm = netdot.channel_norm if (netdot.channel_norm is not None) else False
        # init convnext denoiser
        denoiser = NaiveV2Diff(
            mel_channels=out_dims,
            dim=cn_chans,
            use_mlp=use_mlp,
            mlp_factor=mlp_factor,
            condition_dim=cond_dims,
            num_layers=cn_layers,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            conv_only=conv_only,
            wavenet_like=wavenet_like,
            use_norm=use_norm,
            conv_model_type=conv_model_type,
            conv_dropout=conv_dropout,
            atten_dropout=atten_dropout,
            no_t_emb=no_t_emb,
            conv_model_activation=conv_model_activation,
            GLU_type=GLU_type,
            fix_free_norm=fix_free_norm,
            channel_norm=channel_norm
        )

    else:
        raise TypeError(" [X] Unknow netdot.type")

    return denoiser


def get_z(stack_tensor, mean_only=False, clip_min=None, clip_max=None):
    # stack_tensor: [B x N x D x 2]
    # sample z, or mean only
    m = stack_tensor.transpose(-1, 0)[:1].transpose(-1, 0).squeeze(-1)
    logs = stack_tensor.transpose(-1, 0)[1:].transpose(-1, 0).squeeze(-1)
    if mean_only:
        z = m  # mean only
    else:
        z = m + torch.randn_like(m) * torch.exp(logs)  # sample z
    if (clip_min is not None) or (clip_max is not None):
        assert clip_min is not None
        assert clip_max is not None
        z = z.clamp(min=clip_min, max=clip_max)
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
    ckpt = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args


def load_model_vocoder_from_combo(combo_model_path, device='cpu', loaded_vocoder=None):
    read_dict = torch.load(combo_model_path, map_location=torch.device(device), weights_only=True)
    # 检查是否有键名“_version_”
    if '_version_' in read_dict.keys():
        raise ValueError(" [X] 这是新版本的模型, 请在新仓库中使用")
    # 检查是否有键名“comb_diff_model”, 如果有为带级联训练的模型
    if '_is_comb_diff_model' in read_dict.keys():
        if read_dict['_is_comb_diff_model']:
            is_combo_diff_model = True
    else:
        is_combo_diff_model = False
    # 如果打包了声码器, 则从权重中加载声码器
    if 'vocoder' in read_dict.keys():
        read_vocoder_dict = read_dict['vocoder']  # 从权重中读取声码器, 里面key有 'config' 和 'model'
        vocoder_type = read_dict['vocoder_type']  # 字符串, 用于指定声码器类型
        if loaded_vocoder is not None:
            print(" [WARN] 外部加载了声码器, 但是权重中也包含了声码器, 请确认你清楚自己在做什么")
            print(" [WARN] 权重中的声码器将会被忽略！")
        else:
            loaded_vocoder = Vocoder(vocoder_type, read_vocoder_dict, device=device)

    # args
    diff_args = DotDict(read_dict["diff_config_dict"])
    if not is_combo_diff_model:
        naive_args = DotDict(read_dict["naive_config_dict"])
    else:
        naive_args = None
    # vocoder
    if loaded_vocoder is None:
        vocoder = Vocoder(diff_args.vocoder.type, diff_args.vocoder.ckpt, device=device)
    else:
        vocoder = loaded_vocoder

    # diff_model
    print(' [Loading] ' + combo_model_path)
    diff_model = load_svc_model(args=diff_args, vocoder_dimension=vocoder.dimension)
    diff_model.to(device)
    diff_model.load_state_dict(read_dict["diff_model"]['model'])
    diff_model.eval()

    # naive_model
    if not is_combo_diff_model:
        naive_model = load_svc_model(args=naive_args, vocoder_dimension=vocoder.dimension)
        naive_model.to(device)
        naive_model.load_state_dict(read_dict["naive_model"]['model'])
        naive_model.eval()
    else:
        naive_model = None
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
            denoise_fn=args.model.denoise_fn,
            mask_cond_ratio=args.model.mask_cond_ratio,
            naive_fn=args.model.naive_fn,
            naive_fn_grad_not_by_diffusion=args.model.naive_fn_grad_not_by_diffusion,
            naive_out_mel_cond_diff=args.model.naive_out_mel_cond_diff)

    elif args.model.type == 'ReFlow':
        model = Unit2MelV2ReFlow(
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
            velocity_fn=args.model.velocity_fn,
            mask_cond_ratio=args.model.mask_cond_ratio,
            naive_fn=args.model.naive_fn,
            naive_fn_grad_not_by_reflow=args.model.naive_fn_grad_not_by_reflow,
            naive_out_mel_cond_reflow=args.model.naive_out_mel_cond_reflow,
            loss_type=args.model.loss_type,
            consistency=args.model.consistency,
            consistency_only=args.model.consistency_only,
            consistency_delta_t=args.model.consistency_delta_t,
            consistency_lambda_f=args.model.consistency_lambda_f,
            consistency_lambda_v=args.model.consistency_lambda_v,
        )

    elif args.model.type == 'ReFlow1Step':
        model = Unit2MelV2ReFlow1Step(
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
            velocity_fn=args.model.velocity_fn,
            mask_cond_ratio=args.model.mask_cond_ratio,
            naive_fn=args.model.naive_fn,
            naive_fn_grad_not_by_reflow=args.model.naive_fn_grad_not_by_reflow,
            naive_out_mel_cond_reflow=args.model.naive_out_mel_cond_reflow,
            loss_type=args.model.loss_type,
            consistency=args.model.consistency,
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

    elif args.model.type == 'NaiveV2':
        if args.model.net_fn is None:
            net_fn = {
                'type': 'LYNXNet',  # LYNXNet是ConformerNaiveEncoder(简称NaiveNet)的别名
                'n_layers': args.model.n_layers,
                'n_chans': args.model.n_chans,
                'out_put_norm': True,
                'conv_model_type': 'mode1',
            }
            net_fn = DotDict(net_fn)
        else:
            net_fn = args.model.net_fn

        model = Unit2MelNaiveV2(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder_dimension,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
            net_fn=net_fn)

    else:
        raise TypeError(" [X] Unknow model")

    # check compile
    if args.model.torch_compile_args is not None:
        if str(args.model.torch_compile_args.use_copile).lower() == 'true':
            print(" [INFO] Compile model with torch.compile().")

            # check torch version >= 2.3.0
            if torch.__version__ >= '2.3.0':
                raise ValueError(" [X] torch version must >= 2.3.0 to use torch.compile() in DiffusionSVC.")

            # dynamic
            if str(args.model.torch_compile_args.dynamic).lower() == 'none':
                _dynamic = None
            elif str(args.model.torch_compile_args.dynamic).lower() == 'true':
                _dynamic = True
            elif str(args.model.torch_compile_args.dynamic).lower() == 'false':
                _dynamic = False
            else:
                raise ValueError(" [X] Unknow model.torch_compile_args.dynamic config")

            # options
            if str(args.model.torch_compile_args.use_options).lower() == 'true':
                if args.model.torch_compile_args.options is None:
                    raise ValueError(" [X] model.torch_compile_args.options is None,"
                                     " but model.torch_compile_args.use_options is True.")
                _options = dict(args.model.torch_compile_args.options)
            else:
                _options = None

            # fullgraph, backend, mode
            if args.model.torch_compile_args.fullgraph is None:
                _fullgraph = False
            else:
                _fullgraph = args.model.torch_compile_args.fullgraph
            if args.model.torch_compile_args.backend is None:
                _backend = 'inductor'
            else:
                _backend = args.model.torch_compile_args.backend
            if args.model.torch_compile_args.mode is None:
                _mode = 'default'
            else:
                _mode = args.model.torch_compile_args.mode
                if _backend != 'inductor':
                    _mode = None

            # compile
            if _options is not None:
                model = torch.compile(
                    model, dynamic=_dynamic, fullgraph=_fullgraph, backend=_backend, mode=_mode,options=_options
                )
            else:
                model = torch.compile(
                    model, dynamic=_dynamic, fullgraph=_fullgraph, backend=_backend, mode=_mode
                )

        else:
            print(" [INFO] Not compile this model.")
    else:
        print(" [INFO] Not compile this model.")

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
            denoise_fn=None,
            mask_cond_ratio=None,
            naive_fn=None,
            naive_fn_grad_not_by_diffusion=False,
            naive_out_mel_cond_diff=True
    ):
        super().__init__()
        # check and init mask_cond_ratio
        if mask_cond_ratio is not None:
            mask_cond_ratio = float(mask_cond_ratio) if (str(mask_cond_ratio) != 'NOTUSE') else -99
            if mask_cond_ratio > 0:
                self.mask_cond_ratio = mask_cond_ratio
                # 未实现错误
                raise NotImplementedError(" [X] mask_cond_ratio is not implemented.")
            else:
                self.mask_cond_ratio = None
        else:
            self.mask_cond_ratio = None

        if denoise_fn is None:
            # catch None
            raise ValueError(" [X] denoise_fn is None, please check config file")

        # check naive_fn
        if naive_fn is None:
            self.combo_trained_model = False
            self.naive_stack = None
            self.naive_proj = None
            self.naive_fn_grad_not_by_diffusion = False
            self.naive_out_mel_cond_diff = False
        else:
            # check naive_fn_grad_not_by_diffusion with naive_fn
            self.combo_trained_model = True
            if naive_fn_grad_not_by_diffusion is not None:
                self.naive_fn_grad_not_by_diffusion = bool(naive_fn_grad_not_by_diffusion)
            else:
                self.naive_fn_grad_not_by_diffusion = False
            # check naive_out_mel_cond_diff with naive_fn
            if naive_out_mel_cond_diff is not None:
                self.naive_out_mel_cond_diff = bool(naive_out_mel_cond_diff)
            else:
                self.naive_out_mel_cond_diff = True
            # init naive_fn
            if not isinstance(naive_fn, DotDict):
                assert isinstance(naive_fn, dict)
                naive_fn = DotDict(naive_fn)
            self.naive_stack = Unit2MelNaiveV2ForDiff(
                input_channel=n_hidden,
                out_dims=out_dims,
                net_fn=naive_fn
            )
            self.naive_proj = nn.Linear(out_dims, n_hidden)

        # init denoiser
        denoiser = get_network_from_dot(denoise_fn, out_dims, n_hidden)

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
        self.decoder = self.spawn_decoder(denoiser, out_dims)

    def spawn_decoder(self,denoiser, out_dims):
        decoder = GaussianDiffusion(
            denoiser,
            out_dims=out_dims,
            max_beta=self.max_beta,
            spec_min=self.spec_min,
            spec_max=self.spec_max)
        return decoder

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
        x = self.emb_spk(x, spk_id, len(units), units.device, spk_mix_dict, spk_emb_dict, spk_emb)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)

        # sample z or mean only
        if use_vae and (gt_spec is not None):
            gt_spec = get_z(gt_spec, mean_only=self.mean_only, clip_min=self.spec_min, clip_max=self.spec_max)
            if (self.z_rate is not None) and (self.z_rate != 0):
                gt_spec = gt_spec * self.z_rate  # scale z

        # mask cpmd start
        if not infer:
            self.mask_cond_train_start()

        # combo trained model
        if self.combo_trained_model:
            x, gt_spec, naive_loss = self.naive_fn_forward_for_combo_trained_model(x, gt_spec, infer, use_vae)
        else:
            naive_loss = 0

        # diffusion
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step,
                         use_tqdm=use_tqdm)

        # mask cond end
        self.mask_cond_train_end()

        if infer:
            if (self.z_rate is not None) and (self.z_rate != 0):
                x = x / self.z_rate  # scale z

        if not infer:
            if self.combo_trained_model:
                return {'diff_loss': x, 'naive_loss': naive_loss}
            else:
                return {'diff_loss': (x + naive_loss)}

        return x

    def mask_cond_train_start(self):
        if self.mask_cond_ratio is not None:
            if self.denoise_fn_type == 'NaiveV2Diff':
                self.decoder.denoise_fn.mask_cond_ratio = self.mask_cond_ratio

    def mask_cond_train_end(self):
        if self.mask_cond_ratio is not None:
            if self.denoise_fn_type == 'NaiveV2Diff':
                self.decoder.denoise_fn.mask_cond_ratio = None

    def naive_fn_forward_for_combo_trained_model(self, x, gt_spec, infer, use_vae):
        # forward naive_fn, get _x from input x
        _x = self.naive_stack(x, use_vae=use_vae)
        if infer:
            gt_spec = _x
            naive_loss = 0
        else:
            naive_loss = F.mse_loss(_x, gt_spec)
        _x = self.naive_proj(_x)  # project _x to n_hidden matching x
        # if naive_fn_grad_not_by_diffusion is True, then detach _x, make it not grad by diffusion
        if self.naive_fn_grad_not_by_diffusion:
            _x = _x.detach()
        # if naive_out_mel_cond_diff is True, then use _x as cond for diffusion, else use x
        if self.naive_out_mel_cond_diff:
            x = _x
        return x, gt_spec, naive_loss

    def emb_spk(self, x, spk_id, units_len, units_device, spk_mix_dict, spk_emb_dict , spk_emb):
        if self.use_speaker_encoder:
            if spk_mix_dict is not None:
                assert spk_emb_dict is not None
                for k, v in spk_mix_dict.items():
                    spk_id_torch = spk_emb_dict[str(k)]
                    spk_id_torch = np.tile(spk_id_torch, (units_len, 1))
                    spk_id_torch = torch.from_numpy(spk_id_torch).float().to(units_device)
                    x = x + v * self.spk_embed(spk_id_torch)
            else:
                x = x + self.spk_embed(spk_emb)
        else:
            if self.n_spk is not None and self.n_spk > 1:
                if spk_mix_dict is not None:
                    for k, v in spk_mix_dict.items():
                        spk_id_torch = torch.LongTensor(np.array([[k]])).to(units_device)
                        x = x + v * self.spk_embed(spk_id_torch - 1)
                else:
                    x = x + self.spk_embed(spk_id - 1)
        return x


class Unit2MelV2ReFlow(Unit2MelV2):
    def spawn_decoder(self, velocity_fn, out_dims):
        decoder = RectifiedFlow(
            velocity_fn,
            out_dims=out_dims,
            spec_min=self.spec_min,
            spec_max=self.spec_max,
            loss_type=self.loss_type,
            consistency=self.consistency,
            consistency_only=self.consistency_only,
            consistency_delta_t=self.consistency_delta_t,
            consistency_lambda_f=self.consistency_lambda_f,
            consistency_lambda_v=self.consistency_lambda_v,
        )
        return decoder

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
            max_beta=0.02,  # 暂时废弃，但是极有可能未来会有用吧，所以先不删除, 可以为None
            spec_min=-12,
            spec_max=2,
            velocity_fn=None,
            mask_cond_ratio=None,
            naive_fn=None,
            naive_fn_grad_not_by_reflow=False,
            naive_out_mel_cond_reflow=True,
            loss_type='l2',
            consistency=False,
            consistency_only=True,
            consistency_delta_t=0.1,
            consistency_lambda_f=1.0,
            consistency_lambda_v=1.0,
    ):
        self.loss_type = loss_type if (loss_type is not None) else 'l2'
        self.consistency = consistency if (consistency is not None) else False
        self.consistency_only = consistency_only if (consistency_only is not None) else True
        self.consistency_delta_t = consistency_delta_t if (consistency_delta_t is not None) else 0.1
        self.consistency_lambda_f = consistency_lambda_f if (consistency_lambda_f is not None) else 1.0
        self.consistency_lambda_v = consistency_lambda_v if (consistency_lambda_v is not None) else 1.0
        super().__init__(
            input_channel,
            n_spk,
            use_pitch_aug=use_pitch_aug,
            out_dims=out_dims,
            n_hidden=n_hidden,
            use_speaker_encoder=use_speaker_encoder,
            speaker_encoder_out_channels=speaker_encoder_out_channels,
            z_rate=z_rate,
            mean_only=mean_only,
            max_beta=max_beta,
            spec_min=spec_min,
            spec_max=spec_max,
            denoise_fn=velocity_fn,
            mask_cond_ratio=mask_cond_ratio,
            naive_fn=naive_fn,
            naive_fn_grad_not_by_diffusion=naive_fn_grad_not_by_reflow,
            naive_out_mel_cond_diff=naive_out_mel_cond_reflow
        )

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None,
                gt_spec=None, infer=True, infer_step=10, method='euler', t_start=0.0, use_tqdm=True,
                spk_emb=None, spk_emb_dict=None, use_vae=False):
        '''
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        '''

        # embed
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        x = self.emb_spk(x, spk_id, len(units), units.device, spk_mix_dict, spk_emb_dict, spk_emb)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)

        # sample z or mean only
        if use_vae and (gt_spec is not None):
            gt_spec = get_z(gt_spec, mean_only=self.mean_only, clip_min=self.spec_min, clip_max=self.spec_max)
            if (self.z_rate is not None) and (self.z_rate != 0):
                gt_spec = gt_spec * self.z_rate  # scale z

        # mask cpmd start
        if not infer:
            self.mask_cond_train_start()

        # combo trained model
        if self.combo_trained_model:
            x, gt_spec, naive_loss = self.naive_fn_forward_for_combo_trained_model(x, gt_spec, infer, use_vae)
        else:
            naive_loss = 0

        # reflow
        if infer:
            x = self.decoder(x, gt_spec=gt_spec, infer=True, infer_step=infer_step, method=method, t_start=t_start,
                             use_tqdm=use_tqdm)
            _step_loss_dict = None
        else:
            _step_loss_dict = self.step_train(x, gt_spec, t_start)

        # mask cond end
        self.mask_cond_train_end()

        if infer:
            if (self.z_rate is not None) and (self.z_rate != 0):
                x = x / self.z_rate  # scale z

        if not infer:
            return self.make_loss_dict(_step_loss_dict, naive_loss)
        else:
            return x

    def make_loss_dict(self, _step_loss_dict, naive_loss):
        x = _step_loss_dict['x']
        co_loss = _step_loss_dict['co_loss']
        if self.combo_trained_model:
            if self.consistency:
                if self.consistency_only:
                    return {'reflow_consistency_loss': x, 'naive_loss': naive_loss}
                else:
                    return {'consistency_loss': co_loss,
                            'reflow_loss': x,
                            'naive_loss': naive_loss}
            else:
                return {'reflow_loss': x, 'naive_loss': naive_loss}
        else:
            if self.consistency:
                if self.consistency_only:
                    return {'reflow_consistency_loss': (x + naive_loss)}
                else:
                    return {'consistency_loss': co_loss,
                            'reflow_loss': (x + naive_loss)}
            else:
                return {'reflow_loss': (x + naive_loss)}

    def step_train(self, x, gt_spec, t_start):
        co_loss = None
        if self.consistency:
            if self.consistency_only:
                x = self.decoder(x, gt_spec=gt_spec, t_start=t_start, infer=False)
            else:
                x, co_loss = self.decoder(x, gt_spec=gt_spec, t_start=t_start, infer=False)
        else:
            x = self.decoder(x, gt_spec=gt_spec, t_start=t_start, infer=False)
        return {'x':x, 'co_loss':co_loss}


class Unit2MelV2ReFlow1Step(Unit2MelV2ReFlow):
    def spawn_decoder(self, velocity_fn, out_dims):
        decoder = RectifiedFlow1Step(
            velocity_fn,
            out_dims=out_dims,
            spec_min=self.spec_min,
            spec_max=self.spec_max,
            loss_type=self.loss_type,
            consistency=self.x0_xt_consistency
        )
        return decoder
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
            max_beta=0.02,  # 暂时废弃，但是极有可能未来会有用吧，所以先不删除, 可以为None
            spec_min=-12,
            spec_max=2,
            velocity_fn=None,
            mask_cond_ratio=None,
            naive_fn=None,
            naive_fn_grad_not_by_reflow=False,
            naive_out_mel_cond_reflow=True,
            loss_type='l2',
            consistency=True,
    ):
        self.x0_xt_consistency = consistency if (consistency is not None) else True
        super().__init__(
            input_channel,
            n_spk,
            use_pitch_aug=use_pitch_aug,
            out_dims=out_dims,
            n_hidden=n_hidden,
            use_speaker_encoder=use_speaker_encoder,
            speaker_encoder_out_channels=speaker_encoder_out_channels,
            z_rate=z_rate,
            mean_only=mean_only,
            max_beta=max_beta,
            spec_min=spec_min,
            spec_max=spec_max,
            velocity_fn=velocity_fn,
            mask_cond_ratio=mask_cond_ratio,
            naive_fn=naive_fn,
            naive_fn_grad_not_by_reflow=naive_fn_grad_not_by_reflow,
            naive_out_mel_cond_reflow=naive_out_mel_cond_reflow,
            loss_type=loss_type,
            consistency=False,
        )

    def step_train(self, x, gt_spec, t_start):
        xt_loss, x0_loss, consistency_loss = self.decoder(x, gt_spec=gt_spec, t_start=t_start, infer=False)
        return {'x':xt_loss, 'x0_loss':x0_loss, 'consistency_loss':consistency_loss}

    def make_loss_dict(self, _step_loss_dict, naive_loss):
        x = _step_loss_dict['x']
        x0_loss = _step_loss_dict['x0_loss']
        consistency_loss = _step_loss_dict['consistency_loss']
        if self.combo_trained_model:
            return {'reflow_loss': x,
                    'reflow_loss_x0':x0_loss,
                    'consistency_loss': consistency_loss,
                    'naive_loss': naive_loss}
        else:
            return {'reflow_loss': (x + naive_loss),
                    'reflow_loss_x0': x0_loss,
                    'consistency_loss': consistency_loss}


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
        from .wavenet import WaveNet
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
