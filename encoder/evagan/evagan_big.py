# A inference only version of the FireflyGAN model
# ref from https://github.com/fishaudio/fish-speech/blob/add-flow-vqgan/fish_speech/models/vqgan/modules/firefly.py

from functools import partial
from math import prod
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint


class CombToothSourceModule(nn.Module):
    def __init__(
            self,
            *,
            sampling_rate: int = 44100,
            wave_amp: float = 0.1,
            noise_std: float = 0.003,
            voiced_threshold: float = 0,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.wave_amp = wave_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0 (torch.Tensor): [B, 1, T]

        Returns:
            combtooth (torch.Tensor): [B, 1, T]
        """

        x = torch.cumsum(f0 / self.sampling_rate, axis=2)
        x = x - torch.round(x)
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3)) * self.wave_amp

        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.noise_std + (1 - uv) * self.wave_amp / 3
        noise = noise_amp * torch.randn_like(combtooth)
        combtooth = combtooth * uv + noise

        return combtooth


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_parametrizations(self):
        for conv in self.convs1:
            remove_parametrizations(conv, tensor_name="weight")
        for conv in self.convs2:
            remove_parametrizations(conv, tensor_name="weight")


class ParralelBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_sizes=(3, 7, 11),  # tuple[int]
            dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),  # tuple[tuple[int]]
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilation_sizes)

        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock1(channels, k, d))

    def forward(self, x):
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block.remove_parametrizations()


class HiFiGANGenerator(nn.Module):
    def __init__(
            self,
            *,
            hop_length: int = 512,
            upsample_rates=(8, 8, 2, 2, 2),  # tuple[int]
            upsample_kernel_sizes=(16, 16, 8, 2, 2),  # tuple[int]
            resblock_kernel_sizes=(3, 7, 11),  # tuple[int]
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),  # tuple[tuple[int]]
            num_mels: int = 128,
            upsample_initial_channel: int = 512,
            use_template: bool = True,
            pre_conv_kernel_size: int = 7,
            post_conv_kernel_size: int = 7,
            post_activation: Callable = partial(nn.SiLU, inplace=True),
    ):
        super().__init__()

        assert (
                prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = weight_norm(
            nn.Conv1d(
                num_mels,
                upsample_initial_channel,
                pre_conv_kernel_size,
                1,
                padding=get_padding(pre_conv_kernel_size),
            )
        )

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.use_template = use_template
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            if not use_template:
                continue

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(
                ParralelBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.activation_post = post_activation()
        self.conv_post = weight_norm(
            nn.Conv1d(
                ch,
                1,
                post_conv_kernel_size,
                1,
                padding=get_padding(post_conv_kernel_size),
            )
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, template=None):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            if self.use_template:
                x = x + self.noise_convs[i](template)

            if self.training and self.checkpointing:
                x = checkpoint(
                    self.resblocks[i],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.resblocks[i](x)

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_parametrizations(self):
        for up in self.ups:
            remove_parametrizations(up, tensor_name="weight")
        for block in self.resblocks:
            block.remove_parametrizations()
        remove_parametrizations(self.conv_pre, tensor_name="weight")
        remove_parametrizations(self.conv_post, tensor_name="weight")


# DropPath copied from timm library
def drop_path(
        x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """  # noqa: E501

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""  # noqa: E501

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


# ConvNeXt Block copied from https://github.com/fishaudio/fish-diffusion/blob/main/fish_diffusion/modules/convnext.py
class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
            self,
            dim: int,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
            mlp_ratio: float = 4.0,
            kernel_size: int = 7,
            dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)

        if apply_residual:
            x = input + x

        return x


class ConvNeXtEncoder(nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            depths=[3, 3, 9, 3],  # list[int]
            dims=[96, 192, 384, 768],  # list[int]
            drop_path_rate: float = 0.0,
            layer_scale_init_value: float = 1e-6,
            kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.channel_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                dims[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode="zeros",
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.channel_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.channel_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_size=kernel_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.channel_layers)):
            x = self.channel_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x)


class EVAGANBig(nn.Module):
    def __init__(self, ckpt_path: str = None, pretrained: bool = False, loaded_state_dict=None):
        super().__init__()
        self.hop_length = 512
        self.backbone = ConvNeXtEncoder(
            input_channels=160,
            depths=[3, 3, 9, 3],
            dims=[128, 256, 384, 512],
            drop_path_rate=0.2,
            kernel_size=7,
        )

        self.head = HiFiGANGenerator(
            hop_length=512,
            upsample_rates=[4, 4, 2, 2, 2, 2, 2],
            upsample_kernel_sizes=[8, 8, 4, 4, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11, 13],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
            num_mels=512,
            upsample_initial_channel=1536,
            use_template=True,
            pre_conv_kernel_size=13,
            post_conv_kernel_size=13,
        )
        self.source = CombToothSourceModule(
            sampling_rate=44100,
            wave_amp=0.1,
            noise_std=0.003,
            voiced_threshold=0,
        )

        # diffusion svc特色功能,打包所有权重,此时ckpt_path被忽略,直接加载传入的权重
        if loaded_state_dict is not None:
            state_dict = loaded_state_dict
        else:
            if ckpt_path is not None:
                # self.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                state_dict = torch.load(ckpt_path, map_location="cpu")
            else:
                raise ValueError("ckpt_path must be provided")
            # 讲道理预训练模型未来还是会试用集中统一的管理，就不用自带的了
            '''
            elif pretrained:
                state_dict = torch.hub.load_state_dict_from_url(
                    "https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base.ckpt",
                    map_location="cpu",
                )
            '''

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator." in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        self.load_state_dict(state_dict, strict=True)
        self.head.remove_parametrizations()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = torch.zeros((1, 1, x.shape[-1] * self.hop_length), device=x.device)
        if f0.ndim == 2:
            f0 = f0[:, None]
        f0 = F.interpolate(
            f0, size=x.shape[-1] * self.hop_length, mode="linear"
        )
        template = self.source(f0)
        x = self.backbone(x)
        x = self.head(x, template)
        if x.ndim == 2:
            x = x[:, None, :]
        return x


if __name__ == "__main__":
    path_model = ""
    path_read = ""
    path_save = ""

    model = EVAGANBig(path_model)
    model.eval()
    from torch import Tensor
    import torchaudio
    from torchaudio.transforms import MelScale


    class LinearSpectrogram(nn.Module):
        def __init__(
                self,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                center=False,
                mode="pow2_sqrt",
        ):
            super().__init__()

            self.n_fft = n_fft
            self.win_length = win_length
            self.hop_length = hop_length
            self.center = center
            self.mode = mode

            self.register_buffer("window", torch.hann_window(win_length))

        def forward(self, y: Tensor) -> Tensor:
            if y.ndim == 3:
                y = y.squeeze(1)

            y = torch.nn.functional.pad(
                y.unsqueeze(1),
                (
                    (self.win_length - self.hop_length) // 2,
                    (self.win_length - self.hop_length + 1) // 2,
                ),
                mode="reflect",
            ).squeeze(1)

            spec = torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )

            spec = torch.view_as_real(spec)

            if self.mode == "pow2_sqrt":
                spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

            return spec


    class LogMelSpectrogram(nn.Module):
        def __init__(
                self,
                sample_rate=44100,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                n_mels=128,
                center=False,
                f_min=0.0,
                f_max=None,
        ):
            super().__init__()

            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.win_length = win_length
            self.hop_length = hop_length
            self.center = center
            self.n_mels = n_mels
            self.f_min = f_min
            self.f_max = f_max or sample_rate // 2

            self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
            self.mel_scale = MelScale(
                self.n_mels,
                self.sample_rate,
                self.f_min,
                self.f_max,
                self.n_fft // 2 + 1,
                "slaney",
                "slaney",
            )

        def compress(self, x: Tensor) -> Tensor:
            return torch.log(torch.clamp(x, min=1e-5))

        def decompress(self, x: Tensor) -> Tensor:
            return torch.exp(x)

        def forward(self, x: Tensor) -> Tensor:
            x = self.spectrogram(x)
            x = self.mel_scale(x)
            x = self.compress(x)

            return x
    audio, sr = torchaudio.load(path_read)
    assert sr == 44100
    mel = LogMelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=160,
        center=False,
        f_min=0.0,
        f_max=22050,
    )(audio)
    with torch.no_grad():
        y = model(mel)
    print(y.shape)  # torch.Size([C,1,T])
    torchaudio.save(path_save, y.squeeze().cpu(), 44100)


