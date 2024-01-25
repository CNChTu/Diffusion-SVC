import numpy as np
import torch
import json
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
import os
from vector_quantize_pytorch import VectorQuantize

LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if "Depthwise_Separable" in classname:
        m.depth_conv.weight.data.normal_(mean, std)
        m.point_conv.weight.data.normal_(mean, std)
    elif classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Encoder(nn.Module):
    def __init__(self, h,
                 ):
        super().__init__()

        self.h = h
        # h["inter_channels"]
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.out_channels = h["inter_channels"]
        self.num_downsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(
            Conv1d(1, h["upsample_initial_channel"] // (2 ** len(h["upsample_rates"])), 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(reversed(h["upsample_rates"]), reversed(h["upsample_kernel_sizes"]))):
            self.ups.append(weight_norm(
                Conv1d(h["upsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - i)),
                       h["upsample_initial_channel"] // (2 ** (len(h["upsample_rates"]) - i - 1)),
                       k, u, padding=(k - u + 1) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups), 0, -1):
            ch = h["upsample_initial_channel"] // (2 ** (i - 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 2 * h["inter_channels"], 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

    def remove_weight_norm(self):
        for a_lay in self.ups:
            remove_weight_norm(a_lay)
        for a_lay in self.resblocks:
            a_lay.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def forward(self, x):
        x = x[:, None, :]
        x = self.conv_pre(x)
        for i in range(self.num_downsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        m, logs = torch.split(x, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs

    # def forward(self, x):
    #     x = self.conv_pre(x)
    #     for i in range(self.num_upsamples):
    #         x = F.leaky_relu(x, LRELU_SLOPE)
    #         x = self.ups[i](x)
    #         xs = None
    #         for j in range(self.num_kernels):
    #             if xs is None:
    #                 xs = self.resblocks[i * self.num_kernels + j](x)
    #             else:
    #                 xs += self.resblocks[i * self.num_kernels + j](x)
    #         x = xs / self.num_kernels
    #     x = F.leaky_relu(x)
    #     x = self.conv_post(x)
    #     x = torch.tanh(x)

    #     return x


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h

        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.conv_pre = weight_norm(Conv1d(h["inter_channels"], h["upsample_initial_channel"], 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(
                ConvTranspose1d(h["upsample_initial_channel"] // (2 ** i),
                                h["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u + 1) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = np.prod(h["upsample_rates"])

        if h["use_vq"]:
            self.quantizer = VectorQuantize(
                dim=h["inter_channels"],
                codebook_size=h["codebook_size"],
                decay=0.8,
                commitment_weight=1.)
        else:
            self.quantizer = None

    def forward(self, x):

        # if self.quantizer is not None:
        #    x, _, _ = self.quantizer(x.transpose(1, 2))
        #    x = x.transpose(1, 2)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for a_lay in self.ups:
            remove_weight_norm(a_lay)
        for a_lay in self.resblocks:
            a_lay.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def get_hparams_from_file(config_path, infer_mode=False):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config) if not infer_mode else InferHParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, index):
        return self.__dict__.get(index)


class InferHParams(HParams):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = InferHParams(**v)
            self[k] = v

    def __getattr__(self, index):
        return self.get(index)


class InferModel:
    def __init__(self,
                 config_path,  # 如果是字典，则其实是传入的权重+配置，单独处理。其中包含'config'和'model'两个key
                 model_path=None,
                 device='cuda',
                 fp16=False,
                 load_all=False,
                 _load_from_state_dict=False
                 ):
        # config_path 是否为字典
        if _load_from_state_dict:
            assert type(config_path) == dict
            config_dict = config_path['config']
            hps = InferHParams(**config_dict)
            self.model_path = None
            self.config_path = None
            self.load_from_state_dict(config_path['model'])
        else:
            assert type(config_path) != dict
            hps = get_hparams_from_file(config_path, True)
            self.model_path = model_path
            self.config_path = config_path

        self.inter_channels = hps.model.inter_channels
        self.hidden_channels = hps.model.hidden_channels
        self.kernel_size = hps.model.kernel_size
        self.p_dropout = hps.model.p_dropout
        self.resblock = hps.model.resblock
        self.resblock_kernel_sizes = hps.model.resblock_kernel_sizes
        self.resblock_dilation_sizes = hps.model.resblock_dilation_sizes
        self.upsample_rates = hps.model.upsample_rates
        self.upsample_initial_channel = hps.model.upsample_initial_channel
        self.upsample_kernel_sizes = hps.model.upsample_kernel_sizes
        self.ssl_dim = hps.model.ssl_dim
        self.hop_size = hps.data.hop_length
        self.windows_size = hps.data.hop_length
        self.device = device
        self.fp16 = fp16
        self.sr = hps.data.sampling_rate
        self.use_vq = hps.model.use_vq if (hps.model.use_vq is not None) else False
        self.codebook_size = int(hps.model.codebook_size) if (hps.model.codebook_size is not None) else 4096

        self.hps_ = {
            "sampling_rate": hps.data.sampling_rate,
            "inter_channels": hps.model.inter_channels,
            "resblock": hps.model.resblock,
            "resblock_kernel_sizes": hps.model.resblock_kernel_sizes,
            "resblock_dilation_sizes": hps.model.resblock_dilation_sizes,
            "upsample_rates": hps.model.upsample_rates,
            "upsample_initial_channel": hps.model.upsample_initial_channel,
            "upsample_kernel_sizes": hps.model.upsample_kernel_sizes,
            "use_vq": self.use_vq,
            "codebook_size": self.codebook_size
        }

        if load_all:
            self.dec = Generator(h=self.hps_).cpu()
            self.dec.load_state_dict(self.load('dec'), strict=False)
            self.dec.eval()
            self.dec = self.dec.to(self.device)
            self.enc_q = Encoder(h=self.hps_).cpu()
            self.enc_q.load_state_dict(self.load('enc_q'), strict=False)
            self.enc_q.eval()
            self.enc_q = self.enc_q.to(self.device)
        else:
            self.dec = None
            self.enc_q = None

    """
    @torch.no_grad()
    def forward(self, wav):
        z, m, logs = self.enc_q(wav)
        wav = self.dec(z)

        return z, wav, (m, logs)
    """

    @torch.no_grad()
    def encode(self, wav):
        if self.enc_q is None:
            self.enc_q = Encoder(h=self.hps_).cpu()
            self.enc_q.load_state_dict(self.load('enc_q'), strict=False)
            self.enc_q.eval()
            self.enc_q.remove_weight_norm()
            self.enc_q = self.enc_q.to(self.device)
        z, m, logs = self.enc_q(wav)
        return z, m, logs

    @torch.no_grad()
    def decode(self, z):
        if self.dec is None:
            self.dec = Generator(h=self.hps_).cpu()
            self.dec.load_state_dict(self.load('dec'), strict=False)
            self.dec.eval()
            self.dec.remove_weight_norm()
            self.dec = self.dec.to(self.device)
        wav = self.dec(z)
        return wav

    @torch.no_grad()
    def load(self, model_type):
        assert os.path.isfile(self.model_path)
        model_dict = torch.load(self.model_path, map_location='cpu')["model"]
        load_dict = {}
        for k, v in model_dict.items():
            if k[:len(model_type)] == model_type:
                load_dict[k[len(model_type) + 1:]] = v
        return load_dict

    @torch.no_grad()
    def load_from_state_dict(self, state_dict):
        state_dict = state_dict["model"]
        # dec
        load_dict_dec = {}
        for k, v in state_dict.items():
            if k[:len('dec')] == 'dec':
                load_dict_dec[k[len('dec') + 1:]] = v
        self.dec = Generator(h=self.hps_).cpu()
        self.dec.load_state_dict(load_dict_dec, strict=False)
        self.dec.eval()
        self.dec.remove_weight_norm()
        self.dec = self.dec.to(self.device)

        # enc
        load_dict_enc = {}
        for k, v in state_dict.items():
            if k[:len('enc_q')] == 'enc_q':
                load_dict_enc[k[len('enc_q') + 1:]] = v
        self.enc_q = Encoder(h=self.hps_).cpu()
        self.enc_q.load_state_dict(load_dict_enc, strict=False)
        self.enc_q.eval()
        self.enc_q.remove_weight_norm()
        self.enc_q = self.enc_q.to(self.device)


if __name__ == '__main__':
    UNIT_TEST = False
    if UNIT_TEST:
        import soundfile
        import librosa

        model = InferModel(
            r"C:\Users\29210\Desktop\ylzzvaegan/config.json",
            r"C:\Users\29210\Desktop\ylzzvaegan\新建文件夹/G_200.pth"
        )
        in_wav, in_sr = librosa.load(r"E:\AUFSe04BPyProgram\AUFSd04BPyProgram\AudioGAN\AudioGAN\raw\测试专用.wav",
                                     sr=int(model.sr))
        print(in_wav.shape)
        in_wav = torch.from_numpy(in_wav).float().unsqueeze(0).to(model.device)
        _z, m, logs = model.encode(in_wav)
        z = m + torch.randn_like(m) * torch.exp(logs)
        print(z.max(), z.min(), z.mean(), z.std())
        # for i in z[0][0]:
        #    print(i)
        out_wav = model.decode(z).squeeze().cpu().numpy()
        print(out_wav.max(), out_wav.min(), out_wav.mean(), out_wav.std())
        print(out_wav.shape)
        soundfile.write(r"C:\Users\29210\Desktop\ylzzvaegan\测试专用rmw1.wav", out_wav,
                        int(model.sr))
