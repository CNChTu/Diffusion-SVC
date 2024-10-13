import torch
from nsf_hifigan.nvSTFT import STFT
from nsf_hifigan.models import load_model, load_config
from torchaudio.transforms import Resample
import os
from encoder.hifi_vaegan import InferModel
import json
import yaml
from encoder.fireflygan import FireflyBase
from encoder.evagan import EVAGANBase as EVABase
from encoder.evagan import EVAGANBig as EVABig
from encoder.dct.dct import DCT, IDCT
from encoder.wavs.wavs import WAVS, IWAVS
from encoder.hifi_vaegan2.modules.models import Encoder, Generator


def load_vocoder_for_save(vocoder_type, model_path, device='cpu'):
    if vocoder_type == 'nsf-hifigan':
        vocoder = NsfHifiGAN(model_path, device=device)
    elif vocoder_type == 'nsf-hifigan-log10':
        vocoder = NsfHifiGANLog10(model_path, device=device)
    elif vocoder_type == 'hifivaegan':
        vocoder = HiFiVAEGAN(model_path, device=device)
    elif vocoder_type == 'hifivaegan2':
        vocoder = HiFiVAEGAN2(model_path, device=device)
    elif vocoder_type == 'fireflygan-base':
        vocoder = FireFlyGANBase(model_path, device=device)
    elif vocoder_type == 'evagan-base':
        vocoder = EVAGANBase(model_path, device=device)
    elif vocoder_type == 'evagan-big':
        vocoder = EVAGANBig(model_path, device=device)
    elif vocoder_type == 'dct512':
        vocoder = DCT512(model_path, device=device)
    elif vocoder_type == 'dct512log':
        vocoder = DCT512(model_path, device=device, l_norm=True)
    elif vocoder_type == 'wavs512':
        vocoder = WAVS512(model_path, device=device)
    else:
        raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")
    out_dict = vocoder.load_model_for_combo(model_path=model_path)
    return out_dict


class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if type(vocoder_ckpt) == dict:
            '''传入的是config + model'''
            print(f"  [INFO] Loading vocoder from \'.ptc\' file.")
            assert 'config' in vocoder_ckpt.keys(), "config not in vocoder_ckpt"
            assert 'model' in vocoder_ckpt.keys(), "model not in vocoder_ckpt"

        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt, device=device)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt, device=device)
        elif vocoder_type == 'hifivaegan':
            self.vocoder = HiFiVAEGAN(vocoder_ckpt, device=device)
        elif vocoder_type == 'hifivaegan2':
            self.vocoder = HiFiVAEGAN2(vocoder_ckpt, device=device)
        elif vocoder_type == 'fireflygan-base':
            self.vocoder = FireFlyGANBase(vocoder_ckpt, device=device)
        elif vocoder_type == 'evagan-base':
            self.vocoder = EVAGANBase(vocoder_ckpt, device=device)
        elif vocoder_type == 'evagan-big':
            self.vocoder = EVAGANBig(vocoder_ckpt, device=device)
        elif vocoder_type == 'dct512':
            self.vocoder = DCT512(vocoder_ckpt, device=device)
        elif vocoder_type == 'dct512log':
            self.vocoder = DCT512(vocoder_ckpt, device=device, l_norm=True)
        elif vocoder_type == 'wavs512':
            self.vocoder = WAVS512(vocoder_ckpt, device=device)
        else:
            raise ValueError(f" [x] Unknown vocoder: {vocoder_type}")

        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()

    def extract(self, audio, sample_rate, keyshift=0):

        # resample
        if sample_rate == self.vocoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate,
                                                         lowpass_filter_width=128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)

        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift)  # B, n_frames, bins
        return mel

    def infer(self, mel, f0):
        f0 = f0[:, :mel.size(1), 0]  # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio


class WAVS512(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.h_sampling_rate = 44100
        self.h_num_mels = 512
        self.h_hop_size = 512
        self.wavs = WAVS(self.h_hop_size)
        self.iwavs = IWAVS(self.h_hop_size)

    def sample_rate(self):
        return self.h_sampling_rate

    def hop_size(self):
        return self.h_hop_size

    def dimension(self):
        return self.h_num_mels

    def extract(self, audio, keyshift=0):
        assert keyshift == 0
        with torch.no_grad():
            audio = audio.to(self.device)
            mel = self.wavs(audio)  # B, n_frames, bins
        return mel

    def forward(self, mel, f0):  # mel: B, n_frames, bins; f0: B, n_frames
        assert mel.shape[-1] == 512
        with torch.no_grad():
            audio = self.iwavs(mel)
            return audio.unsqueeze(1)  # B, 1, T

    def load_model_for_combo(self, model_path=None, device='cpu'):
        config = {"sampling_rate": self.sampling_rate, "num_mels": self.num_mels, "hop_size": self.hop_size}
        model = NothingFlag()
        out_dict = {
            "config": config,
            "model": model
        }
        return out_dict


class DCT512(torch.nn.Module):
    def __init__(self, model_path, device=None, l_norm=False):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.h_sampling_rate = 44100
        self.h_num_mels = 512
        self.h_hop_size = 512
        self.dct = DCT(self.h_hop_size)
        self.idct = IDCT(self.h_hop_size)
        self.l_norm = l_norm

    def sample_rate(self):
        return self.h_sampling_rate

    def hop_size(self):
        return self.h_hop_size

    def dimension(self):
        return self.h_num_mels

    def extract(self, audio, keyshift=0):
        assert keyshift == 0
        with torch.no_grad():
            audio = audio.to(self.device)
            mel = self.dct(audio)  # B, n_frames, bins
            if self.l_norm:
                mel = torch.where(mel < 0, -torch.log(2 - mel) + 0.6, torch.log(2 + mel) - 0.6)
        return mel

    def forward(self, mel, f0):  # mel: B, n_frames, bins; f0: B, n_frames
        assert mel.shape[-1] == 512
        with torch.no_grad():
            if self.l_norm:
                mel = torch.where(mel < 0, -torch.exp(-mel + 0.6) + 2, torch.exp(mel + 0.6) - 2)
            audio = self.idct(mel)
            return audio.unsqueeze(1)  # B, 1, T

    def load_model_for_combo(self, model_path=None, device='cpu'):
        config = {"sampling_rate": self.sampling_rate, "num_mels": self.num_mels, "hop_size": self.hop_size}
        model = NothingFlag()
        out_dict = {
            "config": config,
            "model": model
        }
        return out_dict


class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if type(model_path) == dict:
            '''传入的是config + model'''
            self.model_path = None
            self.model, self.h = load_model(model_path, device=self.device)
        else:
            # 原始模式
            self.model_path = model_path
            self.model = None
            self.h = load_config(model_path)
        self.stft = STFT(
            self.h.sampling_rate,
            self.h.num_mels,
            self.h.n_fft,
            self.h.win_size,
            self.h.hop_size,
            self.h.fmin,
            self.h.fmax)

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def dimension(self):
        return self.h.num_mels

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)  # B, n_frames, bins
        return mel

    def forward(self, mel, f0):  # mel: B, n_frames, bins; f0: B, n_frames
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio  # B, 1, T

    def load_model_for_combo(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = self.model_path
        config, model = load_model(model_path, device=device, load_for_combo=True)
        out_dict = {
            "config": config,
            "model": model
        }
        return out_dict


class NsfHifiGANLog10(NsfHifiGAN):
    def forward(self, mel, f0):
        if self.model is None:
            print('| Load HifiGAN: ', self.model_path)
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class HiFiVAEGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 如果model_path是字典，说明传入的是config + model
        if type(model_path) == dict:
            self.config_path = None
            self.model_path = None
            self.model = InferModel(model_path, model_path=None, device=device, _load_from_state_dict=True)
        else:
            self.model_path = model_path
            self.config_path = os.path.join(os.path.split(model_path)[0], 'config.json')
            self.model = InferModel(self.config_path, self.model_path, device=device, _load_from_state_dict=False)

    def sample_rate(self):
        return self.model.sr

    def hop_size(self):
        return self.model.hop_size

    def dimension(self):
        return self.model.inter_channels

    def extract(self, audio, keyshift=0, only_z=False):
        assert keyshift == 0
        if audio.shape[-1] % self.model.hop_size == 0:
            audio = torch.cat((audio, torch.zeros_like(audio[:, :1])), dim=-1)
        if keyshift != 0:
            raise ValueError("HiFiVAEGAN could not use keyshift!")
        with torch.no_grad():
            z, m, logs = self.model.encode(audio)
            if only_z:
                return z.transpose(1, 2)
            mel = torch.stack((m.transpose(-1, -2), logs.transpose(-1, -2)), dim=-1)
        return mel

    def forward(self, mel, f0):
        with torch.no_grad():
            z = mel.transpose(1, 2)
            audio = self.model.decode(z)
            return audio

    def load_model_for_combo(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = self.model_path
            assert self.config_path is not None
        config_path = os.path.join(os.path.split(model_path)[0], 'config.json')
        with open(config_path, "r", encoding='utf-8') as f:
            data = f.read()
        config = json.loads(data)
        model_state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        out_dict = {
            "config": config,
            "model": model_state_dict
        }
        return out_dict


class HiFiVAEGAN2(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 如果model_path是字典，说明传入的是config + model
        if type(model_path) == dict:
            self.config_path = None
            self.model_path = None
            self.config = model_path['config']
            self.enc = Encoder(self.config['hps']).to(device)
            self.dec = Generator(self.config['hps']).to(device)
            self.enc.load_state_dict(model_path['model']['encoder'])
            self.dec.load_state_dict(model_path['model']['decoder'])
            self.enc.eval()
            self.dec.eval()
        else:
            self.model_path = model_path
            self.config_path = os.path.join(os.path.split(model_path)[0], 'config.json')
            with open(self.config_path, "r", encoding='utf-8') as f:
                self.config = json.load(f)
            self.enc = Encoder(self.config['hps']).to(device)
            self.dec = Generator(self.config['hps']).to(device)
            model_state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
            self.enc.load_state_dict(model_state_dict['encoder'])
            self.dec.load_state_dict(model_state_dict['decoder'])
            self.enc.eval()
            self.dec.eval()

    def sample_rate(self):
        return self.config["hps"]["sampling_rate"]

    def hop_size(self):
        return self.config["hop_size"]

    def dimension(self):
        return self.config["hps"]["inter_channels"]

    def extract(self, audio, keyshift=0, only_z=False):
        assert keyshift == 0
        if audio.shape[-1] % self.config["hop_size"] == 0:
            audio = torch.cat((audio, torch.zeros_like(audio[:, :1])), dim=-1)
        if keyshift != 0:
            raise ValueError("HiFiVAEGAN could not use keyshift!")
        with torch.no_grad():
            z, m, logs = self.enc(audio)
            if only_z:
                return z.transpose(1, 2)
            mel = torch.stack((m.transpose(-1, -2), logs.transpose(-1, -2)), dim=-1)
        return mel

    def forward(self, mel, f0):
        with torch.no_grad():
            z = mel.transpose(1, 2)
            audio = self.dec(z)
            return audio

    def load_model_for_combo(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = self.model_path
            assert self.config_path is not None
        config_path = os.path.join(os.path.split(model_path)[0], 'config.json')
        with open(config_path, "r", encoding='utf-8') as f:
            data = f.read()
        config = json.loads(data)
        model_state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        out_dict = {
            "config": config,
            "model": model_state_dict
        }
        return out_dict


class FireFlyGANBase(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if type(model_path) == dict:
            '''传入的是config + model'''
            self.config_path = None
            self.model_path = None
            config = dict(model_path['config'])
            _loaded_state_dict = model_path['model']
            if str(config['model']) == 'fire_fly_gan_base_20240407':
                self.model = FireflyBase(None, loaded_state_dict=_loaded_state_dict)
            else:
                raise ValueError(f" [x] Unknown model: {config['model']}")

        else:
            # 原始模式
            self.config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
            self.model_path = model_path
            config = dict(load_config_dict_from_yaml_path(self.config_path))
            if str(config["model"]) == 'fire_fly_gan_base_20240407':
                self.model = FireflyBase(self.model_path)
            else:
                raise ValueError(f" [x] Unknown model: {config['model']}")

        self.model.eval()
        self.model.to(self.device)
        self.sr = config["sampling_rate"]
        self.hopsize = config["hop_size"]
        self.dim = config["num_mels"]
        self.stft = STFT(
            self.sr,
            self.dim,
            config["n_fft"],
            config["win_size"],
            config["hop_size"],
            config["fmin"],
            config["fmax"]
        )

    def sample_rate(self):
        return self.sr

    def hop_size(self):
        return self.hopsize

    def dimension(self):
        return self.dim

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)  # B, n_frames, bins
        return mel

    def forward(self, mel, f0=None):
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c)
        return audio

    def load_model_for_combo(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = self.model_path
            assert self.config_path is not None
        config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        with open(config_path, "r") as config:
            config = yaml.safe_load(config)
        model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        out_dict = {
            "config": config,
            "model": model
        }
        return out_dict


class EVAGANBase(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if type(model_path) == dict:
            '''传入的是config + model'''
            self.config_path = None
            self.model_path = None
            config = dict(model_path['config'])
            _loaded_state_dict = model_path['model']
            if str(config['model']) == 'evagan_base_20240808':
                self.model = EVABase(None, loaded_state_dict=_loaded_state_dict)
            else:
                raise ValueError(f" [x] Unknown model: {config['model']}")

        else:
            # 原始模式
            self.config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
            self.model_path = model_path
            config = dict(load_config_dict_from_yaml_path(self.config_path))
            if str(config["model"]) == 'evagan_base_20240808':
                self.model = EVABase(self.model_path)
            else:
                raise ValueError(f" [x] Unknown model: {config['model']}")

        self.model.eval()
        self.model.to(self.device)
        self.sr = config["sampling_rate"]
        self.hopsize = config["hop_size"]
        self.dim = config["num_mels"]
        self.stft = STFT(
            self.sr,
            self.dim,
            config["n_fft"],
            config["win_size"],
            config["hop_size"],
            config["fmin"],
            config["fmax"]
        )

    def sample_rate(self):
        return self.sr

    def hop_size(self):
        return self.hopsize

    def dimension(self):
        return self.dim

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)  # B, n_frames, bins
        return mel

    def forward(self, mel, f0=None):
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c)
        return audio

    def load_model_for_combo(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = self.model_path
            assert self.config_path is not None
        config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        with open(config_path, "r") as config:
            config = yaml.safe_load(config)
        model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        out_dict = {
            "config": config,
            "model": model
        }
        return out_dict


class EVAGANBig(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if type(model_path) == dict:
            '''传入的是config + model'''
            self.config_path = None
            self.model_path = None
            config = dict(model_path['config'])
            _loaded_state_dict = model_path['model']
            if str(config['model']) == 'evagan_big_20240808':
                self.model = EVABig(None, loaded_state_dict=_loaded_state_dict)
            else:
                raise ValueError(f" [x] Unknown model: {config['model']}")

        else:
            # 原始模式
            self.config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
            self.model_path = model_path
            config = dict(load_config_dict_from_yaml_path(self.config_path))
            if str(config["model"]) == 'evagan_big_20240808':
                self.model = EVABig(self.model_path)
            else:
                raise ValueError(f" [x] Unknown model: {config['model']}")

        self.model.eval()
        self.model.to(self.device)
        self.sr = config["sampling_rate"]
        self.hopsize = config["hop_size"]
        self.dim = config["num_mels"]
        self.stft = STFT(
            self.sr,
            self.dim,
            config["n_fft"],
            config["win_size"],
            config["hop_size"],
            config["fmin"],
            config["fmax"]
        )

    def sample_rate(self):
        return self.sr

    def hop_size(self):
        return self.hopsize

    def dimension(self):
        return self.dim

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)  # B, n_frames, bins
        return mel

    def forward(self, mel, f0=None):
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c)
        return audio

    def load_model_for_combo(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = self.model_path
            assert self.config_path is not None
        config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        with open(config_path, "r") as config:
            config = yaml.safe_load(config)
        model = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        out_dict = {
            "config": config,
            "model": model
        }
        return out_dict


class NothingFlag:
    pass


def load_config_dict_from_yaml_path(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    return args
