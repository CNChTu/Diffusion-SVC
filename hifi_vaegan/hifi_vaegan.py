import torch
import os
from .modules.models import Generator,Encoder
from nsf_hifigan.nvSTFT import STFT

def load_config(model_path):
    h = torch.load(os.path.join(model_path, 'decoder.pth'))["config"]
    return h

class Hifi_VAEGAN(torch.nn.Module):
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_path = model_path
        self.encoder_model = None
        self.decoder_model = None
        self.scale_factor = 1
        self.h = load_config(model_path)
        self.stft = STFT(
            self.h["sampling_rate"],
            128,
            2048,
            2048,
            512,
            40,
            16000)

    def sample_rate(self):
        return self.h["sampling_rate"]

    def hop_size(self):
        return self.h["hop_size"]

    def dimension(self):
        return self.h["inter_channels"]

    @torch.no_grad()
    def extract(self, audio, keyshift=0, only_z = False, only_mean = False):
        if self.encoder_model is None:
            print('| Load Vaegan Encoder: ', self.model_path)
            state = torch.load(os.path.join(self.model_path, 'encoder.pth'))["model"]
            self.encoder_model = Encoder(self.h)
            self.encoder_model.load_state_dict(state)
            self.encoder_model.eval()
            self.encoder_model.remove_weight_norm()
            self.encoder_model.to(self.device)
        if audio.shape[-1] % self.hop_size() != 0: # PAD
            audio = torch.nn.functional.pad(audio, (0, self.hop_size() - audio.shape[-1] % self.hop_size()))
        z, m, logs = self.encoder_model(audio)
        if only_mean:
            logs = torch.zeros_like(logs)
        z = z * self.scale_factor
        if only_z:
            return z.transpose(-1,-2)
        else:
            rtn = torch.cat([m, logs],dim=-2).transpose(-1,-2)
            return rtn
        
    @torch.no_grad()
    def forward(self, z, f0):
        z = z.transpose(-1,-2)
        z = z / self.scale_factor
        if self.decoder_model is None:
            print('| Load Vaegan Decoder: ', self.model_path)
            state = torch.load(os.path.join(self.model_path, 'decoder.pth'))["model"]
            self.decoder_model = Generator(self.h)
            self.decoder_model.load_state_dict(state)
            self.decoder_model.eval()
            self.decoder_model.remove_weight_norm()
            self.decoder_model.to(self.device)
            
        wav = self.decoder_model(z)
        return wav
    
    @torch.no_grad()
    def get_mel(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)
        return mel