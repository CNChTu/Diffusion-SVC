import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pyworld as pw
import parselmouth
import torchcrepe
import librosa
import fsspec
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, AutoFeatureExtractor, Wav2Vec2BertModel, AutoProcessor
from fairseq import checkpoint_utils
from encoder.hubert.model import HubertSoft
from encoder.speaker_encoder.model import SpeakerEncoder as TTSSpeakerEncoder
import scipy.signal
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
from torch.optim.lr_scheduler import StepLR
from encoder.funasr.model import SpeechEncoder
from encoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from encoder.whisper.model import ModelDimensions, Whisper

CREPE_RESAMPLE_KERNEL = {}


class SpeakerEncoder:
    def __init__(self, speaker_encoder, speaker_encoder_config, speaker_encoder_ckpt, encoder_sample_rate,
                 device='cuda',
                 use_torchaudio=False):
        self.use_torchaudio = use_torchaudio
        self.encoder_sample_rate = encoder_sample_rate
        self.device = device
        self.resample_kernel = {}
        if speaker_encoder == "ge2e":
            self.encoder = GE2E(speaker_encoder_config, speaker_encoder_ckpt, device=device)
        else:
            raise ValueError(f" [x] Unknown speaker encoder: {speaker_encoder}")

    def __call__(self, audio=None, audio_t=None,
                 sample_rate=44100):  # if use torchaudio, audio_t must be a tensor; else audio must be a np
        audio_res = None
        if sample_rate == self.encoder_sample_rate:
            if self.use_torchaudio and (audio_t is not None):
                audio_res = audio_t.cpu().numpy().squeeze(0)
            else:
                if audio is not None:
                    audio_res = audio
        else:
            key_str = str(sample_rate)
            if self.use_torchaudio and (audio_t is not None):
                if key_str not in self.resample_kernel:
                    self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate,
                                                             lowpass_filter_width=128).to(self.device)
                audio_res = self.resample_kernel[key_str](audio_t).cpu().numpy().squeeze(0)
            else:
                if audio is not None:
                    audio_res = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.encoder_sample_rate)
        assert audio_res is not None
        return self.encoder(audio_res)

    def mean_spk_emb_from_wav_list(self, audio_list, sr_list):
        assert len(audio_list) == len(sr_list)
        batch_spk_emb = None
        print("Get mean spk_emb from audio_list")
        for index in tqdm(range(len(audio_list))):
            audio = audio_list[index]
            sample_rate = sr_list[index]
            f_len = int(50 * len(audio) / sample_rate)  # 50f/s is for sr=16000，hop_size=320
            spk_emb = self.__call__(audio=audio, sample_rate=sample_rate)
            spk_emb = np.tile(spk_emb, (f_len, 1))
            if batch_spk_emb is None:
                batch_spk_emb = spk_emb
            else:
                batch_spk_emb = np.concatenate([spk_emb, batch_spk_emb], axis=0)
        return np.mean(batch_spk_emb, axis=0)

    def mean_spk_emb_from_path_list(self, path_list):
        batch_spk_emb = None
        print("Get mean spk_emb from path_list")
        for path in tqdm(path_list):
            audio, sample_rate = librosa.load(path, sr=None)
            f_len = int(50 * len(audio) / sample_rate)  # 50f/s is for sr=16000，hop_size=320
            spk_emb = self.__call__(audio=audio, sample_rate=sample_rate)
            spk_emb = np.tile(spk_emb, (f_len, 1))
            if batch_spk_emb is None:
                batch_spk_emb = spk_emb
            else:
                batch_spk_emb = np.concatenate([spk_emb, batch_spk_emb], axis=0)
        return np.mean(batch_spk_emb, axis=0)


class GE2E:
    def __init__(self, config_path, ckpt_path, device='cuda'):
        import json5
        with open(config_path) as f:
            self.config = json5.load(f)
        # load model
        self.model = TTSSpeakerEncoder(
            self.config['model']["input_dim"],
            self.config['model']["proj_dim"],
            self.config['model']["lstm_dim"],
            self.config['model']["num_lstm_layers"],
        )
        with fsspec.open(ckpt_path, "rb") as f:
            state = torch.load(f, map_location=device)
        self.model.load_state_dict(state["model"])
        self.model = self.model.to(device)
        self.model.eval()

        self.preemphasis = self.config["audio"]["preemphasis"]
        self.do_amp_to_db_mel = True
        self.fft_size = self.config["audio"]["fft_size"]
        self.hop_length = self.config["audio"]["hop_length"]
        self.win_length = self.config["audio"]["win_length"]
        self.signal_norm = self.config['audio']['signal_norm']
        self.num_mels = self.config["audio"]["num_mels"]
        self.ref_level_db = self.config["audio"]['ref_level_db']
        self.min_level_db = self.config["audio"]['min_level_db']
        self.symmetric_norm = self.config["audio"]['symmetric_norm']
        self.clip_norm = self.config["audio"]['clip_norm']
        self.max_norm = self.config["audio"]['max_norm']
        self.stft_pad_mode = 'reflect'
        self.spec_gain = 20.0
        self.base = 10
        self.device = device
        mel_basis = librosa.filters.mel(
            sr=self.config["audio"]["sample_rate"], n_fft=self.config["audio"]['fft_size'],
            n_mels=self.num_mels, fmin=self.config["audio"]['mel_fmin'],
            fmax=self.config["audio"]['mel_fmax']
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    def __call__(self, audio, use_old_infer=True):
        y = audio
        if self.preemphasis != 0:
            y = scipy.signal.lfilter([1, -self.preemphasis], [1], y)
        D = librosa.stft(
            y=y,
            n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length, pad_mode=self.stft_pad_mode,
            window="hann", center=True)
        D = np.abs(D)
        D = np.dot(self.mel_basis, D)
        if self.base == 10:
            spec = self.spec_gain * np.log10(np.maximum(1e-5, D))
        else:
            spec = self.spec_gain * np.log(np.maximum(1e-5, D))
        spec = self.normalize(spec).astype(np.float32)
        spec = torch.from_numpy(spec.T)
        spec = spec.to(self.device)
        spec = spec.unsqueeze(0)
        if use_old_infer:
            spk_emb = self.compute_embedding_old(spec).detach().cpu().numpy()
        else:
            spk_emb = self.model.compute_embedding(spec).detach().cpu().numpy()
        return spk_emb.squeeze()

    def normalize(self, S) -> np.ndarray:
        S = S.copy()
        if self.signal_norm:
            S -= self.ref_level_db
            S_norm = (S - self.min_level_db) / (-self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def compute_embedding_old(self, x, num_frames=250, num_eval=10, return_mean=True):
        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.model.inference(frames_batch)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)

        return embeddings


class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate=44100, hop_size=512, f0_min=65, f0_max=800,
                 block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.transformer_f0 = None
        self.rmvpe = None
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, uv_interp=False, device=None, silence_front=0, sr=None, mel = None):  # audio: 1d numpy array
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
            if (self.f0_extractor == 'crepe') and (sr != self.sample_rate):
                key_str = str(sr)
                if key_str not in CREPE_RESAMPLE_KERNEL:
                    CREPE_RESAMPLE_KERNEL[key_str] = Resample(sr, 16000, lowpass_filter_width=128)
                self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
            self.sample_rate = sr

        if audio is not None:
            # extractor start time
            raw_audio = audio
            n_frames = int(len(audio) // self.hop_size) + 1

            start_frame = int(silence_front * self.sample_rate / self.hop_size)
            real_silence_front = start_frame * self.hop_size / self.sample_rate
            audio = audio[int(np.round(real_silence_front * self.sample_rate)):]

        # extract f0 using parselmouth
        if self.f0_extractor == 'parselmouth':
            f0 = parselmouth.Sound(audio, self.sample_rate).to_pitch_ac(
                time_step=self.hop_size / self.sample_rate,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max).selected_array['frequency']
            pad_size = start_frame + (int(len(audio) // self.hop_size) - len(f0) + 1) // 2
            f0 = np.pad(f0, (pad_size, n_frames - len(f0) - pad_size))

        # extract f0 using dio
        elif self.f0_extractor == 'dio':
            _f0, t = pw.dio(
                audio.astype('double'),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                channels_in_octave=2,
                frame_period=(1000 * self.hop_size / self.sample_rate))
            f0 = pw.stonemask(audio.astype('double'), _f0, t, self.sample_rate)
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # extract f0 using harvest
        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=(1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        # extract f0 using crepe
        elif self.f0_extractor == 'crepe':
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            resample_kernel = self.resample_kernel.to(device)
            wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))

            f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full',
                                        batch_size=512, device=device, return_periodicity=True)
            pd = median_pool_1d(pd, 4)
            f0 = torchcrepe.threshold.At(0.05)(f0, pd)
            f0 = masked_avg_pool_1d(f0, 4)

            f0 = f0.squeeze(0).cpu().numpy()
            f0 = np.array(
                [f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in
                 range(n_frames - start_frame)])
            f0 = np.pad(f0, (start_frame, 0))

        elif self.f0_extractor == "fcpe":
            _JUMP_SAFE_PAD = False
            if self.transformer_f0 is None:
                from encoder.fcpe.model import FCPEInfer
                self.transformer_f0 = FCPEInfer(model_path='pretrain/fcpe/fcpe.pt')
            if _JUMP_SAFE_PAD:
                raw_audio = audio
            if mel is None:
                f0 = self.transformer_f0(audio=raw_audio, sr=self.sample_rate)
            else:
                if audio is None:
                    n_frames = mel.shape[1]
                f0 = self.transformer_f0.model(mel=mel, infer=True, return_hz_f0=True)
            f0 = f0.transpose(1, 2)
            if not _JUMP_SAFE_PAD:
                f0 = torch.nn.functional.interpolate(f0, size=int(n_frames), mode='nearest')
            f0 = f0.transpose(1, 2)
            f0 = f0.squeeze().cpu().numpy()
            if _JUMP_SAFE_PAD:
                f0 = np.array(
                    [f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.01)), len(f0) - 1))] for n in
                     range(n_frames - start_frame)])
                f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        elif self.f0_extractor == "rmvpe":
            if self.rmvpe is None:
                from encoder.rmvpe import RMVPE
                self.rmvpe = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")

        # interpolate the unvoiced f0
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0


class Volume_Extractor:
    def __init__(self, hop_size=512, block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.hop_size = hop_size
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, sr=None):  # audio: 1d numpy array
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        volume = np.array(
            [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        '''
        if isinstance(audio, torch.Tensor):
            n_frames = int(audio.size(-1) // self.hop_size) + 1
            audio2 = audio ** 2
            audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)),
                                             mode='reflect')
            audio_frame = torch.nn.functional.unfold(audio2[:, None, None, :], (1, int(self.hop_size)),
                                                     stride=int(self.hop_size))[:, :, :n_frames]
            volume = audio_frame.mean(dim=1)[0]
            volume = torch.sqrt(volume).squeeze().cpu().numpy()
        else:
            n_frames = int(len(audio) // self.hop_size) + 1
            audio2 = audio ** 2
            audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
            volume = np.array(
                [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
            volume = np.sqrt(volume)
        '''
        return volume

    def get_mask_from_volume(self, volume, threhold=-60.0,device='cpu'):
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.block_size).squeeze(-1)
        return mask

class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate=16000, encoder_hop_size=320, device=None,
                 cnhubertsoft_gate=10, units_forced_mode='nearest'):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.encoder = encoder

        if cnhubertsoft_gate is None:
            cnhubertsoft_gate = 10
        if units_forced_mode is None:
            units_forced_mode = 'left'
        self.units_forced_mode = units_forced_mode

        is_loaded_encoder = False
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt).to(device)
            is_loaded_encoder = True
        if encoder == 'hubertbase':
            self.model = Audio2HubertBase(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertbase768':
            self.model = Audio2HubertBase768(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertbase768l12':
            self.model = Audio2HubertBase768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertlarge1024l24':
            self.model = Audio2HubertLarge1024L24(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'wav2vec2large1024l24':
            self.model = Audio2Wav2Vec2Large1024L24(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec':
            self.model = Audio2ContentVec(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768':
            self.model = Audio2ContentVec768(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768l12':
            self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'cnhubertsoftfish':
            self.model = CNHubertSoftFish(encoder_ckpt, device=device, gate_size=cnhubertsoft_gate)
            is_loaded_encoder = True
        if encoder == 'funasr':
            self.model = FunASR(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'whisper':
            self.model = Whisper(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'w2v-bert':
            self.model = Wav2Vec2Bert(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder in ('wav2vec2', 'wav2vec2-xlsr-53-espeak-cv-ft'):
            self.model = Wav2Vec2(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
        print(f"Units Forced Mode:{self.units_forced_mode}")

        if self.units_forced_mode == 'rfa512to441':
            encoder_sample_rate = encoder_sample_rate * 441 / 512
        if self.units_forced_mode == 'rfa441to512':
            encoder_sample_rate = encoder_sample_rate * 512 / 441

        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size

    def encode(self,
               audio,  # B, T
               sample_rate,
               hop_size,
               padding_mask=None):

        # resample
        if self.units_forced_mode not in ('rfa441to512', 'rfa512to441'):
            if sample_rate == self.encoder_sample_rate:
                audio_res = audio
            else:
                key_str = str(sample_rate)
                if key_str not in self.resample_kernel:
                    self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate).to(self.device)
                audio_res = self.resample_kernel[key_str](audio)
        else:
            if isinstance(audio, np.ndarray):
                _audio = audio
            else:
                _audio = audio.cpu().numpy()
            audio_res = librosa.resample(_audio, orig_sr=sample_rate, target_sr=self.encoder_sample_rate)
            audio_res = torch.from_numpy(audio_res).to(self.device)

        if self.encoder == 'w2v-bert' and isinstance(audio_res, torch.Tensor):
            audio_res = audio_res.cpu().numpy()
            
        # encode
        if isinstance(audio_res, torch.Tensor) and audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res, padding_mask=padding_mask)

        if units.shape[0] == 1:
            units = units.squeeze(0)

        return units


def units_forced_alignment(units, audio = None, sample_rate = None, hop_size = None, n_frames = None, scale_factor = None, units_forced_mode = 'nearest'):
    # alignment
    assert (audio is not None and sample_rate is not None and hop_size is not None) or n_frames is not None or scale_factor is not None
    n_frames = int(audio.size(-1) // hop_size + 1) if n_frames is None else n_frames
    unit_is_tensor = True
    units_dim = len(units.shape)
    if isinstance(units, np.ndarray):
        units = torch.from_numpy(units)
        unit_is_tensor = False
    if units_dim == 2:
        units = units.unsqueeze(0)
    if units_forced_mode == 'left':
        assert scale_factor is not None
        index = torch.clamp(torch.round(scale_factor * torch.arange(n_frames).to(self.device)).long(), max=units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
    elif units_forced_mode in ('rfa441to512', 'rfa512to441'):
        units = units.transpose(1, 2)
        units_aligned = torch.nn.functional.interpolate(units, size=n_frames, scale_factor=scale_factor, mode='nearest')
        units_aligned = units_aligned.transpose(-1, -2)
    else:
        units = units.transpose(-1, -2)
        units_aligned = torch.nn.functional.interpolate(units, size=n_frames, scale_factor=scale_factor, mode=units_forced_mode)
        units_aligned = units_aligned.transpose(-1, -2)
    if not unit_is_tensor:
        units_aligned = units_aligned.numpy()
    if units_dim == 2:
        units_aligned = units_aligned.squeeze(0)
    return units_aligned

class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()

    def forward(self, audio, padding_mask=None):  # B, T
        with torch.inference_mode():
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(1)
            units = self.hubert.units(audio)
            return units

class FunASR(torch.nn.Module):
    def __init__(self, path, device='cpu'):
        super().__init__()
        print(' [Encoder Model] ASR')
        print(' [Loading] ' + path)
        model_path = path
        config_path = path.replace('model.pb', 'config.yaml')
        cmvn_path = path.replace('model.pb', 'am.mvn')
        self.model = SpeechEncoder(model_path, config_path, cmvn_path, device=device)

    @torch.inference_mode()
    def forward(self, audio, padding_mask=None):  # B, T
        audio = audio.view(1,-1)
        if padding_mask is None:
            # 得到每个batch的mask的True的个数
            padding_mask = torch.tensor([audio.shape[-1]]).to(audio.device)
        else:
            padding_mask = padding_mask.sum(1)
        units = self.model(audio, padding_mask)
        return units

class Whisper(torch.nn.Module):
    def __init__(self, path, device='cpu'):
        super().__init__()
        print(' [Encoder Model] ASR')
        print(' [Loading] ' + path)
        checkpoint = torch.load(path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.hidden_dim = dims
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        del model.decoder

    @torch.inference_mode()
    def forward(self, audio, padding_mask=None):  # B, T
        audio = audio.view(1,-1)
        # audln = audio.shape[0]
        # units_len = audln // 320
        # audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).to(self.device)
        with torch.no_grad():
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            units = self.model.encoder(mel).squeeze().data.cpu().float()
            return units
        

class Audio2ContentVec():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        if padding_mask is None:
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        else:
            padding_mask = padding_mask.bool()
            padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = self.hubert.final_proj(logits)
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        if padding_mask is None:
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        else:
            padding_mask = padding_mask.bool()
            padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        if padding_mask is None:
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        else:
            padding_mask = padding_mask.bool()
            padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits
        units = feats  # .transpose(2, 1)
        return units


class CNHubertSoftFish(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', gate_size=10):
        super().__init__()
        self.device = device
        print(' [Encoder Model] CN Hubert Soft fish')
        print(' [Loading] ' + path)
        self.gate_size = gate_size

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./pretrain/TencentGameMate/chinese-hubert-base")
        self.model = HubertModel.from_pretrained("./pretrain/TencentGameMate/chinese-hubert-base")
        self.proj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 256))
        # self.label_embedding = nn.Embedding(128, 256)

        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, audio, padding_mask=None):  # B, T
        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        return self._forward(input_values[0])

    @torch.no_grad()
    def _forward(self, input_values):
        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)

        # Top-k gating
        topk, indices = torch.topk(features, self.gate_size, dim=2)
        features = torch.zeros_like(features).scatter(2, indices, topk)
        features = features / features.sum(2, keepdim=True)

        return features.to(self.device)  # .transpose(1, 2)


class Audio2HubertBase():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        with torch.no_grad():
            if padding_mask is None:
                padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            else:
                padding_mask = padding_mask.bool()
                padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = self.hubert.final_proj(logits)
            return units


class Audio2HubertBase768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        with torch.no_grad():
            if padding_mask is None:
                padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            else:
                padding_mask = padding_mask.bool()
                padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits
            return units


class Audio2HubertBase768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        with torch.no_grad():
            if padding_mask is None:
                padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            else:
                padding_mask = padding_mask.bool()
                padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 12,  # layer 12
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits
            return units


class Audio2HubertLarge1024L24():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Large')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        with torch.no_grad():
            if padding_mask is None:
                padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            else:
                padding_mask = padding_mask.bool()
                padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 24,  # layer 24
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits
            return units


class Audio2Wav2Vec2Large1024L24():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Wav2Vec2Large Large')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):  # B, T
        with torch.no_grad():
            if padding_mask is None:
                padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            else:
                padding_mask = padding_mask.bool()
                padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device)
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits["x"]
            return units

class Wav2Vec2:
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        self.model = Wav2Vec2ForCTC.from_pretrained(path)
        self.model.eval()
        self.model.to(device)

    def __call__(self, audio, padding_mask=None):  # B, T
        with torch.no_grad():
            logits = self.model(audio).logits
        return logits


class Wav2Vec2Bert:
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", cache_dir=path)
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", cache_dir=path)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, audio, padding_mask=None):  # B, T
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        for k, input in inputs.items():
            inputs[k] = input.to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state



class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def masked_avg_pool_1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)


def median_pool_1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1,
                                       mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

class StepLRWithWarmUp(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, warm_up_steps=1000, start_lr = 1e-6, verbose=False):
        self.warm_up_steps = warm_up_steps
        self.start_lr = start_lr
        super().__init__(optimizer,step_size, gamma, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warm_up_steps:
            return [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / self.warm_up_steps
                    for base_lr in self.base_lrs]
        else:
            return super().get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warm_up_steps:
            return [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / self.warm_up_steps
                    for base_lr in self.base_lrs]
        else:
            return super()._get_closed_form_lr()

if __name__ == "__main__":
    w2v_bert = Wav2Vec2Bert(path="pretrin")
    audio1, sr = librosa.load('c:/Users/lenovo/Desktop/gt.wav', sr=16000)
    audio2, sr = librosa.load('c:/Users/lenovo/Desktop/pred_audio_0.wav', sr=16000)
    audio = [audio1, audio2]
    w2v_bert(audio)