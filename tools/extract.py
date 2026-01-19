import numpy as np
import librosa
import argparse
import torch
import torch.nn.functional as F
import parselmouth
import soundfile as sf


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    return parser.parse_args(args=args, namespace=namespace)

def extract_base_harmonic(
                audio, # [B, T]
                sample_rate,
                f0,    # [B, n_frames, 1]
                time_step=None,
                win_time=0.08,
                half_width=3.5):
    if time_step is None:
        hop_length = int(audio.shape[-1] // (f0.shape[1] - 1))
    else:
        hop_length = int(np.round(time_step * sample_rate))
    win_length = int(np.round(win_time * sample_rate))
    phase = 2 * np.pi * torch.arange(win_length).to(audio_t) / win_length
    nuttall_window = 0.355768 - 0.487396 * torch.cos(phase) + 0.144232 * torch.cos(2 * phase) - 0.012604 * torch.cos(3 * phase)
    spec = torch.stft(
                audio,
                n_fft = win_length,
                win_length = win_length,
                hop_length = hop_length,
                window = nuttall_window,
                center = True,
                return_complex = True).permute(0, 2, 1) # [B, n_frames, n_spec]
    idx = torch.arange(spec.shape[-1]).unsqueeze(0).unsqueeze(0).to(f0) # [1, 1, n_spec]
    center = f0 * win_length / sample_rate
    start = torch.clip(center - half_width, min=0)
    end = torch.clip(center + half_width, max=spec.shape[-1])
    idx_mask = (center >= 1) & (idx >= start) & (idx < end) # [B, n_frames, n_spec]
    if idx_mask.shape[1] < spec.shape[1]:
        idx_mask = F.pad(idx_mask, (0, 0, 0, spec.shape[1] - idx_mask.shape[1]))
    spec = spec * idx_mask[:, : spec.shape[1], :]
    base_harmonic = torch.istft(
                        spec.permute(0, 2, 1),
                        n_fft = win_length,
                        win_length = win_length,
                        hop_length = hop_length,
                        window = nuttall_window,
                        center = True,
                        length = audio.shape[-1])
    return base_harmonic
    
    
if __name__ == '__main__':
    cmd = parse_args()
    
    device = cmd.device
    input_path = cmd.input
    output_path = cmd.output
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    audio, sr = librosa.load(input_path, sr=None)
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device) # [B, T]
    
    # f0 analysis using parselmouth
    time_step = 0.01
    hop_size = int(np.round(sr * time_step))
    f0 = parselmouth.Sound(audio, sr).to_pitch_ac(
            time_step=hop_size/sr, voicing_threshold=0.6,
            pitch_floor=65, pitch_ceiling=1100).selected_array['frequency']
    pad_size=(int(len(audio) // hop_size) - len(f0) + 1) // 2
    f0 = f0[pad_size :]
    
    # interpolate the unvoiced f0 
    uv = f0 == 0
    f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device) # [B, T]
    f0_t = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0) # [B, n_frames, 1]
    base_harmonic_t = extract_base_harmonic(audio_t, sr, f0_t, time_step)
    base_harmonic = base_harmonic_t.squeeze(0).cpu().numpy()
    sf.write(output_path, base_harmonic, sr)
