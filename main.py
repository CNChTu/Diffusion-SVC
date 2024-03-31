import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
from ast import literal_eval
from tools.infer_tools import DiffusionSVC


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        required=True,
        help="path to the diffusion model checkpoint",
    )
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
    parser.add_argument(
        "-id",
        "--spk_id",
        type=str,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='crepe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe (default) or rmvpe",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-th4sli",
        "--threhold_for_split",
        type=str,
        required=False,
        default=-40,
        help="threhold for split (dB) | default: -40",
    )
    parser.add_argument(
        "-min_len",
        "--min_len",
        type=str,
        required=False,
        default=5000,
        help="min split len | default: 5000",
    )
    parser.add_argument(
        "-speedup",
        "--speedup",
        type=str,
        required=False,
        default=10,
        help="[Diffusion Only]speed up | default: 10",
    )
    parser.add_argument(
        "-inferstep",
        "--infer_step",
        type=str,
        required=False,
        default=10,
        help="[Reflow Only]infer steps | default: 10",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='dpm-solver',
        help="[Diffusion method: ddim, pndm, dpm-solver or unipc]"
             "[Reflow] method: euler or rk45 "
             "| default: dpm-solver",
    )
    parser.add_argument(
        "-kstep",
        "--k_step",
        type=str,
        required=False,
        default=None,
        help="shallow diffusion steps | default: None",
    )
    parser.add_argument(
        "-tstart",
        "--t_start",
        type=str,
        required=False,
        default=None,
        help="[Reflow Only]t start, 0.0 mean Reflow do everything, 1.0 mean do nothing | default: None",
    )
    parser.add_argument(
        "-nmodel",
        "--naive_model",
        type=str,
        required=False,
        default=None,
        help="path to the naive model, shallow diffusion if not None and k_step not None",
    )
    parser.add_argument(
        "-ir",
        "--index_ratio",
        type=str,
        required=False,
        default=0,
        help="index_ratio, if > 0 will use index | default: 0",
    )
    parser.add_argument(
        "-votype",
        "--vocoder_type",
        type=str,
        required=False,
        default=None,
        help="If not None, will use designated vocoder replace the default one | default: None",
    )
    parser.add_argument(
        "-vopath",
        "--vocoder_path",
        type=str,
        required=False,
        default=None,
        help="If use vocoder_type, please set vocoder_path; else nothing to do | default: None",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if cmd.vocoder_type is not None:
        assert cmd.vocoder_path is not None, "If use vocoder_type, please set vocoder_path"
        other_vocoder_dict = {'type': cmd.vocoder_type, 'path': cmd.vocoder_path}
    else:
        other_vocoder_dict = None

    diffusion_svc = DiffusionSVC(device=device)  # 加载模型
    diffusion_svc.load_model(
        model_path=cmd.model,
        f0_model=cmd.pitch_extractor,
        f0_max=float(cmd.f0_max),
        f0_min=float(cmd.f0_min),
        other_vocoder_dict=other_vocoder_dict
    )

    spk_mix_dict = literal_eval(cmd.spk_mix_dict)

    naive_model_path = cmd.naive_model
    if naive_model_path is not None:
        if cmd.k_step is None:
            naive_model_path = None
            print(" [WARN] When get naive_model_path, it could not do shallow diffusion without k_step value")
        else:
            diffusion_svc.load_naive_model(naive_model_path=naive_model_path)

    spk_emb = None

    # load wav
    in_wav, in_sr = librosa.load(cmd.input, sr=None)
    if len(in_wav.shape) > 1:
        in_wav = librosa.to_mono(in_wav)
    # infer
    out_wav, out_sr = diffusion_svc.infer_from_long_audio(
        in_wav, sr=in_sr,
        key=float(cmd.key),
        spk_id=int(cmd.spk_id),
        spk_mix_dict=spk_mix_dict,
        aug_shift=int(cmd.formant_shift_key),
        infer_speedup=int(cmd.speedup),
        method=cmd.method,
        k_step=cmd.k_step,
        use_tqdm=True,
        spk_emb=spk_emb,
        threhold=float(cmd.threhold),
        threhold_for_split=float(cmd.threhold_for_split),
        min_len=int(cmd.min_len),
        index_ratio=float(cmd.index_ratio),
        t_start=cmd.t_start,
        infer_step=int(cmd.infer_step)
    )
    # save
    sf.write(cmd.output, out_wav, out_sr)
