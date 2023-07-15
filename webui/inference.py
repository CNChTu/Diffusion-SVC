import librosa
import soundfile as sf
from ast import literal_eval
from loguru import logger

import os
import sys

import torch

sys.path.append(".")
# sys.path.append("..")

from tools.infer_tools import DiffusionSVC
# from diffusion_svc import DiffusionSVC

# 全局变量，用于存储加载的模型
diffusion_svc = None


def load_model(model_path, pitch_extractor='crepe', isNaive=False, naive_model=None, device=None, f0_min=50, f0_max=1100):
    logger.info("Loading model...")
    global diffusion_svc
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion_svc = DiffusionSVC(device=device)
    diffusion_svc.load_model(model_path=model_path, f0_model=pitch_extractor, f0_max=f0_max, f0_min=f0_min)
    logger.success("model loaded successfully.")
    if isNaive and naive_model is not None:
        logger.info("Loading naive {}",naive_model)
        diffusion_svc.load_naive_model(naive_model_path=naive_model)
    # elif naive_model is not None:
        # logger.warning("Could not shallow diffusion without k_step value when Only set naive_model path")


def audio_processing(device, input_file, output_file, spk_id=1, spk_mix_dict=None, key=0,
                     formant_shift_key=0, threhold=-60, threhold_for_split=-40, min_len=5000,
                     speedup=10, method='dpm-solver', k_step=None,
                     index_ratio=0):
    global diffusion_svc

    if diffusion_svc is None:
        raise Exception("Model has not been loaded. Please call load_model() before audio_processing().")

    spk_mix_dict = literal_eval(spk_mix_dict) if spk_mix_dict is not None else None
    logger.debug("spk_mix_dict {}",spk_mix_dict)
    spk_emb = None

    in_wav, in_sr = librosa.load(input_file, sr=None)
    if len(in_wav.shape) > 1:
        in_wav = librosa.to_mono(in_wav)

    out_wav, out_sr = diffusion_svc.infer_from_long_audio(
        in_wav, sr=in_sr,
        key=float(key),
        spk_id=int(spk_id),
        spk_mix_dict=spk_mix_dict,
        aug_shift=int(formant_shift_key),
        infer_speedup=int(speedup),
        method=method,
        k_step=k_step,
        show_progress=True,
        spk_emb=spk_emb,
        threhold=float(threhold),
        threhold_for_split=float(threhold_for_split),
        min_len=int(min_len),
        index_ratio=float(index_ratio)
    )
    sf.write(output_file, out_wav, out_sr)
# load_model("./exp/diffusionsvc/model_112000.pt","dio")

# audio_processing("cuda","input.wav","output.wav",1,'{"1": 0.5,"2": 0.5}',0,0,-60,-40,5000,10,"dpm-solver")
# load_model("./exp/diffusionsvc/model_112000.pt","dio")
# audio_processing("cuda","input.wav","output.wav",1,'{"1": 0.5,"2": 0.5}')
