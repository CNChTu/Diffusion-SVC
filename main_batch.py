import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
from ast import literal_eval

from loguru import logger

from rich.progress import track

from tools.infer_tools import DiffusionSVC

if __name__ == '__main__':
    # 讲道理应该从命令行读参数的，但是这样文件会很大，所以干脆在文件里写参数吧，还省的反炒饭。
    input_path = ''  # 目录下必须是只存在可以读的音频文件，如wav，也只推荐wav
    output_path = ''
    # 模型加载设置
    device = 'cuda'  # 设备
    model_path = ''  # 模型位置，会自动读取同目录下的config.yaml
    f0_model = 'crepe'  # f0模式
    f0_max = 800
    f0_min = 65
    # 以下设置和main.py一样，但是需要注意，这里推理是不切片的
    key = 0
    spk_id = 1
    spk_mix_dict = None  # 如要启用，需要是一个字典
    aug_shift = 0
    infer_speedup = 1
    method = 'dpm-solver'
    threhold = -60.0
    k_step = None
    spk_emb_path = None
    spk_emb_dict_path = None
    naive_model_path = None
    index_ratio = 0  # 大于0则使用检索，需要已经训练过检索
    # -------------------------------下面不用动----------------------------------
    diffusion_svc = DiffusionSVC(device=device)  # 加载模型
    diffusion_svc.load_model(model_path=model_path, f0_model=f0_model, f0_max=f0_max, f0_min=f0_min)

    if diffusion_svc.args.model.use_speaker_encoder:  # 如果使用声纹，则处理声纹选项
        diffusion_svc.set_spk_emb_dict(spk_emb_dict_path)
        spk_emb = diffusion_svc.encode_spk_from_path(spk_emb_path)
    else:
        spk_emb = None

    if naive_model_path is not None:
        if k_step is None:
            naive_model_path = None
            logger.warning("Could not shallow diffusion without k_step value when Only set naive_model path")
        else:
            diffusion_svc.load_naive_model(naive_model_path=naive_model_path)

    for file in track(os.listdir(input_path)):
        in_path = os.path.join(input_path, file)
        assert os.path.isfile(in_path)
        out_path = os.path.join(output_path, file)
        in_wav, in_sr = librosa.load(in_path, sr=None)
        out_wav, out_sr = diffusion_svc.infer_from_audio(in_wav, sr=in_sr, key=key, spk_id=spk_id,
                                                         spk_mix_dict=spk_mix_dict,
                                                         aug_shift=aug_shift,
                                                         infer_speedup=infer_speedup, method=method, k_step=k_step,
                                                         show_progress=False, spk_emb=spk_emb, threhold=threhold,
                                                         index_ratio=index_ratio)
        sf.write(out_path, out_wav, out_sr)
