import io
import logging
import torch
import numpy as np
from torchaudio.transforms import Resample
from ast import literal_eval
import soundfile as sf
import librosa
from flask import Flask, request, send_file
from flask_cors import CORS
from loguru import logger

from tools.infer_tools import DiffusionSVC

app = Flask(__name__)

CORS(app)

logging.getLogger("numba").setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    raw_sample = int(float(request_form.get("sampleRate", 0)))

    # get fSafePrefixPadLength
    f_safe_prefix_pad_length = float(request_form.get("fSafePrefixPadLength", 0))
    logger.info("get f_safe_prefix_pad_length:" + str(f_safe_prefix_pad_length))
    if f_safe_prefix_pad_length > 0.025:
        silence_front = f_safe_prefix_pad_length
    else:
        silence_front = 0

    # get sample_method
    sample_method = str(request_form.get("sample_method", None))
    if sample_method == 'None':
        sample_method = 'pndm'
    else:
        sample_method = 'dpm-solver'
    logger.info('get sample_method: {}', sample_method)

    # get speed_up
    speed_up = int(float(request_form.get("sample_interval", 20)))
    logger.info('get speed_up: {}', speed_up)

    # get skip_steps
    skip_steps = int(float(request_form.get("skip_steps", 0)))
    logger.info('get skip_steps: {}', skip_steps)
    kstep = 1000 - skip_steps
    if kstep < speed_up:
        kstep = 300
    if svc_model.args.model.k_step_max is not None:
        k_step_max = int(svc_model.args.model.k_step_max)
        if kstep > k_step_max:
            logger.warning(f"k_step=(1000 - skip_step) must <= k_step_max={k_step_max}, not k_step set to{k_step_max}.")
            kstep = k_step_max

    # 变调信息
    key = float(request_form.get("fPitchChange", 0))

    # 获取spk_id
    raw_speak_id = str(request_form.get("sSpeakId", 0))
    logger.info("get speak_id:" + raw_speak_id)
    if str.isdigit(raw_speak_id):
        spk_id = int(raw_speak_id)
        spk_mix_dict = None
    else:
        spk_id = 1
        spk_mix_dict = literal_eval(raw_speak_id)

    # http获得wav文件并转换
    input_wav_read = io.BytesIO(wave_file.read())
    audio, read_sample_rate = librosa.load(input_wav_read, sr=None, mono=True)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    inlen = len(audio)

    # 模型推理
    _audio, _model_sr = svc_model.infer_from_audio_for_realtime(
        audio, read_sample_rate, key,
        spk_id=spk_id,
        spk_mix_dict=spk_mix_dict,
        aug_shift=0,
        infer_speedup=speed_up,
        method=sample_method,
        k_step=kstep,
        show_progress=False,
        spk_emb=spk_emb,
        silence_front=silence_front,
        diff_jump_silence_front=diff_jump_silence_front,
        threhold=-60
    )
    _audio = _audio.cpu().numpy()
    if raw_sample != _model_sr:
        tar_audio = librosa.resample(_audio, _model_sr, raw_sample)
    else:
        tar_audio = _audio
    tar_audio = tar_audio[:inlen]

    # 返回音频
    out_wav_path = io.BytesIO()
    sf.write(out_wav_path, tar_audio, raw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == "__main__":
    # 与冷月佬的GUI搭配使用，仓库地址:https://github.com/fishaudio/realtime-vc-gui
    # config和模型得同一目录。如果是组合模型(.ptc)则不用配置文件。
    checkpoint_path = "exp/vec51220/model_600000.pt"
    # f0提取器，有parselmouth, dio, harvest, crepe
    select_pitch_extractor = 'crepe'
    # f0范围限制(Hz)
    limit_f0_min = 50
    limit_f0_max = 1100
    # device
    device = 'cuda'
    # 扩散部分完全不合成安全区，打开可以减少硬件压力并加速，但是会损失合成效果
    diff_jump_silence_front = False
    # 如果需要使用naive模型进行浅扩散，在这里设置naive model的路径, 如果是组合模型(.ptc)就不用了
    naive_model_path = 'exp/naivetest/model_300000.pt'
    # 以下参数仅在使用speaker_encoder时生效
    spk_emb_path = None  # 非None导入声纹，会覆盖spk_id
    spk_emb_dict_path = None  # 非None导入声纹字典，会覆盖模型自带的
    # 加载svc模型，以下内容无需修改
    svc_model = DiffusionSVC(device=device)
    svc_model.load_model(checkpoint_path, select_pitch_extractor, limit_f0_min, limit_f0_max)
    if naive_model_path is not None:  # 加载naive模型
        svc_model.load_naive_model(naive_model_path=naive_model_path)
    if svc_model.args.model.use_speaker_encoder:  # 如果使用声纹，则处理声纹选项
        svc_model.set_spk_emb_dict(spk_emb_dict_path)
        spk_emb = svc_model.encode_spk_from_path(spk_emb_path)
    else:
        spk_emb = None
    # 此处与GUI对应，端口必须接上。
    app.run(port=6844, host="0.0.0.0", debug=False, threaded=False)
