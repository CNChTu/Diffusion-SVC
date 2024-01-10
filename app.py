import gradio as gr
import librosa
from ast import literal_eval
import torch
from tools.infer_tools import DiffusionSVC
from openxlab.model import download
import openxlab
import os

ak = os.getenv(OPENXLAB_AK)
sk = os.getenv(OPENXLAB_SK)

openxlab.login(ak=ak, sk=sk)

download(model_repo='CNChiTu/ContentVec', 
        model_name='ContentVec', output='pretrain/contentvec')

download(model_repo='CNChiTu/DiffusionSVC', 
        model_name='DiffusionSVC_1.0_DEMO_Combo_Model_opencpop-kiritan.ptc')

download(model_repo='CNChiTu/TEST', 
        model_name='TEST')

download(model_repo='CNChiTu/TEST', 
        model_name='TEST_CONFIG')

G_MODEL_PATH = "opencpop-kiritan.ptc"
G_NAIVE_MODEL_PATH = None
G_OTHER_VOCODER_DICT = {'type': 'nsf-hifigan', 'path': 'opki_4000.ckpt'}
G_NUM_SPK = 2
G_SPKID_SPKNAME_DICT = {
    "OpenCPop": "1",
    "Kiritan": "2",
}
G_SPK_NAME_LIST = list(G_SPKID_SPKNAME_DICT.keys())
G_DEVICE = "cuda"
G_F0_MAX = 1200
G_F0_MIN = 40
G_SVC_MODEL = DiffusionSVC(device=G_DEVICE)

G_SVC_MODEL.load_model(
    model_path=G_MODEL_PATH,
    f0_model='rmvpe',
    f0_max=G_F0_MAX,
    f0_min=G_F0_MIN,
    other_vocoder_dict=G_OTHER_VOCODER_DICT
)

if G_NAIVE_MODEL_PATH is not None:
    G_SVC_MODEL.load_naive_model(naive_model_path=G_NAIVE_MODEL_PATH)

G_K_STEP_MAX = int(G_SVC_MODEL.args.model.k_step_max) if G_SVC_MODEL.args.model.k_step_max is not None else 1000


# 上传输入音频文件并读取
def upload_audio():
    return gr.components.Audio(label="Upload Audio", type="numpy")


# 选择说话人id，从1到NUM_SPK
def select_speaker():
    global G_SPK_NAME_LIST
    spk_name = gr.components.Dropdown(choices=G_SPK_NAME_LIST, label="Select Speaker")
    return spk_name


# 选择是否混合启用说话人混合
def select_is_mix_speaker():
    return gr.components.Checkbox(label="Mix Speaker")


# 输入混合说话人字典
def input_mix_speaker_dict():
    return gr.components.Textbox(label="Mix Speaker Dict, like 1:0.5, 2:0.5")


# 选择变调程度，从-24到24
def select_pitch_shift():
    return gr.components.Slider(minimum=-24, maximum=24, step=0.5, label="Select Pitch Shift")


# 选择性别参数，从-5到5
def select_formant_shift():
    return gr.components.Slider(minimum=-5, maximum=5, step=1, label="Select Formant Shift")


# 选择扩散深度，从1到K_STEP_MAX
def select_k_step():
    global G_K_STEP_MAX
    return gr.components.Slider(minimum=1, maximum=G_K_STEP_MAX, step=1, label="Select K Step")


# 选择扩散加速，从1到K_STEP_MAX//2
def select_infer_speedup():
    global G_K_STEP_MAX
    return gr.components.Slider(minimum=1, maximum=G_K_STEP_MAX // 2, step=1, label="Select Infer Speedup")


# 选择扩散方法
def select_method():
    return gr.components.Dropdown(["unipc", "ddim", 'pndm', 'dpm-solver'], label="Select Method")


# 转换音频
def convert_audio(
        audio,
        spkid_str,
        is_mix_spk,
        mix_spk_dict_str,
        key,
        formant_shift,
        k_step,
        infer_speedup,
        method
):
    sr = audio[0]
    audio = audio[1]

    # 处理int场合，如果数组是int
    if audio.dtype == "int16":
        audio = audio.astype("float32")
        audio /= 32768.0

    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    global G_SPKID_SPKNAME_DICT
    spkid_str = G_SPKID_SPKNAME_DICT[str(spkid_str)]
    spkid = int(spkid_str)

    if is_mix_spk:
        spk_mix_dict = literal_eval(mix_spk_dict_str)
    else:
        spk_mix_dict = None

    key = float(key)
    formant_shift = int(formant_shift)

    if int(k_step) == 1000:
        k_step = None
    else:
        k_step = int(k_step)

    infer_speedup = int(infer_speedup)

    method = method

    global G_SVC_MODEL
    out_wav, out_sr = G_SVC_MODEL.infer_from_long_audio(
        audio,
        sr=sr,
        key=float(key),
        spk_id=spkid,
        spk_mix_dict=spk_mix_dict,
        aug_shift=formant_shift,
        infer_speedup=infer_speedup,
        method=method,
        k_step=k_step,
        use_tqdm=True,
    )

    audio = out_sr, out_wav

    # 释放显存，但保留模型
    torch.cuda.empty_cache()

    return audio


app = gr.Interface(
    fn=convert_audio,
    inputs=[
        upload_audio(),
        select_speaker(),
        select_is_mix_speaker(),
        input_mix_speaker_dict(),
        select_pitch_shift(),
        select_formant_shift(),
        select_k_step(),
        select_infer_speedup(),
        select_method(),
    ],
    outputs="audio",
    title="Diffusion SVC",
    description="Diffusion SVC",
)

if __name__ == "__main__":
    app.launch()
