import enum
import gradio as gr
import os
import pathlib
from loguru import logger
import yaml, json

from get_models import getModels
import inference
from typing import Sequence

def convert_keys_to_numbers(jsonstr):
    dictionary = json.loads(jsonstr)
    co_dictionary = {int(key): value for key, value in dictionary.items()}
    return ("{" + ", ".join(f"{key}: {value}" for key, value in co_dictionary.items()) + "}")


def get_max_num_file_path(folder_path: str | pathlib.Path):
    max_num = -1
    max_num_file_path = ""
    
    if not isinstance(folder_path, pathlib.Path):
        folder_path = pathlib.Path(folder_path)
    
    for file in folder_path.iterdir():
        if file.suffix == ".pt":
            num = int(file.stem.split("_")[1].split(".")[0])
            if num > max_num:
                max_num = num
                max_num_file_path = file
                
    return str(max_num_file_path)

@logger.catch
def load_model(model, f0, is_naive, naive, device):
    try:
        logger.trace("{} {} {} {}", model, f0, is_naive, naive)
        inference.load_model(get_max_num_file_path(model),f0, is_naive, get_max_num_file_path(naive),device)
        return "👌 模型加载成功"
    except Exception as e:
        logger.error(e)
        # logger.info(get_max_num_file_path(model))
        return "👎 模型加载失败"

def infer(dev,inp_f,speaker_method,speaker_id,speaker_mix,key,speedup,naive_method,k_step):
    if speaker_method == 0:
        speaker_id += 1
    print(dev,inp_f,"output.wav",1 if speaker_method == 1 else speaker_id,speaker_method,None if speaker_method == 0 else convert_keys_to_numbers(speaker_mix),0,key,-60,-40,5000,speedup,naive_method,k_step)
    inference.audio_processing(dev,inp_f,"output.wav",1 if speaker_method == 1 else speaker_id,None if speaker_method == 0 else convert_keys_to_numbers(speaker_mix),0,key,-60,-40,5000,speedup,naive_method,k_step)
    return "output.wav"

with gr.Blocks() as block: 
    gr.Markdown("# DiffusionSvc") 

    # gr.Dropdown([None], label="naive_model"),
    model = gr.Dropdown(getModels(pathlib.Path("./exp")), label="主模型选择", info="自动选择最新模型及配置")
    f0 = gr.Dropdown(
        ['parselmouth', 'dio', 'harvest', 'crepe'], 
        value="crepe", 
        label="F0 提取器", 
        info="harvest 低音强，crepe 哪都好，上述二位推理速度慢，特别是你 harvest"
    )
    
    '''
    naive start
    '''
    
    is_naive = gr.Checkbox(value=False, label="是否使用浅扩散模型 (不选后续浅扩散模型相关设置忽略)")
    naive = gr.Dropdown(
        getModels(pathlib.Path("./exp")), 
        label="主模型选择", 
        info="自动选择最新模型及配置",
        visible=is_naive.value
    )

    device = gr.Dropdown(["cpu", "cuda"], label="设备", value="cuda")

    def onChangeIsNaive(value: bool):
        return gr.update(visible=value)

    is_naive.change(fn=onChangeIsNaive, inputs=is_naive, outputs=naive)


    '''
    naive end
    '''

    speakers: list[int | str] = []

    btn = gr.Button(value="加载模型")

    def read_spk_map(path):
        file_path = path / "spk_map.json"

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
            return data
        else:
            return False

    '''
        切换选择模型
    '''
    def onChangeModel(value): 
        global speakers

        if not value:
            raise ValueError(f"What the f__k it is? {value}")
        config_path = pathlib.Path(value) / 'config.yaml'

        with config_path.open('r') as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
            speakers = tuple(range(1, config["model"]["n_spk"]+1))
        
            spk_map = read_spk_map(pathlib.Path(value))
            if spk_map:
                speakers = spk_map

        # col_cnt=(len(speakers), "fixed") 
            logger.info("Load speakers {}",[str(x) for x in speakers])
            return gr.Dropdown.update(choices=[str(x) for x in speakers],value="1" if not spk_map else speakers[0])
        spk_map

    
    
    output_html = gr.HTML()
    
    btn.click(load_model, inputs=[model, f0, is_naive, naive, device], outputs=[output_html])


    '''
        切换发声源
    '''
    def onchangeSpeakerMethod(value):
        if value == 0:
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    infer_inputs = {
        "inp_f": gr.Audio(label="输入文件",type="filepath"),
        "speaker_method": gr.Dropdown(choices=['单人/多人中一人','多人混合'], type="index", value='单人/多人中一人', label="发声人"),
        "speaker_id": gr.Dropdown(choices=['请先选择模型'], value='请先选择模型', label="说话人id", info="单人选择 1", type="index"),
        "speaker_mix": gr.Code(value='''
{
    "1": 0.5,
    "2": 0.5
}
                               ''',language="json",label="多说话人混合, json 格式, 键填 spkid", info="所有发声人的权重加起来必须是", visible=False),
        # {spkid: 百分比(0.1)} 
        # https://www.gradio.app/docs/dataframe
        "key": gr.Slider(value=0, label="升降调", minimum=-50, maximum=50, step=1),    
        "speedup": gr.Slider(value=10, label="推理加速", minimum=0, maximum=100, step=1),
        "naive_method?": gr.Dropdown(["pndm", "ddim", "unipc", "dpm-solver"], value="dpm-solver", label="浅扩散方法"),
        "k_step": gr.Slider(value=100, label="浅扩散步数", minimum=0, maximum=100, step=1)
    }
    model.change(fn=onChangeModel, inputs=[model], outputs=[infer_inputs['speaker_id']])

    output_audio = gr.Audio(label="输出")

    infer_inputs["speaker_method"].change(fn=onchangeSpeakerMethod, inputs=infer_inputs["speaker_method"], outputs=[infer_inputs["speaker_id"],infer_inputs["speaker_mix"]])

    infer_button = gr.Button("开始推理")

    infer_button.click(infer, inputs=list(infer_inputs.values()), outputs=output_audio)

if __name__ == "__main__":
    block.launch(debug=True)
