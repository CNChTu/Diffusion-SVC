import os
import torch
import librosa
import argparse
import numpy as np
import soundfile as sf
from ast import literal_eval
from tools.infer_tools import DiffusionSVC
from text2semantic.utils import get_language_model
import yaml
from tools.tools import DotDict

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dm",
        "--diffusion_model",
        type=str,
        required=True,
        help="path to the diffusion model checkpoint",
    )
    parser.add_argument(
        "-lm",
        "--language_model",
        type=str,
        required=True,
        help="path to the language model checkpoint",
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
        help="text",
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
        "-speedup",
        "--speedup",
        type=str,
        required=False,
        default=10,
        help="speed up | default: 10",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='dpm-solver',
        help="ddim, pndm, dpm-solver or unipc | default: dpm-solver",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载扩散模型
    diffusion_svc = DiffusionSVC(device=device)  # 加载模型
    diffusion_svc.load_model(model_path=cmd.diffusion_model, f0_model="fcpe", f0_max=800, f0_min=65)
    # 加载语言模型
    config_file = os.path.join(os.path.split(cmd.language_model)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    lm = get_language_model(**args.model.text2semantic)
    lm.load_state_dict(torch.load(cmd.language_model, map_location=torch.device(device)))
    lm.eval()

    # 生成语音
    text = cmd.input
    spk_id = cmd.spk_id
    speedup = cmd.speedup
    method = cmd.method
    
