import os
import argparse
import torch
from train_log import utils
from batch_proccessor.dataloader import get_data_loaders
from diffusion.vocoder import Vocoder
import accelerate
import itertools
from tools.tools import StepLRWithWarmUp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=6,
        help="batch size")
    
    return parser.parse_args(args=args, namespace=namespace)


def save_acoutstic(acoustic, mel_lenth, path_meldir, name):
    acoustic = acoustic[..., :int(mel_lenth), :]
    path_melfile = os.path.join(path_meldir, name + ".npy")
    os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
    if isinstance(acoustic, torch.Tensor):
        acoustic = acoustic.cpu().numpy()
    np.save(path_melfile, acoustic)

if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # load config
    args = utils.load_config(cmd.config)
    if accelerator.is_main_process:
        print(' > config:', cmd.config)

    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)

    # datas
    loader_train, loader_valid = get_data_loaders(args, batch_size=cmd.batch_size, accelerator=accelerator, return_audio_list=False)

    vocoder = accelerator.prepare(
        vocoder
    )
    
    train_path_meldir = os.path.join(args.data.train_path, 'mel')

    # path_melfile = os.path.join(path_meldir, binfile)
    # os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
    # np.save(path_melfile, mel)

    for audios, audio_lenth, names in tqdm(loader_train):
        audios = audios.to(device)
        acoustics = vocoder.extract(audios, int(args.data.sampling_rate), only_mean=args.vocoder.only_mean)
        ac_len = np.ceil(audio_lenth / vocoder.vocoder_hop_size)
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(save_acoutstic, acoustics, ac_len, itertools.repeat(train_path_meldir), names)
    
    valid_path_meldir = os.path.join(args.data.valid_path, 'mel')

    for audios, audio_lenth, names in tqdm(loader_valid):
        audios = audios.to(device)
        acoustics = vocoder.extract(audios, int(args.data.sampling_rate), only_mean=args.vocoder.only_mean)
        ac_len = np.ceil(audio_lenth / vocoder.vocoder_hop_size)
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(save_acoutstic, acoustics, ac_len, itertools.repeat(valid_path_meldir), names)