import os
import argparse
import torch
from train_log import utils
from batch_proccessor.dataloader import get_data_loaders, lenth_to_mask
from tools.tools import F0_Extractor, Units_Encoder
import accelerate
import itertools
from tools.tools import StepLRWithWarmUp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from torchaudio.transforms import Resample
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


def save_semantic(acoustic, mel_lenth, path_meldir, name):
    acoustic = acoustic[:int(mel_lenth), :]
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
    
    # load units_encoder
    units_encoder = Units_Encoder(
            args['data']['encoder'],
            args['data']['encoder_ckpt'],
            args['data']['encoder_sample_rate'],
            args['data']['encoder_hop_size'],
            cnhubertsoft_gate=args.data.cnhubertsoft_gate,
            device=device,
            units_forced_mode=args['data']['units_forced_mode']
        )

    is_resample = args['data']['sampling_rate'] != args['data']['encoder_sample_rate']

    if is_resample:
        resample_kernel = Resample(args['data']['sampling_rate'], args['data']['encoder_sample_rate']).to(device)
        resample_scale_factor = args['data']['encoder_sample_rate'] / args['data']['sampling_rate']

    # datas
    loader_train, loader_valid = get_data_loaders(args, batch_size=cmd.batch_size, accelerator=accelerator)

    units_encoder = accelerator.prepare(
        units_encoder
    )
    
    train_path_meldir = os.path.join(args.data.train_path, 'units')

    # path_melfile = os.path.join(path_meldir, binfile)
    # os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
    # np.save(path_melfile, mel)

    for audios, audio_lenth, names in tqdm(loader_train):
        audio_lenth = torch.from_numpy(audio_lenth).to(device)
        if is_resample:
            if isinstance(audios, torch.Tensor):
                audios = audios.to(device)
                audios = resample_kernel(audios)
            elif isinstance(audios, list):
                for i in range(len(audios)):
                    audios[i] = resample_kernel(torch.from_numpy(audios[i]).to(device)).cpu().numpy()
            audio_lenth = audio_lenth * resample_scale_factor
        padding_mask = lenth_to_mask(audio_lenth)
        semantic = units_encoder.encode(audios, int(args.data.encoder_sample_rate), args.data.encoder_hop_size, padding_mask)
        
        if args.data.force_units_interpolation:
            units_t = torch.nn.functional.interpolate(units_t.transpose(-1,-2), scale_factor=args.data.encoder_hop_size/args.data.source_encoder_hop_size, mode='linear', align_corners=False).transpose(-1,-2)
        
        audio_lenth = audio_lenth.cpu().numpy()
        ac_len = np.ceil(audio_lenth / args.data.encoder_hop_size)
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(save_semantic, semantic, ac_len, itertools.repeat(train_path_meldir), names)
    
    valid_path_meldir = os.path.join(args.data.valid_path, 'units')

    for audios, audio_lenth, names in tqdm(loader_valid):
        audio_lenth = torch.from_numpy(audio_lenth).to(device)
        if is_resample:
            if isinstance(audios, torch.Tensor):
                audios = audios.to(device)
                audios = resample_kernel(audios)
            elif isinstance(audios, list):
                for i in range(len(audios)):
                    audios[i] = resample_kernel(torch.from_numpy(audios[i]).to(device)).cpu().numpy()
            audio_lenth = audio_lenth * resample_scale_factor
        padding_mask = lenth_to_mask(audio_lenth)
        semantic = units_encoder.encode(audios, int(args.data.encoder_sample_rate), args.data.encoder_hop_size, padding_mask)
        
        if args.data.force_units_interpolation:
            units_t = torch.nn.functional.interpolate(units_t.transpose(-1,-2), scale_factor=args.data.encoder_hop_size/args.data.source_encoder_hop_size, mode='linear', align_corners=False).transpose(-1,-2)
        
        audio_lenth = audio_lenth.cpu().numpy()
        ac_len = np.ceil(audio_lenth / args.data.encoder_hop_size)
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(save_semantic, semantic, ac_len, itertools.repeat(train_path_meldir), names)
    