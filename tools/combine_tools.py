import os
import yaml
import torch
from diffusion.unit2mel import Unit2Mel, Unit2MelNaive, load_model_vocoder
from loguru import logger

class NaiveAndDiffModel:
    def __init__(self, diff_model_path, naive_model_path, device="cpu"):
        # diff
        diff_config_path = os.path.join(os.path.split(diff_model_path)[0], 'config.yaml')
        with open(diff_config_path, "r") as _f:
            self.diff_config_dict = yaml.safe_load(_f)
        _, _, self.diff_args = load_model_vocoder(diff_model_path, device=device)
        # naive
        naive_config_path = os.path.join(os.path.split(naive_model_path)[0], 'config.yaml')
        with open(naive_config_path, "r") as _f:
            self.naive_config_dict = yaml.safe_load(_f)
        _, _, self.naive_args = load_model_vocoder(naive_model_path, device=device)
        _ = None
        self.device = device
        # check config
        assert self.naive_args.model.type == 'Naive'
        assert self.diff_args.model.type == 'Diffusion'
        if self.naive_args.data.encoder != self.diff_args.data.encoder:
            raise ValueError("encoder of Naive Model and Diffusion Model are different")
        if self.naive_args.model.n_spk != self.diff_args.model.n_spk:
            raise ValueError("n_spk of Naive Model and Diffusion Model are different")
        if bool(self.naive_args.model.use_speaker_encoder) != bool(self.diff_args.model.use_speaker_encoder):
            raise ValueError("use_speaker_encoder of Naive Model and Diffusion Model are different")
        if self.naive_args.vocoder.type != self.diff_args.vocoder.type:
            raise ValueError("vocoder of Naive Model and Diffusion Model are different")
        if self.naive_args.data.block_size != self.diff_args.data.block_size:
            raise ValueError("block_size of Naive Model and Diffusion Model are different")
        if self.naive_args.data.sampling_rate != self.diff_args.data.sampling_rate:
            raise ValueError("sampling_rate of Naive Model and Diffusion Model are different")
        # load ckpt
        self.diff_model = torch.load(diff_model_path, map_location=torch.device(device))
        self.naive_model = torch.load(naive_model_path, map_location=torch.device(device))
        logger.info("Loaded model and config check out.")

    def save_combo_model(self, save_path, save_name):
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, save_name + ".ptc")
        save_dict = {
            "diff_model": self.diff_model,
            "diff_config_dict": self.diff_config_dict,
            "naive_model": self.naive_model,
            "naive_config_dict": self.naive_config_dict
        }
        torch.save(save_dict, out_path)
        logger.info("Combo model saved. Done.")
