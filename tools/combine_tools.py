import os
import yaml
import torch
from diffusion.unit2mel import Unit2Mel, Unit2MelNaive, load_model_vocoder
from diffusion.vocoder import load_vocoder_for_save


class NaiveAndDiffModel:
    def __init__(self, diff_model_path, naive_model_path=None, vocoder_type=None, vocoder_path=None, device="cpu"):
        # diff
        diff_config_path = os.path.join(os.path.split(diff_model_path)[0], 'config.yaml')
        with open(diff_config_path, "r") as _f:
            self.diff_config_dict = yaml.safe_load(_f)
        _, _, self.diff_args = load_model_vocoder(diff_model_path, device=device)
        # check for combo trained model
        if self.diff_args.model.naive_fn is not None:
            self.is_combo_trained_model = True
            if str(naive_model_path).lower() != 'none':
                raise ValueError("This is a combo trained diff model, naive_model_path should be None")
        else:
            self.is_combo_trained_model = False

        # naive
        if not self.is_combo_trained_model:
            naive_config_path = os.path.join(os.path.split(naive_model_path)[0], 'config.yaml')
            with open(naive_config_path, "r") as _f:
                self.naive_config_dict = yaml.safe_load(_f)
            _, _, self.naive_args = load_model_vocoder(naive_model_path, device=device)
            _ = None

            # check config
            assert self.naive_args.model.type[:5] == 'Naive'
            assert self.diff_args.model.type[:9] == 'Diffusion'
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
        self.device = device

        # load ckpt
        self.diff_model = torch.load(diff_model_path, map_location=torch.device(device))
        if not self.is_combo_trained_model:
            self.naive_model = torch.load(naive_model_path, map_location=torch.device(device))
        print(" [INFO] Loaded model and config check out.")

        # vocoder
        if vocoder_path is not None:
            assert vocoder_type is not None, "If use vocoder_path, please set vocoder_type"
        if vocoder_type is not None:
            assert vocoder_path is not None, "If use vocoder_type, please set vocoder_path"
            self.vocoder = load_vocoder_for_save(vocoder_type=vocoder_type, model_path=vocoder_path, device=device)
            self.vocoder_type = vocoder_type
        else:
            self.vocoder = None

    def save_combo_model(self, save_path, save_name):
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, save_name + ".ptc")
        if self.is_combo_trained_model:
            save_dict = {
                "diff_model": self.diff_model,
                "diff_config_dict": self.diff_config_dict
            }
        else:
            save_dict = {
                "diff_model": self.diff_model,
                "diff_config_dict": self.diff_config_dict,
                "naive_model": self.naive_model,
                "naive_config_dict": self.naive_config_dict
            }
        if self.vocoder is not None:
            print(self.vocoder['model'])
            save_dict["vocoder"] = self.vocoder
            save_dict["vocoder_type"] = self.vocoder_type
        torch.save(save_dict, out_path)
        print(" [INFO] Combo model saved. Done.")
