import os
from typing import Any
import librosa
import torch
from torch.utils.data import Dataset
from train_log.utils import traverse_dir, filelist_path_to_file_list
import numpy as np

class Collate:
    def __init__(self, return_audio_list = False):
        self.return_audio_list = return_audio_list

    def __call__(self, batch):
        audios, names = zip(*batch)
        audio_lenth = [len(audio) for audio in audios]

        if not self.return_audio_list:
            audios = torch.nn.utils.rnn.pad_sequence(
                [torch.from_numpy(audio) for audio in audios],
                batch_first=True,
                padding_value=0.0
            )

        return audios, np.array(audio_lenth), names


def lenth_to_mask(lenth, max_lenth = None):
    if max_lenth is None:
        max_lenth = max(lenth)
    mask = torch.arange(max_lenth).to(lenth.device)
    mask = mask[None, :] < lenth[:, None]
    return mask

def get_data_loaders(args, accelerator=None, batch_size = 6, return_audio_list = None):
    
    if args.data.filelist_path is not None:
        file_list, root_path = filelist_path_to_file_list(os.path.join(args.data.train_path,"filelist.txt"))
    else:
        file_list = None
        root_path = None

    data_train = AudioDataset(
        args.data.train_path,
        sample_rate=args.data.sampling_rate,
        extensions=args.data.extensions,
        file_list=file_list,
        root_path=root_path,
        accelerator=accelerator
    )

    data_valid = AudioDataset(
        args.data.valid_path,
        sample_rate=args.data.sampling_rate,
        extensions=args.data.extensions,
        file_list=file_list,
        root_path=root_path,
        accelerator=accelerator
    )

    if return_audio_list is not None:
        return_audio_list = return_audio_list
    else:
        if args.data.encoder == "w2v-bert":
            return_audio_list = True
        else:
            return_audio_list = False

    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=args.train.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=Collate(return_audio_list = return_audio_list),
    )

    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=batch_size,
        num_workers=args.train.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=Collate(return_audio_list = return_audio_list),
    )
    
    return loader_train, loader_valid


class AudioDataset(Dataset):
    def __init__(
            self,
            path_root,
            sample_rate,
            extensions=['wav'],
            file_list=None,
            root_path=None,
            accelerator=None
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.path_root = path_root
        self.root_path = root_path

        if file_list is not None:
            self.paths = file_list
        else:
            self.paths = traverse_dir(
                os.path.join(path_root, 'audio'),
                extensions=extensions,
                is_pure=True,
                is_sort=True,
                is_ext=True
            )
        
        if accelerator is not None:
            self.paths = self.paths[accelerator.process_index::accelerator.num_processes]
        
    def __getitem__(self, file_idx):
        try:
            name_ext = self.paths[file_idx]
            return self.get_data(name_ext)
        except Exception as e:
            print('Error in loading data:', e)
            return self.__getitem__((file_idx + 1) % len(self.paths))

    def get_data(self, name_ext):
        if self.root_path is not None:
            path_audio = os.path.join(self.root_path, name_ext)
        else:
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
        
        audio, sr = librosa.load(
                path_audio, 
                sr = self.sample_rate)
        
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max() * 0.98

        return audio, name_ext

    def __len__(self):
        return len(self.paths)



if __name__ == '__main__':
    from train_log import utils
    import accelerate
    args = utils.load_config("configs/config.yaml")
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps = args.model.text2semantic.train.gradient_accumulation_steps
    )
    data_loader = get_data_loaders(args, accelerator=accelerator)
    for audio, audio_lenth, names in data_loader:
        print(len(audio))
        print(audio_lenth)
        print(names)
        break