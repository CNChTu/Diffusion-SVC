import os
import random
import re
import numpy as np
import librosa
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from train_log.utils import traverse_dir


def get_data_loaders(args, accelerate = None):
    data_train = TextDataset(
        path_root = args.data.train_path,
        use_cache = args.train.cache_all_data,
        accelerate=accelerate
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=args.train.num_workers if not args.train.cache_all_data else 1,
        persistent_workers= (args.train.num_workers > 0) if not args.train.cache_all_data else False,
        pin_memory=True if not args.train.cache_all_data else False,
        collate_fn=colle_fn
    )
    data_valid = TextDataset(
        path_root = args.data.valid_path,
        use_cache = args.train.cache_all_data,
        accelerate = None
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=colle_fn
    )
    return loader_train, loader_valid


class TextDataset(Dataset):
    def __init__(
            self,
            path_root,
            use_cache=True,
            extensions=['npy'],
            accelerate=None,
            model = None,
            n_spk = None
    ):
        super().__init__()

        self.path_root = path_root
        self.path_units_root = os.path.join(path_root, 'units')
        self.use_cache = use_cache
        self.n_spk = n_spk
        self.model = model

        self.paths = traverse_dir(
            self.path_units_root,
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )

        if accelerate is not None:
            self.paths = self.paths[accelerate.process_index::accelerate.num_processes]
        
        self.data_buffer = {}
        self.spk_name_id_map = {}

        if use_cache:
            print('Load all the data from :', path_root)
        else:
            print('Load file list :', path_root)
        self.spk_id = 1
        if use_cache:
            for name_ext in tqdm(self.paths, total=len(self.paths), position=accelerate.process_index if accelerate is not None else 0):
                try:
                    path_units = os.path.join(self.path_units_root, name_ext)
                    units = torch.tensor(np.load(path_units))

                    self.data_buffer[name_ext] = {
                        'units': units,
                        'mask': self.get_attention_mask(units.shape[-2]),
                        'name_ext':name_ext
                    }
                except Exception as e:
                    print(' [!] error :', name_ext)
                    self.paths.remove(name_ext)
                    continue

    def __getitem__(self, file_idx):
        try:
            name_ext = self.paths[file_idx]
            if self.use_cache:
                data_buffer = self.data_buffer[name_ext]
            else:
                path_units = os.path.join(self.path_units_root, name_ext)
                units = torch.tensor(np.load(path_units))

                data_buffer = {
                    'units': units,
                    'mask': self.get_attention_mask(units.shape[-2]),
                    'name_ext':name_ext
                }
                # get item
            return data_buffer
        except Exception as e:
            return self.__getitem__((file_idx+1)%len(self.paths))

    def get_attention_mask(self,length):
        attention_mask = torch.ones((length))
        return attention_mask        

    def __len__(self):
        return len(self.paths)


def colle_fn(batch):
    units = []
    name = []
    mask = []
    for batch_item in batch:
        units.append(batch_item['units'])
        name.append(batch_item['name_ext'])
        mask.append(batch_item['mask'])
    rtn = {
            'units': pad_sequence(units, batch_first=True, padding_value=0.),
            'mask': pad_sequence(mask, batch_first=True, padding_value=0),
            'name':name
    }
    return rtn


if __name__  == '__main__':
    from train_log import utils
    args = utils.load_config("configs/config.yaml")
    print(args)
    loader_train, loader_valid = get_data_loaders(args)
    for batch in loader_train:
        print(batch["units"].shape)