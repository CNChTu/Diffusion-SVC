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

def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args,model, accelerate = None):
    data_train = TextDataset(
        path_root = args.data.train_path,
        use_cache = args.model.text2semantic.train.cache_all_data,
        n_spk = args.model.text2semantic.model.n_spk,
        model = model,
        accelerate=accelerate
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.model.text2semantic.train.batch_size,
        shuffle=True,
        num_workers=args.model.text2semantic.train.num_workers if not args.model.text2semantic.train.cache_all_data else 1,
        persistent_workers= (args.model.text2semantic.train.num_workers > 0) if not args.model.text2semantic.train.cache_all_data else False,
        pin_memory=True if not args.model.text2semantic.train.cache_all_data else False,
        collate_fn=colle_fn
    )
    data_valid = TextDataset(
        path_root = args.data.valid_path,
        use_cache = args.model.text2semantic.train.cache_all_data,
        n_spk = args.model.text2semantic.model.n_spk,
        model = model,
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
        self.path_utt_root = os.path.join(path_root, 'utt')
        self.path_semantic_token_root = os.path.join(path_root, 'semantic_token')
        self.use_cache = use_cache
        self.n_spk = n_spk
        self.model = model

        self.paths = traverse_dir(
            self.path_utt_root,
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
                    path_utt = os.path.join(self.path_utt_root, name_ext)
                    path_semantic_token = os.path.join(self.path_semantic_token_root, name_ext)
                    
                    phones, tones, lang_ids, word2ph = np.load(path_utt, allow_pickle=True)
                    
                    if n_spk is not None and n_spk > 1:
                        dirname_split = os.path.dirname(name_ext)
                        if self.spk_name_id_map.get(dirname_split) is None:
                            self.spk_name_id_map[dirname_split] = self.spk_id
                            self.spk_id += 1
                        spk_id_seq = torch.LongTensor(np.ones_like(phones)) * self.spk_id
                        if self.spk_id < 1 or self.spk_id > n_spk:
                            raise ValueError(
                                ' [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk ')
                    else:
                        spk_id_seq = None

                    semantic_tokens = np.load(path_semantic_token)
                    semantic_tokens = np.concatenate([[self.model.semantic_bos_token_id],semantic_tokens,[self.model.semantic_eos_token_id]] ,axis=-1)

                    phones_length = len(phones)
                    semantic_length = len(semantic_tokens)

                    self.data_buffer[name_ext] = {
                        'phones': phones,
                        'tones': tones,
                        'lang_ids': lang_ids,
                        'word2ph': word2ph,
                        'semantic_tokens': semantic_tokens,
                        'phones_length': phones_length,
                        'semantic_length': semantic_length,
                        'spk_id':spk_id_seq,
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
                path_utt = os.path.join(self.path_utt_root, name_ext)
                path_semantic_token = os.path.join(self.path_semantic_token_root, name_ext)

                phones, tones, lang_ids, word2ph = np.load(path_utt, allow_pickle=True)

                if self.n_spk is not None and self.n_spk > 1:
                    dirname_split = os.path.dirname(name_ext)
                    if self.spk_name_id_map.get(dirname_split) is None:
                        self.spk_name_id_map[dirname_split] = self.spk_id
                        self.spk_id += 1
                        spk_id = self.spk_id
                    else:
                        spk_id = self.spk_name_id_map[dirname_split]
                    spk_id_seq = torch.LongTensor(np.ones_like(phones)) * spk_id
                else:
                    spk_id_seq = None    

                semantic_tokens = np.load(path_semantic_token)
                semantic_tokens = np.concatenate([[self.model.semantic_bos_token_id],semantic_tokens,[self.model.semantic_eos_token_id]] ,axis=-1)
                phones_length = len(phones)
                semantic_length = len(semantic_tokens)
                data_buffer = {
                    'phones': phones,
                    'tones': tones,
                    'lang_ids': lang_ids,
                    'word2ph': word2ph,
                    'semantic_tokens': semantic_tokens,
                    'phones_length': phones_length,
                    'semantic_length': semantic_length,
                    'spk_id':spk_id_seq,
                    'name_ext':name_ext
                }
                # get item
            return self.get_data(data_buffer)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.__getitem__((file_idx+1)%len(self.paths))

    def get_data(self, data_buffer):
        attention_mask = self.get_attention_mask(data_buffer['semantic_length'])
        encoder_attention_mask = self.get_attention_mask(data_buffer['phones_length'])
        
        rtn = {
            'phone': torch.LongTensor(data_buffer['phones'].astype(np.int64)),
            'tone': torch.LongTensor(data_buffer['tones'].astype(np.int64)),
            'semantic': torch.LongTensor(data_buffer['semantic_tokens'].astype(np.int64)),
            'labels': torch.LongTensor(data_buffer['semantic_tokens'].astype(np.int64)),
            'attention_mask': attention_mask,
            'encoder_attention_mask': encoder_attention_mask,
            'spk_id': data_buffer['spk_id'],
            'name':data_buffer['name_ext']
        }

        return rtn

    def get_attention_mask(self,length):
        attention_mask = torch.ones((length))
        return attention_mask        

    def __len__(self):
        return len(self.paths)


def colle_fn(batch):
    phone = []
    tone = []
    semantic = []
    labels = []
    attention_mask = []
    encoder_attention_mask = []
    spk_id_seq = []
    name = []
    for batch_item in batch:
        phone.append(batch_item['phone'])
        tone.append(batch_item['tone'])
        semantic.append(batch_item['semantic'])
        labels.append(batch_item['labels'])
        attention_mask.append(batch_item['attention_mask'])
        encoder_attention_mask.append(batch_item['encoder_attention_mask'])
        if batch_item['spk_id'] is not None:
            spk_id_seq.append(batch_item['spk_id'])
        else:
            spk_id_seq = None
        name.append(batch_item['name'])
    rtn = {
            'phone': pad_sequence(phone, batch_first=True, padding_value=-100),
            'tone': pad_sequence(tone, batch_first=True, padding_value=-100),
            'semantic': pad_sequence(semantic, batch_first=True, padding_value=-100),
            'labels': pad_sequence(labels, batch_first=True, padding_value=-100),
            'attention_mask': pad_sequence(attention_mask, batch_first=True, padding_value=0),
            'encoder_attention_mask': pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0),
            'spk_id': pad_sequence(spk_id_seq, batch_first=True, padding_value=0) if spk_id_seq is not None else None,
            'name':name
    }
    return rtn


if __name__  == '__main__':
    from logger import utils
    args = utils.load_config("configs/config.yaml")
    print(args)
    loader_train, loader_valid = get_data_loaders(args)
    for batch in loader_train:
        print(batch["phone"].shape)