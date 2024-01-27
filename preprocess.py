import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from train_log import utils
from tqdm import tqdm
from tools.tools import F0_Extractor, Volume_Extractor, Units_Encoder, SpeakerEncoder
from diffusion.vocoder import Vocoder
from train_log.utils import traverse_dir, filelist_path_to_file_list
from text.cleaner import text_to_sequence
import torch.multiprocessing as mp
import traceback

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
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=0,
        required=False,
        help="proceeding workers, 0 if not set")
    parser.add_argument(
        '-skf0', 
        '--skip_f0',
        action='store_true',
        default=False,
        help='skip f0 extract')
    parser.add_argument(
        '-skac', 
        '--skip_acoustic',
        action='store_true',
        default=False,
        help='skip acoustic(like mel) extract')
    parser.add_argument(
        '-skse', 
        '--skip_semantic',
        action='store_true',
        default=False,
        help='skip semantic(like mel) extract')
    
    return parser.parse_args(args=args, namespace=namespace)

def preprocess_worker(rank, path, args, skip_flag, sample_rate, hop_size,
               device='cuda', use_pitch_aug=None, extensions=['wav'], is_tts = False, filelist=None, root_path=None, num_workers = 0):
    if device == 'cuda':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)

    path_srcdir = os.path.join(path, 'audio')
    if filelist is None:
        filelist = traverse_dir(
            path_srcdir,
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True)
    else:
        filelist = filelist

    if num_workers != 0:
        filelist = filelist[rank::num_workers]
    skip_f0, skip_acoustic, skip_semantic = skip_flag
    # initialize f0 extractor
    if not skip_f0:
        f0_extractor = F0_Extractor(
            f0_extractor=args['data']['f0_extractor'],
            sample_rate=44100,
            hop_size=512,
            f0_min=args['data']['f0_min'],
            f0_max=args['data']['f0_max'],
            block_size=args['data']['block_size'],
            model_sampling_rate=args['data']['sampling_rate']
        )
    else:
        f0_extractor = None
        print('Skip f0 extraction!')

    # initialize volume extractor
    volume_extractor = Volume_Extractor(
            hop_size=512,
            block_size=args['data']['block_size'],
            model_sampling_rate=args['data']['sampling_rate']
        )

    # initialize mel extractor
    mel_extractor = None
    use_pitch_aug = use_pitch_aug
    if not skip_acoustic:
        mel_extractor = Vocoder(args['vocoder']['type'], args['vocoder']['ckpt'], device=device)
        if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
            mel_extractor = None
            print('Unmatch vocoder parameters, mel extraction is ignored!')
        elif use_pitch_aug is None:
            use_pitch_aug = args['model']['use_pitch_aug']
    else:
        mel_extractor = None
        print('Skip acoustic extraction!')
        
    # initialize units encoder
    if args['data']['encoder'] == 'cnhubertsoftfish':
        cnhubertsoft_gate = args['data']['cnhubertsoft_gate']
    else:
        cnhubertsoft_gate = 10
    if not skip_semantic:
        units_encoder = Units_Encoder(
            args['data']['encoder'],
            args['data']['encoder_ckpt'],
            args['data']['encoder_sample_rate'],
            args['data']['encoder_hop_size'],
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=device,
            units_forced_mode=args['data']['units_forced_mode']
        )
    else:
        units_encoder = None
        print('Skip semantic extraction!')

    preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate,
            hop_size, device=device, use_pitch_aug=use_pitch_aug,
            extensions=extensions,is_tts = is_tts,
            text2semantic_mode=args["model"]["text2semantic"]["mode"],
            filelist=filelist,root_path=root_path,
            force_units_interpolation=args["data"]["force_units_interpolation"],
            source_encoder_hop_size=args["data"]["source_encoder_hop_size"],
            target_encoder_hop_size=args["data"]["encoder_hop_size"],
            rank = rank)



def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size,
                device='cuda', use_pitch_aug=False, extensions=['wav'], is_tts = False, text2semantic_mode = "phone",
                filelist=None, root_path=None,
                force_units_interpolation=False,
                source_encoder_hop_size=320,
                target_encoder_hop_size=320,
                rank = 0):
    path_srcdir = os.path.join(path, 'audio')
    path_unitsdir = os.path.join(path, 'units')
    path_f0dir = os.path.join(path, 'f0')
    path_volumedir = os.path.join(path, 'volume')
    path_augvoldir = os.path.join(path, 'aug_vol')
    path_meldir = os.path.join(path, 'mel')
    path_augmeldir = os.path.join(path, 'aug_mel')
    path_skipdir = os.path.join(path, 'skip')
    if is_tts:
        path_uttdir = os.path.join(path, 'utt')
        utt_text = {}
        
    spk_set = set()
    if filelist is None:
        # list files
        filelist = traverse_dir(
            path_srcdir,
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True)
    else:
        filelist = filelist
        
    if root_path is not None:
        path_srcdir = root_path

    # run  
    def process(file):
        binfile = file + '.npy'
        spk_name = os.path.dirname(file)
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        if is_tts:
            if spk_name not in spk_set:
                print(f" [INFO] Loading utt_text from {spk_name}")
                path_uttfile = os.path.join(path_srcdir, file)
                path_uttfile = os.path.dirname(path_uttfile)
                path_uttfile = os.path.join(path_uttfile,"utt_text.txt")
                with open(path_uttfile,"r",encoding="UTF8") as f:
                    for f_i in f.readlines():
                        k, v = f_i.replace("\n","").split("|")
                        utt_text[k] = v
                spk_set.add(spk_name)
            path_uttfile = os.path.join(path_uttdir, binfile)
        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        # extract volume
        volume = volume_extractor.extract(audio, sr=sample_rate)
        
        if is_tts:
            # 得到文件名
            file_name = os.path.split(file)[-1]
            # 得到文本
            text = utt_text[file_name]
            if text2semantic_mode == "phone":
            # 文本转换为音素
                (phones, tones, lang_ids), (norm_text, word2ph) = text_to_sequence(text, "ZH")
                # 保存音素
            elif text2semantic_mode == "text":
                from transformers import BertTokenizer
                from text.multi_language_bert import get_bert_token
                tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir="./pretrain")
                tones = []
                lang_ids = []
                word2ph = []
                phones, norm_text = get_bert_token(text, tokenizer)
            os.makedirs(os.path.dirname(path_uttfile), exist_ok=True)
            np.save(path_uttfile, np.array((np.array(phones), np.array(tones), np.array(lang_ids), np.array(word2ph)),dtype=object), allow_pickle=True)
            
        # extract mel and volume augmentaion
        if mel_extractor is not None:
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()

            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0

            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift=keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio * (10 ** log10_vol_shift), sr=sample_rate)

        # units encode
        if units_encoder is not None:
            units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
            if force_units_interpolation:
                units_t = torch.nn.functional.interpolate(units_t.transpose(-1,-2), scale_factor=target_encoder_hop_size/source_encoder_hop_size, mode='linear', align_corners=False).transpose(-1,-2)
            units = units_t.squeeze().to('cpu').numpy()
        
        # extract f0
        if f0_extractor is not None:
            f0 = f0_extractor.extract(audio, uv_interp=False, sr=sample_rate)

        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # save npy     
            if units_encoder is not None:
                os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
                np.save(path_unitsfile, units)
            if f0_extractor is not None:
                os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
                np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
            np.save(path_volumefile, volume)
            if mel_extractor is not None:
                os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
                np.save(path_melfile, mel)
                os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
                np.save(path_augmelfile, aug_mel)
                os.makedirs(os.path.dirname(path_augvolfile), exist_ok=True)
                np.save(path_augvolfile, np.asarray((aug_vol, keyshift), dtype = object) , allow_pickle=True)
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))
            print('This file has been moved to ' + path_skipfile)

    print('Preprocess the audio clips in :', path_srcdir)

    # process
    for file in tqdm(filelist, total=len(filelist), position=rank):
        try:
            process(file)
        except Exception as e:
            traceback.print_exc()

if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    device = cmd.device
    num_workers = cmd.num_workers
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    skip_flag = [cmd.skip_f0, cmd.skip_acoustic, cmd.skip_semantic]
    # load config
    args = utils.load_config(cmd.config)
    is_tts = args.model.is_tts
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    train_set_rate = args.data.train_set_rate
    extensions = args.data.extensions

    if args.data.filelist_path is not None:
        file_list, root_path = filelist_path_to_file_list(args.data.filelist_path)
        random.shuffle(file_list)
        split_len = int(len(file_list)*train_set_rate)
        train_file_list = file_list[:split_len]
        valid_file_list = file_list[split_len:]
        print(f" [INFO] Train set: {len(train_file_list)} files, Valid set: {len(valid_file_list)} files")
        with open(os.path.join(args.data.train_path, "filelist.txt"),"w") as f:
            f.write(f"{root_path}\n")
            f.write("\n".join(train_file_list))
        with open(os.path.join(args.data.valid_path, "filelist.txt"),"w") as f:
            f.write(f"{root_path}\n")
            f.write("\n".join(valid_file_list))
    else:
        train_file_list, valid_file_list, root_path = None, None, None
    if cmd.num_workers == 0:
        # preprocess training set
        preprocess_worker(0, args.data.train_path, dict(args), skip_flag, sample_rate, hop_size,
               device=device, use_pitch_aug=None, extensions=extensions, is_tts = is_tts, filelist=train_file_list,root_path=root_path)
        # preprocess validation set
        preprocess_worker(0, args.data.valid_path, dict(args), skip_flag, sample_rate, hop_size,
               device=device, use_pitch_aug=False, extensions=extensions, is_tts = is_tts, filelist=valid_file_list,root_path=root_path)
    else:
        # preprocess training set
        mp.spawn(preprocess_worker, args=(args.data.train_path, dict(args), skip_flag, sample_rate, hop_size,
               device, None, extensions, is_tts, args.model.text2semantic.mode, train_file_list, root_path, num_workers), nprocs=num_workers)
        # preprocess validation set
        mp.spawn(preprocess_worker, args=(args.data.valid_path, dict(args), skip_flag, sample_rate, hop_size,
                device, False, extensions, is_tts, args.model.text2semantic.mode, valid_file_list, root_path, num_workers), nprocs=num_workers)