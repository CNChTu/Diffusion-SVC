import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from logger import utils
from tqdm import tqdm
from tools.tools import F0_Extractor, Volume_Extractor, Units_Encoder, SpeakerEncoder
from diffusion.vocoder import Vocoder
from logger.utils import traverse_dir
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
    return parser.parse_args(args=args, namespace=namespace)

def preprocess_worker(rank, path, args, sample_rate, hop_size,
               device='cuda', use_pitch_aug=None, extensions=['wav'], is_tts = False, text2semantic_mode = "phone",num_workers = 0):
    if device == 'cuda':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)

    path_srcdir = os.path.join(path, 'audio')
    filelist = traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)
    if num_workers != 0:
        filelist = filelist[rank::num_workers]

    # initialize f0 extractor
    f0_extractor = F0_Extractor(
        f0_extractor=args['data']['f0_extractor'],
        sample_rate=44100,
        hop_size=512,
        f0_min=args['data']['f0_min'],
        f0_max=args['data']['f0_max'],
        block_size=args['data']['block_size'],
        model_sampling_rate=args['data']['sampling_rate']
    )

    # initialize volume extractor
    volume_extractor = Volume_Extractor(
            hop_size=512,
            block_size=args['data']['block_size'],
            model_sampling_rate=args['data']['sampling_rate']
        )

    # initialize mel extractor
    mel_extractor = None
    use_pitch_aug = use_pitch_aug
    mel_extractor = Vocoder(args['vocoder']['type'], args['vocoder']['ckpt'], device=device)
    if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
        mel_extractor = None
        print('Unmatch vocoder parameters, mel extraction is ignored!')
    elif use_pitch_aug is None:
        use_pitch_aug = args['model']['use_pitch_aug']
    
    # initialize units encoder
    if args['data']['encoder'] == 'cnhubertsoftfish':
        cnhubertsoft_gate = args['data']['cnhubertsoft_gate']
    else:
        cnhubertsoft_gate = 10
    units_encoder = Units_Encoder(
        args['data']['encoder'],
        args['data']['encoder_ckpt'],
        args['data']['encoder_sample_rate'],
        args['data']['encoder_hop_size'],
        cnhubertsoft_gate=cnhubertsoft_gate,
        device=device,
        units_forced_mode=args['data']['units_forced_mode']
    )

    preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate,
            hop_size, device=device, use_pitch_aug=use_pitch_aug, extensions=extensions,is_tts = is_tts, text2semantic_mode=args["model"]["text2semantic"]["mode"], filelist=filelist, rank = rank)



def preprocess(path, f0_extractor, volume_extractor, mel_extractor, units_encoder, sample_rate, hop_size,
               device='cuda', use_pitch_aug=False, extensions=['wav'], is_tts = False, text2semantic_mode = "phone", filelist=None, rank = 0):
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

    # run  
    def process(file):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)
        path_augvolfile = os.path.join(path_augvoldir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        if is_tts:
            if len(utt_text) == 0:
                path_uttfile = os.path.join(path_srcdir, file)
                path_uttfile = os.path.dirname(path_uttfile)
                path_uttfile = os.path.join(path_uttfile,"utt_text.txt")
                with open(path_uttfile,"r",encoding="UTF8") as f:
                    for f_i in f.readlines():
                        k, v = f_i.replace("\n","").split("|")
                        utt_text[k] = v
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
        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()
        
        # extract f0
        f0 = f0_extractor.extract(audio, uv_interp=False, sr=sample_rate)

        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # save npy     
            os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
            np.save(path_unitsfile, units)
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

    # load config
    args = utils.load_config(cmd.config)
    is_tts = args.model.is_tts
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size

    extensions = args.data.extensions

    if cmd.num_workers == 0:
        # preprocess training set
        preprocess_worker(0, args.data.train_path, dict(args), sample_rate, hop_size,
               device=device, use_pitch_aug=None, extensions=extensions, is_tts = is_tts, text2semantic_mode=args.model.text2semantic.mode)
        # preprocess validation set
        preprocess_worker(0, args.data.valid_path, dict(args), sample_rate, hop_size,
               device=device, use_pitch_aug=False, extensions=extensions, is_tts = is_tts, text2semantic_mode=args.model.text2semantic.mode)
    else:
        # preprocess training set
        mp.spawn(preprocess_worker, args=(args.data.train_path, dict(args), sample_rate, hop_size,
               device, None, extensions, is_tts, args.model.text2semantic.mode, num_workers), nprocs=num_workers)
        # preprocess validation set
        mp.spawn(preprocess_worker, args=(args.data.valid_path, dict(args), sample_rate, hop_size,
                device, False, extensions, is_tts, args.model.text2semantic.mode, num_workers), nprocs=num_workers)