import os
import torch
import argparse
from tools.infer_tools import DiffusionSVC
from text2semantic.utils import get_language_model
import yaml
from tools.tools import DotDict
from text.cleaner import text_to_sequence
from cluster import get_cluster_model, get_cluster_result
import soundfile as sf
import numpy as np
from tools.tools import units_forced_alignment
import torchaudio
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
    parser.add_argument(
        "-tk",
        "--topk",
        type=int,
        required=False,
        default=0,
        help="topk",
    )
    parser.add_argument(
        "-tp",
        "--topp",
        type=float,
        required=False,
        default=0.7,
        help="topp",
    )
    parser.add_argument(
        "-rp",
        "--repetition_penalty",
        type=float,
        required=False,
        default=1.1,
        help="repetition_penalty",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        required=False,
        default=0.7,
        help="temperature",
    )
    parser.add_argument(
        "-cfg",
        "--cfg_sclae",
        type=float,
        required=False,
        default=1.0,
        help="temperature",
    )
    parser.add_argument(
        "-ref",
        "--audio_ref",
        type=str,
        required=False,
        default=None,
        help="audio_reference_path | default: None",
    )
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    with torch.no_grad():
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

        if args.train.units_quantize_type == "kmeans":
            codebook = get_cluster_model(args.model.text2semantic.codebook_path)
            codebook = codebook.__dict__["cluster_centers_"]
            
            semantic_embedding = torch.nn.Embedding(
                codebook.shape[0],
                codebook.shape[1],
                _freeze = True
                )
            semantic_embedding.weight.data = torch.from_numpy(codebook)
            semantic_embedding.to(device)
        elif args.train.units_quantize_type == "vq":
            from vector_quantize_pytorch import VectorQuantize
            semantic_embedding = VectorQuantize(
                    dim = args.data.encoder_out_channels,
                    codebook_size = args.model.text2semantic.semantic_kmeans_num,
                    decay = 0.8,             
                    commitment_weight = 1.,
                    freeze_codebook=True,
                    use_cosine_sim=True,
                    codebook_dim = 32,
                )
            model_para = torch.load(args.model.text2semantic.codebook_path)
            semantic_embedding.load_state_dict(model_para["model"])
            semantic_embedding = semantic_embedding.to(device)
        elif args.train.units_quantize_type == "vqae":
            from vq_ae import get_model
            semantic_embedding = get_model(args)
            semantic_embedding.load_state_dict(torch.load(args.model.text2semantic.codebook_path)["model"])
            semantic_embedding.set_eval_mode()
            semantic_embedding = semantic_embedding.to(device)
        else:
            raise ValueError(' [x] Unknown quantize_type: ' + args.train.units_quantize_type)
        
        args.model.text2semantic.model.gradient_checkpointing = False
        lm = get_language_model(**args.model.text2semantic).to(device)
        lm.load_state_dict(torch.load(cmd.language_model, map_location=torch.device(device))["model"])
        lm.eval()

        # 生成语音
        text = cmd.input
        spk_id = cmd.spk_id
        speedup = cmd.speedup
        method = cmd.method
        
        top_k = cmd.topk
        top_p = cmd.topp
        repetition_penalty = cmd.repetition_penalty
        temperature = cmd.temperature
        cfg_sclae = cmd.cfg_sclae
        audio_ref_path = cmd.audio_ref
        
        if cmd.audio_ref is not None:
            audio, sr = torchaudio.load(audio_ref_path)
            audio = audio.to(device)
            units = diffusion_svc.encode_units(audio, sr)[None,...]
            if args.train.units_quantize_type == "kmeans":
                prefix = get_cluster_result(semantic_embedding, units.cpu().numpy()[0])
                prefix = torch.from_numpy(prefix)[None, ...].long().to(device)
            elif args.train.units_quantize_type == "vq" or args.train.units_quantize_type == "vqae":
                _, prefix, _ = semantic_embedding(units)
        else:
            prefix = None

        (phones, tones, lang_ids), (norm_text, word2ph) = text_to_sequence(text, 'ZH')
        
        phones, tones = torch.from_numpy(np.array(phones)).unsqueeze(0).long().to(device), torch.from_numpy(np.array(tones)).long().unsqueeze(0).to(device)
        
        spk_id_seq = torch.ones_like(phones) * spk_id
        semantic_token = lm.generate(phones,
                            tones,
                            prefix=prefix,
                            attention_mask=None,
                            use_cache=True,
                            max_length=1024,
                            do_sample=True,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            num_beams=1,
                            no_repeat_ngram_size = 0,
                            early_stopping = True,
                            spk_id = spk_id_seq,
                            end_gate_threshold = None,
                            cfg_scale = cfg_sclae
                            )

        if semantic_token[:,-1] == lm.semantic_eos_token_id:
            semantic_token = semantic_token[:,1:-1]
        else:
            semantic_token = semantic_token[:,1:]

        if args.train.units_quantize_type == "kmeans":
            semantic_emb = semantic_embedding(semantic_token)
        elif args.train.units_quantize_type == "vq":
            semantic_emb = semantic_embedding.project_out(semantic_embedding.get_codes_from_indices(semantic_token[:,:,None]))
        elif args.train.units_quantize_type == "vqae":
            semantic_emb = semantic_embedding.quantizer.project_out(semantic_embedding.quantizer.get_codes_from_indices(semantic_token[:,:,None]))

        semantic_emb = units_forced_alignment(semantic_emb, scale_factor=(diffusion_svc.args.data.sampling_rate/diffusion_svc.args.data.block_size)/(diffusion_svc.args.data.encoder_sample_rate/args.data.encoder_hop_size),units_forced_mode=diffusion_svc.args.data.units_forced_mode)

        wav = diffusion_svc.infer(semantic_emb,f0=None,volume=None, spk_id = spk_id, infer_speedup=speedup, method=method)
        
        sf.write(cmd.output, wav.detach().cpu().numpy()[0,0], diffusion_svc.args.data.sampling_rate)