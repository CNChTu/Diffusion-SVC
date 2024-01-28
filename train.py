import os
import argparse
import torch
from torch.optim import lr_scheduler
from train_log import utils
from diffusion.data_loaders import get_data_loaders
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel, Unit2MelNaive
from diffusion.vocoder import Vocoder
import accelerate
import itertools
from tools.tools import StepLRWithWarmUp

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    # load config
    args = utils.load_config(cmd.config)
    if accelerator.is_main_process:
        print(' > config:', cmd.config)
        print(' >    exp:', args.env.expdir)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    if args.model.type == 'Diffusion':
        spec_norm = False if args.vocoder.type == "hifi-vaegan" else True

        model = Unit2Mel(
                    args.data.encoder_out_channels, 
                    args.model.n_spk,
                    args.model.use_pitch_aug,
                    vocoder.dimension,
                    args.model.n_layers,
                    args.model.block_out_channels,
                    args.model.n_heads,
                    args.model.n_hidden,
                    use_speaker_encoder=args.model.use_speaker_encoder,
                    speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
                    is_tts=args.model.is_tts,
                    spec_norm=spec_norm,
                    acoustic_scale=args.data.acoustic_scale
                    )
    
    elif args.model.type == 'Naive':
        model = Unit2MelNaive(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                use_speaker_encoder=args.model.use_speaker_encoder,
                speaker_encoder_out_channels=args.data.speaker_encoder_out_channels)

    elif args.model.type == 'NaiveFS':
        model = Unit2MelNaive(
            args.data.encoder_out_channels,
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder.dimension,
            args.model.n_layers,
            args.model.n_chans,
            use_speaker_encoder=args.model.use_speaker_encoder,
            speaker_encoder_out_channels=args.data.speaker_encoder_out_channels,
            use_full_siren=True,
            l2reg_loss=args.model.l2_reg_loss)
    
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    

    if args.train.use_units_quantize:
        if accelerator.is_main_process:
            print("load quantizer")
        if args.train.units_quantize_type == "kmeans":
            from quantize.kmeans_codebook import EuclideanCodebook
            from cluster import get_cluster_model
            codebook_weight = get_cluster_model(args.model.text2semantic.codebook_path).__dict__["cluster_centers_"]
            quantizer = EuclideanCodebook(codebook_weight).to(device)
        elif args.train.units_quantize_type == "vq":
            from vector_quantize_pytorch import VectorQuantize
            quantizer = VectorQuantize(
                dim = args.data.encoder_out_channels,
                codebook_size = args.model.text2semantic.semantic_kmeans_num,
                codebook_dim = 32,
                decay = 0.8,             
                commitment_weight = 1.,
                use_cosine_sim=True
            ).to(device)
        else:
            raise ValueError(' [x] Unknown quantize_type: ' + args.train.units_quantize_type)
        # load parameters
        optimizer = torch.optim.AdamW(itertools.chain(model.parameters(),quantizer.parameters()))
    else:
        quantizer = None
        # load parameters
        optimizer = torch.optim.AdamW(model.parameters())

    
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    if quantizer is not None and args.train.units_quantize_type == "vq":
        try:
            _, quantizer, _ = utils.load_model(args.env.expdir, quantizer, optimizer, device=args.device, postfix=f'{initial_global_step}_semantic_codebook')
        except:
            print(" [x] No semantic codebook found, use random codebook instead.")
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = StepLRWithWarmUp(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2, warm_up_steps=args.train.warm_up_steps, start_lr=float(args.train.start_lr))
    
    # device
    # if args.device == 'cuda':
    #     torch.cuda.set_device(args.env.gpu_id)
    model.to(device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False,accelerator=accelerator)

    _, model, quantizer, optim, scheduler = accelerator.prepare(
        loader_train, model, quantizer, optimizer, scheduler
    )

    
    # run
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, quantizer, accelerator)
    
