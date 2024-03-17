import os
import time
import numpy as np
import torch
import librosa
from text2semantic.saver import Saver,Saver_empty
from train_log import utils
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from cluster import get_cluster_model
from ..utils import get_topk_acc
from tools.tools import clip_grad_value_

@torch.no_grad()
def test(args, model, loader_test, diffusion_model, saver,semantic_embedding, accelerator):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    topk_acc = 0
    # intialization
    num_batches = len(loader_test)

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if type(data[k]) is torch.Tensor:
                    data[k] = data[k].to(accelerator.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            semantic_token = model.generate(
                phone = data["phone"],
                tone = data["tone"],
                attention_mask = data["encoder_attention_mask"],
                spk_id = data["spk_id"],
            )
            
            if semantic_token[:,-1] == model.semantic_eos_token_id:
                semantic_token = semantic_token[:,1:-1]
            else:
                semantic_token = semantic_token[:,1:]

            if args.train.units_quantize_type == "kmeans":
                semantic_emb = semantic_embedding(semantic_token)
            elif args.train.units_quantize_type == "vq" or args.train.units_quantize_type == "vqae":
                semantic_emb = semantic_embedding.project_out(semantic_embedding.get_codes_from_indices(semantic_token[:,:,None]))

            if diffusion_model is not None:
                signal = diffusion_model.infer(semantic_emb, None, None)
            else:
                signal = None
            ed_time = time.time()

            run_time = ed_time - st_time
            print('Run time: {}'.format(run_time))

            # loss
            result = model(
                **data
                )
            test_loss += result.loss.item()
            topk_acc += get_topk_acc(data["semantic"][0][1:], result.logits[0][:-1,:], k = 5)
            

            # log audio
            if signal is not None:
                path_audio = os.path.join(args.data.valid_path, 'audio', data['name'][0].replace(".npy",""))
                audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
                saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})

    # report
    test_loss /= num_batches
    topk_acc /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    return test_loss, topk_acc

def train(args, initial_global_step, model, optimizer, scheduler, diffusion_model, loader_train, loader_valid, accelerator):
        # saver
    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
    else:
        saver = Saver_empty(args, initial_global_step=initial_global_step)
    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    saver.log_info('load semantic codebook')
    
    clip_grad_norm = float(args.model.text2semantic.train.clip_grad_norm) if args.model.text2semantic.train.clip_grad_norm != -1 else None

    if args.train.units_quantize_type == "kmeans":
        codebook = get_cluster_model(args.model.text2semantic.codebook_path)
        codebook = codebook.__dict__["cluster_centers_"]
        
        semantic_embedding = torch.nn.Embedding(
            codebook.shape[0],
            codebook.shape[1],
            _freeze = True
            )
        semantic_embedding.weight.data = torch.from_numpy(codebook)
        semantic_embedding.to(accelerator.device)
    elif args.train.units_quantize_type == "vq":
        from vector_quantize_pytorch import VectorQuantize
        semantic_embedding = VectorQuantize(
                dim = args.data.encoder_out_channels,
                codebook_size = args.model.text2semantic.semantic_kmeans_num,
                codebook_dim=32,
                decay = 0.8,             
                commitment_weight = 1.,
                use_cosine_sim=True,
                freeze_codebook=True
            )
        model_para = torch.load(args.model.text2semantic.codebook_path)
        semantic_embedding.load_state_dict(model_para["model"])
        semantic_embedding = semantic_embedding.to(accelerator.device)
    elif args.train.units_quantize_type == "vqae":
        from vq_ae import get_model
        quantizer = get_model(args)
        quantizer.load_state_dict(torch.load(args.model.text2semantic.codebook_path)["model"])
        quantizer.set_eval_mode()
        quantizer = quantizer
        semantic_embedding = quantizer.quantizer.to(accelerator.device)
    else:
        raise ValueError(' [x] Unknown quantize_type: ' + args.train.units_quantize_type)

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')

    for epoch in range(start_epoch, args.model.text2semantic.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            with accelerator.accumulate(model):
                if accelerator.sync_gradients:
                    saver.global_step_increment()

                optimizer.zero_grad()

                # unpack data
                for k in data.keys():
                    if type(data[k]) is torch.Tensor:
                        data[k] = data[k].to(accelerator.device)
                        if k == "phone":
                            data[k][data[k] == -100] = accelerator.unwrap_model(model).PAD
                        if k == "tone" and data[k] is not None:
                            data[k][data[k] == -100] = accelerator.unwrap_model(model).num_tones
                        if k == "semantic":
                            data[k][data[k] == -100] = accelerator.unwrap_model(model).semantic_pad_token_id
                # forward
                loss = model(**data).loss

                # handle nan loss
                if torch.isnan(loss):
                    raise ValueError(' [x] nan loss ')
                else:
                    accelerator.backward(loss)
                    grad_norm = clip_grad_value_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    scheduler.step()

            # log loss
            if accelerator.is_main_process and saver.global_step % args.train.interval_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | grad_norm: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.model.text2semantic.train.expdir,
                        args.train.interval_log / saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        grad_norm,
                        saver.get_total_time(),
                        saver.global_step
                    )
                )

                saver.log_value({
                    'train/loss': loss.item()
                })

                saver.log_value({
                    'train/lr': current_lr
                })

            # validation
            if accelerator.is_main_process and saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.model.text2semantic.train.save_opt else None
                unwrap_model = accelerator.unwrap_model(model)

                # save latest
                if saver.global_step % args.train.interval_force_save == 0:
                    saver.save_model(unwrap_model, optimizer_save, postfix=f'{saver.global_step}_Force')
                else:
                    saver.save_model(unwrap_model, optimizer, postfix=f'{saver.global_step}')

                last_val_step = saver.global_step - args.train.interval_val * (args.train.last_save_model_num + 1)
                saver.delete_model(postfix=f'{last_val_step}')

                # run testing set
                test_loss, topk_acc = test(args, unwrap_model, loader_valid, diffusion_model, saver,semantic_embedding, accelerator)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. \ntop_acc@5: {:.3f}.'.format(
                        test_loss, topk_acc
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                saver.log_value({
                    'validation/top_acc@5': topk_acc
                })

                model.train()
            accelerator.wait_for_everyone()