import os
import time
import numpy as np
import torch
import librosa
from train_log.saver import Saver, Saver_empty
from train_log import utils
from tools.tools import clip_grad_value_

def test(args, model, vocoder, loader_test, f0_extractor, quantizer, saver, accelerator):
    print(' [*] testing...')
    model.eval()
    if isinstance(quantizer, torch.nn.Module):
        quantizer.eval()
    test_loss = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    with torch.no_grad():
        utilization = 0
        count = torch.zeros(args.model.text2semantic.semantic_kmeans_num).to(accelerator.device)
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))
            
            if args.model.is_tts:
                data['f0'] = None
            if args.model.is_tts:
                data['volume'] = None
            if args.model.is_tts:
                data['aug_shift'] = None


            # unpack data
            for k in data.keys():
                if type(data[k]) is torch.Tensor:
                    data[k] = data[k].to(accelerator.device)

            if quantizer is not None:
                if args.train.units_quantize_type == "kmeans":
                    if args.train.only_load_token:
                        data['units'] = quantizer.decode(data['units'])
                        commit_loss = 0
                    else:
                        data['units'] = quantizer(data['units']).detach()
                        commit_loss = 0
                elif args.train.units_quantize_type == "vq" or args.train.units_quantize_type == "vqae":
                    if args.train.only_load_token:
                        data['units'] = quantizer.project_out(quantizer.codebook[data['units']])
                        commit_loss = 0
                    else:
                        data['units'], indices, commit_loss = quantizer(data['units'])
                else:
                    raise ValueError(' [x] Unknown quantize_type: ' + args.train.units_quantize_type)
            else:
                commit_loss = 0

            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                data['units'],
                data['f0'],
                data['volume'],
                data['spk_id'],
                gt_spec=data['mel'],
                infer=True,
                infer_speedup=args.infer.speedup,
                method=args.infer.method,
                k_step=args.model.k_step_max,
                spk_emb=data['spk_emb'])
            
            if data['f0'] is None and vocoder.vocoder_type != 'hifi-vaegan':
                f0 = f0_extractor.model(mel=mel, infer=True, return_hz_f0=True)
            else:
                f0 = data['f0']

            signal = vocoder.infer(mel, f0)
            ed_time = time.time()

            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)

            # loss
            loss = model(
                data['units'],
                data['f0'],
                data['volume'],
                data['spk_id'],
                gt_spec=data['mel'],
                infer=False,
                k_step=args.model.k_step_max,
                spk_emb=data['spk_emb'])
            test_loss += loss.item()
            test_loss += commit_loss

            # log mel
            if vocoder.vocoder_type != 'hifi-vaegan':
                saver.log_spec(data['name'][0], data['mel'], mel)
            else:
                gt_wav = vocoder.infer(data['mel'], data['f0'])
                gt_mel = vocoder.vocoder.get_mel(gt_wav[0,...])
                mel = vocoder.vocoder.get_mel(signal[0,...])
                saver.log_spec(data['name'][0], gt_mel, mel)

            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})

    # report
    test_loss /= num_batches
    utilization = torch.sum(count > 0).item() / args.model.text2semantic.semantic_kmeans_num
    
    # check
    print(' [test_loss] test_loss:', test_loss.item())
    print(' Real Time Factor', np.mean(rtf_all))
    print(f' Codebook utilization: {utilization*100}%')
    saver.log_value({
        'valid/utilization': utilization
    })
    return test_loss.item()


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test,quantizer, accelerator):
    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
    else:
        saver = Saver_empty(args, initial_global_step=initial_global_step)

    clip_grad_norm = float(args.train.clip_grad_norm) if args.train.clip_grad_norm is not -1 else None

    device = accelerator.device

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    if args.model.is_tts and vocoder.vocoder_type != 'hifi-vaegan':
        from encoder.fcpe.model import FCPEInfer
        f0_extractor = FCPEInfer(model_path='pretrain/fcpe/fcpe.pt')
    else:
        f0_extractor = None

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    if isinstance(quantizer, torch.nn.Module):
        quantizer.train()
    saver.log_info('======= start training =======')
    
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            with accelerator.accumulate(model):
                if args.model.is_tts:
                    data['f0'] = None
                if args.model.is_tts:
                    data['volume'] = None
                if args.model.is_tts:
                    data['aug_shift'] = None

                if accelerator.sync_gradients:
                    saver.global_step_increment()
                
                optimizer.zero_grad()

                # unpack data
                for k in data.keys():
                    if type(data[k]) is torch.Tensor:
                        data[k] = data[k].to(device)

                if quantizer is not None:
                    if args.train.units_quantize_type == "kmeans":
                        if args.train.only_load_token:
                            data['units'] = quantizer.decode(data['units'])
                            commit_loss = 0
                        else:
                            data['units'] = quantizer(data['units']).detach()
                            commit_loss = 0
                    elif args.train.units_quantize_type == "vq" or args.train.units_quantize_type == "vqae":
                        if args.train.only_load_token:
                            data['units'] = quantizer.project_out(quantizer.codebook[data['units']])
                            commit_loss = 0
                        else:
                            data['units'], indices, commit_loss = quantizer(data['units'])
                    else:
                        raise ValueError(' [x] Unknown quantize_type: ' + args.train.units_quantize_type)
                else:
                    commit_loss = 0

                # forward
                loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'],
                            aug_shift=data['aug_shift'], gt_spec=data['mel'].float(), infer=False, k_step=args.model.k_step_max,
                            spk_emb=data['spk_emb']) + commit_loss
                


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
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | vq_loss: {:.3f} | grad_norm: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log / saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        commit_loss.item() if type(commit_loss) is torch.Tensor else 0,
                        grad_norm,
                        saver.get_total_time(),
                        saver.global_step
                    )
                )

                saver.log_value({
                    'train/loss': loss.item()
                })

                saver.log_value({
                    'train/vq_loss': commit_loss.item() if type(commit_loss) is torch.Tensor else 0
                })

                saver.log_value({
                    'train/lr': current_lr
                })

            # validation
            if accelerator.is_main_process and saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.model.text2semantic.train.save_opt else None
                unwrap_model = accelerator.unwrap_model(model)
                unwrap_quantizer = quantizer
                # save latest
                if saver.global_step % args.train.interval_force_save == 0:
                    saver.save_model(unwrap_model, optimizer_save, postfix=f'{saver.global_step}_Force')
                else:
                    saver.save_model(unwrap_model, optimizer, postfix=f'{saver.global_step}')

                last_val_step = saver.global_step - args.train.interval_val * (args.train.last_save_model_num + 1)
                saver.delete_model(postfix=f'{last_val_step}')

                if args.train.units_quantize_type == "vq" and quantizer is not None and not args.train.vq_freeze:
                    # save latest
                    unwrap_quantizer = accelerator.unwrap_model(quantizer)
                    if saver.global_step % args.train.interval_force_save == 0:
                        saver.save_model(unwrap_quantizer, None, postfix=f'semantic_codebook_{saver.global_step}_Force')
                    else:
                        saver.save_model(unwrap_quantizer, None, postfix=f'semantic_codebook_{saver.global_step}')
                    
                    last_val_step = saver.global_step - args.train.interval_val * (args.train.last_save_model_num + 1)
                    saver.delete_model(postfix=f'semantic_codebook_{last_val_step}')

                # run testing set
                test_loss = test(args, unwrap_model, vocoder, loader_test, f0_extractor, unwrap_quantizer, saver, accelerator)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                model.train()
                if isinstance(quantizer, torch.nn.Module):
                    quantizer.train()
            accelerator.wait_for_everyone()
