import os
import time
import numpy as np
import torch
import librosa
from logger.saver import Saver, Saver_empty
from logger import utils
from torch.cuda.amp import GradScaler

def test(args, model, vocoder, loader_test, f0_extractor, quantizer, saver, accelerator):
    print(' [*] testing...')
    model.eval()

    test_loss = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))
            
            if data['f0'][0] == -1:
                data['f0'] = None
            if data['volume'][0] == -1:
                data['volume'] = None
            if data['aug_shift'][0] == -1:
                data['aug_shift'] = None

            # unpack data
            for k in data.keys():
                if type(data[k]) is torch.Tensor:
                    data[k] = data[k].to(accelerator.device)

            if quantizer is not None:
                if args.train.units_quantize_type == "kmeans":
                    data['units'] = quantizer(data['units']).detach()
                    commit_loss = 0
                elif args.train.units_quantize_type == "vq":
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
            
            if data['f0'] is None:
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
            saver.log_spec(data['name'][0], data['mel'], mel)

            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})

    # report
    test_loss /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test,quantizer, accelerator):
    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
    else:
        saver = Saver_empty(args, initial_global_step=initial_global_step)

    device = accelerator.device

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    if args.model.is_tts:
        from encoder.fcpe.model import FCPEInfer
        f0_extractor = FCPEInfer(model_path='pretrain/fcpe/fcpe.pt')
    else:
        f0_extractor = None

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            with accelerator.accumulate(model):
                if data['f0'][0] == -1:
                    data['f0'] = None
                if data['volume'][0] == -1:
                    data['volume'] = None
                if data['aug_shift'][0] == -1:
                    data['aug_shift'] = None


                saver.global_step_increment()
                optimizer.zero_grad()

                # unpack data
                for k in data.keys():
                    if type(data[k]) is torch.Tensor:
                        data[k] = data[k].to(device)

                if quantizer is not None:
                    if args.train.units_quantize_type == "kmeans":
                        data['units'] = quantizer(data['units']).detach()
                        commit_loss = 0
                    elif args.train.units_quantize_type == "vq":
                        data['units'], indices, commit_loss = quantizer(data['units'])
                        data['units'] = data['units'].detach()
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
                    optimizer.step()
                    scheduler.step()

            # log loss
            if accelerator.is_main_process and saver.global_step % args.train.interval_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | vq_loss: {:3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log / saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        commit_loss.item() if type(commit_loss) is torch.Tensor else 0,
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

                # save latest
                if saver.global_step % args.train.interval_force_save == 0:
                    saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}_Force')
                else:
                    saver.save_model(model, optimizer, postfix=f'{saver.global_step}')

                last_val_step = saver.global_step - args.train.interval_val * args.train.last_save_model_num
                saver.delete_model(postfix=f'{last_val_step}')

                if args.train.units_quantize_type == "vq":
                    # save latest
                    saver.save_model(quantizer, None, postfix=f'{saver.global_step}_semantic_codebook')
                    last_val_step = saver.global_step - args.train.interval_val
                    if last_val_step % args.train.interval_force_save != 0:
                       saver.delete_model(postfix=f'{last_val_step}_semantic_codebook')

                # run testing set
                if type(model) is torch.nn.parallel.DistributedDataParallel:
                    raw_model = model.module
                else:
                    raw_model = model
                    
                test_loss = test(args, raw_model, vocoder, loader_test, f0_extractor, quantizer, saver, accelerator)
                
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
