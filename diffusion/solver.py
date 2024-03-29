import os
import time
import numpy as np
import torch
import librosa
from logger.saver import Saver
from logger import utils
from torch import autocast
from torch.cuda.amp import GradScaler
from nsf_hifigan.nvSTFT import STFT

WAV_TO_MEL = None


def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
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

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
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
                infer_step=args.infer.infer_step,
                method=args.infer.method,
                t_start=args.model.t_start,
                spk_emb=data['spk_emb'])
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()

            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)

            # loss
            for i in range(args.train.batch_size):
                loss_dict = model(
                    data['units'],
                    data['f0'],
                    data['volume'],
                    data['spk_id'],
                    gt_spec=data['mel'],
                    infer=False,
                    t_start=args.model.t_start,
                    spk_emb=data['spk_emb'],
                    use_vae=(args.vocoder.type == 'hifivaegan')
                )
                _loss = 0
                for k in loss_dict.keys():
                    _loss += loss_dict[k].item()
                test_loss += _loss

            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})

            # log mel
            if args.vocoder.type == 'hifivaegan':
                log_from_signal = True
            else:
                log_from_signal = False

            if log_from_signal:
                global WAV_TO_MEL
                if WAV_TO_MEL is None:
                    WAV_TO_MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=0,
                        fmax=22050,
                        clip_val=1e-5)
                audio = audio.unsqueeze(0)
                pre_mel = WAV_TO_MEL.get_mel(signal[0, ...])
                pre_mel = pre_mel.transpose(-1, -2)
                gt_mel = WAV_TO_MEL.get_mel(audio[0, ...])
                gt_mel = gt_mel.transpose(-1, -2)
                # 如果形状不同,裁剪使得形状相同
                if pre_mel.shape[1] != gt_mel.shape[1]:
                    gt_mel = gt_mel[:, :pre_mel.shape[1], :]
                saver.log_spec(data['name'][0], gt_mel, pre_mel)
            else:
                saver.log_spec(data['name'][0], data['mel'], mel)

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    if args.vocoder.type == 'hifivaegan':
        use_vae = True
    else:
        use_vae = False

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)

            # forward
            if args.model.type == 'ReFlow':
                if dtype == torch.float32:
                    loss_dict = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'],
                                      aug_shift=data['aug_shift'],
                                      gt_spec=data['mel'].float(), infer=False,
                                      t_start=args.model.t_start,
                                      spk_emb=data['spk_emb'], use_vae=use_vae)
                else:
                    with autocast(device_type=args.device, dtype=dtype):
                        loss_dict = model(data['units'], data['f0'], data['volume'], data['spk_id'],
                                          aug_shift=data['aug_shift'], gt_spec=data['mel'], infer=False,
                                          t_start=args.model.t_start,
                                          spk_emb=data['spk_emb'], use_vae=use_vae)

            else:
                if dtype == torch.float32:
                    loss_dict = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'],
                                      aug_shift=data['aug_shift'], gt_spec=data['mel'].float(), infer=False,
                                      k_step=args.model.k_step_max,
                                      spk_emb=data['spk_emb'], use_vae=use_vae)
                else:
                    with autocast(device_type=args.device, dtype=dtype):
                        loss_dict = model(data['units'], data['f0'], data['volume'], data['spk_id'],
                                          aug_shift=data['aug_shift'], gt_spec=data['mel'], infer=False,
                                          k_step=args.model.k_step_max,
                                          spk_emb=data['spk_emb'], use_vae=use_vae)

            # sum loss
            if not isinstance(loss_dict, dict):
                loss_dict = {f'{args.model.type}_loss': loss_dict}

            loss = None
            loss_float_dict = {}
            for k in loss_dict.keys():
                _loss = loss_dict[k]
                loss_float_dict[k] = _loss.item()
                if loss is None:
                    loss = _loss
                else:
                    loss += _loss

            # handle nan loss
            if torch.isnan(loss):
                # raise ValueError(' [x] nan loss ')
                # 如果是nan,则跳过这个batch,并清理以防止内存泄漏
                print(' [x] nan loss ')
                optimizer.zero_grad()
                del loss
                continue
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log / saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )

                saver.log_value({
                    'train/loss': loss.item()
                })

                for k in loss_float_dict.keys():
                    saver.log_value({
                        'train/' + k: loss_float_dict[k]
                    })

                saver.log_value({
                    'train/lr': current_lr
                })

            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None

                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')

                # run testing set
                test_loss = test(args, model, vocoder, loader_test, saver)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                for k in loss_float_dict.keys():
                    saver.log_value({
                        'validation/' + k: loss_float_dict[k]
                    })

                model.train()
