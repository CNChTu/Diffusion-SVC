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


def calculate_mel_snr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SNR
    snr = 10 * torch.log10(mean_square_reference / variance_error)
    return snr


def calculate_mel_si_snr(gt_mel, pred_mel):
    # 将测试图像按比例调整以最小化误差
    scale = torch.sum(gt_mel * pred_mel) / torch.sum(gt_mel ** 2)
    test_image_scaled = scale * pred_mel
    # 计算误差图像
    error_image = gt_mel - test_image_scaled
    # 计算参考图像的平方均值
    mean_square_reference = torch.mean(gt_mel ** 2)
    # 计算误差图像的方差
    variance_error = torch.var(error_image)
    # 计算并返回SI-SNR
    si_snr = 10 * torch.log10(mean_square_reference / variance_error)
    return si_snr


def calculate_mel_psnr(gt_mel, pred_mel):
    # 计算误差图像
    error_image = gt_mel - pred_mel
    # 计算误差图像的均方误差
    mse = torch.mean(error_image ** 2)
    # 计算参考图像的最大可能功率
    max_power = torch.max(gt_mel) ** 2
    # 计算并返回PSNR
    psnr = 10 * torch.log10(max_power / mse)
    return psnr


def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.

    # intialization
    num_batches = len(loader_test)
    rtf_all = []

    # mel mse val
    mel_val_mse_all = 0
    mel_val_mse_all_num = 0
    mel_val_snr_all = 0
    mel_val_psnr_all = 0
    mel_val_sisnr_all = 0

    test_loss_dict = {}
    test_loss_dict_num = 0

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
            if args.model.type == 'ReFlow' or args.model.type == 'ReFlow1Step':
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
            else:
                mel = model(
                    data['units'],
                    data['f0'],
                    data['volume'],
                    data['spk_id'],
                    gt_spec=data['mel'],
                    infer=True,
                    infer_speedup=args.infer.speedup,
                    k_step=args.model.k_step_max,
                    method=args.infer.method,
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
                if args.model.type == 'ReFlow' or args.model.type == 'ReFlow1Step':
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
                else:
                    loss_dict = model(
                        data['units'],
                        data['f0'],
                        data['volume'],
                        data['spk_id'],
                        gt_spec=data['mel'],
                        infer=False,
                        k_step=args.model.k_step_max,
                        spk_emb=data['spk_emb'],
                        use_vae=(args.vocoder.type == 'hifivaegan')
                    )
                _loss = 0
                if not isinstance(loss_dict, dict):
                    loss_dict = {f'{args.model.type}_loss': loss_dict}
                for k in loss_dict.keys():
                    _loss += loss_dict[k].item()
                    if k not in test_loss_dict:
                        test_loss_dict[k] = loss_dict[k].item()
                    else:
                        test_loss_dict[k] += loss_dict[k].item()
                test_loss_dict_num += 1
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

            spec_min = float(args.model.spec_min) if args.model.spec_min is not None else -12
            spec_max = float(args.model.spec_max) if args.model.spec_max is not None else 2
            spec_range = spec_max - spec_min

            if log_from_signal:
                global WAV_TO_MEL
                if WAV_TO_MEL is None:
                    WAV_TO_MEL = STFT(
                        sr=args.data.sampling_rate,
                        n_mels=128,
                        n_fft=2048,
                        win_size=2048,
                        hop_length=512,
                        fmin=40,
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
                # 计算指标
                mel_val_mse_all += torch.nn.functional.mse_loss(pre_mel, gt_mel).detach().cpu().numpy()
                gt_mel_norm = torch.clip(gt_mel, spec_min, spec_max)
                gt_mel_norm = gt_mel_norm / spec_range + spec_min
                pre_mel_norm = torch.clip(pre_mel, spec_min, spec_max)
                pre_mel_norm = pre_mel_norm / spec_range + spec_min
                mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_mse_all_num += 1
            else:
                saver.log_spec(data['name'][0], data['mel'], mel)
                # 计算指标
                mel_val_mse_all += torch.nn.functional.mse_loss(mel, data['mel']).detach().cpu().numpy()
                gt_mel_norm = torch.clip(data['mel'], spec_min, spec_max)
                gt_mel_norm = gt_mel_norm / spec_range + spec_min
                pre_mel_norm = torch.clip(mel, spec_min, spec_max)
                pre_mel_norm = pre_mel_norm / spec_range + spec_min
                mel_val_snr_all += calculate_mel_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_psnr_all += calculate_mel_psnr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_sisnr_all += calculate_mel_si_snr(gt_mel_norm, pre_mel_norm).detach().cpu().numpy()
                mel_val_mse_all_num += 1

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches
    mel_val_mse_all /= mel_val_mse_all_num
    mel_val_snr_all /= mel_val_mse_all_num
    mel_val_psnr_all /= mel_val_mse_all_num
    mel_val_sisnr_all /= mel_val_mse_all_num

    for k in test_loss_dict.keys():
        test_loss_dict[k] /= test_loss_dict_num

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    print(' Mel Val MSE', mel_val_mse_all)
    saver.log_value({
        'validation/mel_val_mse': mel_val_mse_all
    })
    print(' Mel Val SNR', mel_val_snr_all)
    saver.log_value({
        'validation/mel_val_snr': mel_val_snr_all
    })
    print(' Mel Val PSNR', mel_val_psnr_all)
    saver.log_value({
        'validation/mel_val_psnr': mel_val_psnr_all
    })
    print(' Mel Val SI-SNR', mel_val_sisnr_all)
    saver.log_value({
        'validation/mel_val_sisnr': mel_val_sisnr_all
    })
    return test_loss_dict, test_loss


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

    # set up EMA
    if args.train.use_ema:
        ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        saver.log_info('ModelEmaV2 is enable')

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
            if (args.model.type == 'ReFlow') or (args.model.type == 'ReFlow1Step'):
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
                
                if args.train.use_ema:
                    ema_model.update_parameters(model)
                
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
                if args.train.use_ema:
                    saver.save_model(ema_model, optimizer_save, postfix=f'{saver.global_step}')
                else:
                    saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')

                # run testing set
                if args.train.use_ema:
                    test_loss_dict, test_loss = test(args, ema_model, vocoder, loader_test, saver)
                else:
                    test_loss_dict, test_loss = test(args, model, vocoder, loader_test, saver)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                for k in test_loss_dict.keys():
                    saver.log_value({
                        'validation/' + k: test_loss_dict[k]
                    })

                model.train()
