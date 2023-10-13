import os
import time
import numpy as np
import torch
import librosa
from text2semantic.saver import Saver,Saver_empty
from logger import utils
from torch import autocast
from torch.cuda.amp import GradScaler

@torch.no_grad()
def test(args, model, loader_test, diffusion_model, saver, accelerator):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.

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
            )
            if diffusion_model is not None:
                signal = diffusion_model.infer(semantic_token, None, None)
            else:
                signal = None
            ed_time = time.time()

            run_time = ed_time - st_time
            print('Run time: {}'.format(run_time))

            # loss
            for i in range(args.train.batch_size):
                loss = model(
                    **data
                    ).loss
                test_loss += loss.item()

            # log audio
            if signal is not None:
                path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
                audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
                saver.log_audio({fn + '/gt.wav': audio, fn + '/pred.wav': signal})

    # report
    test_loss /= args.train.batch_size
    test_loss /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    return test_loss

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
                            data[k][data[k] == -100] = model.PAD
                        if k == "tone":
                            data[k][data[k] == -100] = model.num_tones
                        if k == "semantic":
                            data[k][data[k] == -100] = model.semantic_pad_token_id
                # forward
                loss = model(**data).loss
                
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
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.model.text2semantic.train.expdir,
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

                # run testing set
                test_loss = test(args, model, loader_valid, diffusion_model, saver, accelerator)

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
