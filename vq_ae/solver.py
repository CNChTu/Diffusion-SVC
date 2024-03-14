import os
import time
import numpy as np
import torch
import librosa
from vq_ae.saver import Saver,Saver_empty
from train_log import utils
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from cluster import get_cluster_model
from tools.tools import clip_grad_value_

@torch.no_grad()
def test(args, model, loader_test, accelerator):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    # intialization
    num_batches = len(loader_test)
    count = torch.zeros(args.model.text2semantic.semantic_kmeans_num).to(accelerator.device)
    
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

            # loss
            loss_1, loss_2 = model(
                training = True, **data
            )
            _, indices, _ = model(**data)
            count += torch.bincount(indices.flatten(), minlength=args.model.text2semantic.semantic_kmeans_num)

            ed_time = time.time()

            run_time = ed_time - st_time
            print('Run time: {}'.format(run_time))
            
            test_loss += loss_1.item() + loss_2.item()

    
    utilization = torch.sum(count > 0).item() / args.model.text2semantic.semantic_kmeans_num
    print(f' Codebook utilization: {utilization*100}%')
    # report
    test_loss /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    return test_loss

def train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_valid, accelerator):
        # saver
    if accelerator.is_main_process:
        saver = Saver(args, initial_global_step=initial_global_step)
    else:
        saver = Saver_empty(args, initial_global_step=initial_global_step)
    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    clip_grad_norm = float(args.train.clip_grad_norm) if args.train.clip_grad_norm is not -1 else None

    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')

    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            with accelerator.accumulate(model):
                if accelerator.sync_gradients:
                    saver.global_step_increment()
                optimizer.zero_grad()
                # unpack data
                for k in data.keys():
                    if type(data[k]) is torch.Tensor:
                        data[k] = data[k].to(accelerator.device)
                # forward
                l1_loss, com_loss = model(**data)
                grad_norm = clip_grad_value_(model.parameters(), clip_grad_norm)
                loss = l1_loss + com_loss

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
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | grad_norm: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.vq_expdir,
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
                optimizer_save = optimizer if args.train.save_opt else None
                unwrap_model = accelerator.unwrap_model(model)

                # save latest
                if saver.global_step % args.train.interval_force_save == 0:
                    saver.save_model(unwrap_model, optimizer_save, postfix=f'{saver.global_step}_Force')
                else:
                    saver.save_model(unwrap_model, optimizer, postfix=f'{saver.global_step}')
                
                unwrap_quantizer = accelerator.unwrap_model(model.quantizer)
                if saver.global_step % args.train.interval_force_save == 0:
                    saver.save_model(unwrap_quantizer, None, postfix=f'{saver.global_step}_semantic_codebook_Force')
                else:
                    saver.save_model(unwrap_quantizer, None, postfix=f'{saver.global_step}_semantic_codebook')
                
                last_val_step = saver.global_step - args.train.interval_val * (args.train.last_save_model_num + 1)
                saver.delete_model(postfix=f'{last_val_step}')
                saver.delete_model(postfix=f'{last_val_step}_semantic_codebook')

                # run testing set
                test_loss = test(args, unwrap_model, loader_valid, accelerator)

                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss
                    )
                )

                saver.log_value({
                    'validation/loss': test_loss
                })

                model.train()
            accelerator.wait_for_everyone()