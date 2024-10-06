import os
import argparse
import torch
from torch.optim import lr_scheduler
from logger import utils
from diffusion.data_loaders import get_data_loaders
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel, Unit2MelNaive, load_svc_model
from diffusion.vocoder import Vocoder
import time


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
        "-p",
        "--print",
        type=str,
        required=False,
        default=None,
        help="print model")
    parser.add_argument(
        "-pre",
        "--pretrain",
        type=str,
        required=False,
        default=None,
        help="path to the pretraining model")
    return parser.parse_args(args=args, namespace=namespace)


def train_run(rank, config_path, print_model, pretrain, ddp=False, ddp_device_list=None):
    args = utils.load_config(config_path)
    # load vocoder
    if not ddp:
        vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=f"{args.device}:{args.env.gpu_id}")
    else:
        vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=ddp_device_list[rank])

    # load model
    model = load_svc_model(args=args, vocoder_dimension=vocoder.dimension)

    # load parameters not ddp
    if not ddp:
        optimizer = torch.optim.AdamW(model.parameters())
        initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer,
                                                                 device=f"{args.device}:{args.env.gpu_id}")
        if pretrain is not None:  # 加载预训练模型
            if initial_global_step == 0:
                _ckpt = torch.load(pretrain, map_location=torch.device(f"{args.device}:{args.env.gpu_id}"),
                                   weights_only=True)
                model.load_state_dict(_ckpt['model'], strict=False)
                if _ckpt.get('optimizer') != None:
                    optimizer = torch.optim.AdamW(model.parameters())
                    optimizer.load_state_dict(_ckpt['optimizer'])
    else:
        optimizer = None
        initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer,
                                                                 device=ddp_device_list[rank], model_only=True)
        if pretrain is not None:  # 加载预训练模型
            if initial_global_step == 0:
                _ckpt = torch.load(pretrain, map_location=torch.device(ddp_device_list[rank]),weights_only=True)
                model.load_state_dict(_ckpt['model'], strict=False)

    # device
    if ddp:
        # init
        if rank != 0:
            time.sleep(5)
        torch.distributed.init_process_group(
            backend='gloo' if os.name == 'nt' else 'nccl',
            init_method='env://', world_size=len(ddp_device_list),
            rank=rank
        )
        # device
        device = ddp_device_list[rank]
        torch.cuda.set_device(torch.device(device))
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters())
        if pretrain is not None:  # 加载预训练模型
            if initial_global_step == 0:
                _ckpt = torch.load(pretrain, map_location=device, weights_only=True)
                if _ckpt.get('optimizer') != None:
                    optimizer.load_state_dict(_ckpt['optimizer'])
                else:
                    optimizer = utils.load_optimizer(args.env.expdir, optimizer, device=device)
            else:
                optimizer = utils.load_optimizer(args.env.expdir, optimizer, device=device)
        else:
            optimizer = utils.load_optimizer(args.env.expdir, optimizer, device=device)
    else:
        device = args.device
        if args.device == 'cuda':
            torch.cuda.set_device(args.env.gpu_id)
        model = model.to(device)
    # 打印模型结构
    if (str(print_model) == 'True') or (str(print_model) == 'true'):
        print(model)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    # init scheduler
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max(
            (initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma,
                                    last_epoch=initial_global_step - 2)

    # datas  run
    if ddp:
        loader_train, loader_valid, samper_train = get_data_loaders(args, whole_audio=False, ddp=True, rank=rank,
                                                                    ddp_cache_gpu=args.ddp.ddp_cache_gpu,
                                                                    ddp_device_list=ddp_device_list)
        if rank != 0:
            loader_valid = None
        train(rank, args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, device,
              ddp=ddp, samper_train=samper_train)
    else:
        loader_train, loader_valid, samper_train = get_data_loaders(args, whole_audio=False, ddp=False, rank=0)
        train(0, args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid, device,
              ddp=ddp, samper_train=samper_train)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    if args.ddp.use_ddp:
        # device
        device_list = args.ddp.ddp_device
        device_ids = []
        for device in device_list:
            _device_ids = device.split(':')[-1]
            device_ids.append(int(_device_ids))
        # init gloo or nccl
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.ddp.port
        # run
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.spawn(train_run, args=(cmd.config, cmd.print, True, device_list), nprocs=len(device_ids))

    else:
        train_run(0, cmd.config, cmd.print, cmd.pretrain, ddp=False)
