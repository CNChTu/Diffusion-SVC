import os
import argparse
import torch
from torch.optim import lr_scheduler
from train_log import utils
import accelerate
from tools.infer_tools import DiffusionSVC
from tools.tools import StepLRWithWarmUp
from vq_ae.dataloader import get_data_loaders
from vq_ae.model import get_model
from vq_ae.solver import train
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
    
    # load config
    args = utils.load_config(cmd.config)
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(' > config:', cmd.config)
        print(' >    exp:', args.env.vq_expdir)
    
    # load vocoder
    # vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    model = get_model(args)
    
    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.vq_expdir, model, optimizer, device=device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step, 0)
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = StepLRWithWarmUp(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma, last_epoch=initial_global_step-2, warm_up_steps=args.train.warm_up_steps, start_lr=float(args.train.start_lr))
    
    model = model.to(device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, accelerate=accelerator)
    _, model, optimizer, scheduler = accelerator.prepare(
        loader_train, model, optimizer, scheduler
    )

    # run
    train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_valid, accelerator)
    
