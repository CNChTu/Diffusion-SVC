import os
import torch

def get_data_loaders(args, model, accelerate = None):
    model_type = args.model.text2semantic.type
    if model_type == "roformer":
        from text2semantic.roformer.dataloader import get_data_loaders
    else:
        raise ValueError(f" [x] Unknown Model: {model_type}")
    
    loader_train, loader_val = get_data_loaders(args, model, accelerate = accelerate)
    
    return loader_train, loader_val

