import os
import torch

def get_data_loaders(args, model, accelerate = None):
    model_type = args.model.text2semantic.type
    if model_type == "roformer":
        from text2semantic.roformer.dataloader import get_data_loaders
    elif model_type == "llama":
        from text2semantic.llama.dataloader import get_data_loaders
    else:
        raise ValueError(f" [x] Unknown Model: {model_type}")
    
    loader_train, loader_val = get_data_loaders(args, model, accelerate = accelerate)
    
    return loader_train, loader_val

@torch.no_grad()
def get_topk_acc(gt_token, logist, k = 5):
    _, topk = torch.topk(logist, k, dim=-1)
    topk = topk.cpu().numpy()
    gt_token = gt_token.cpu().numpy()
    return sum([1 if gt_token[i] in topk[i] else 0 for i in range(len(gt_token))]) / len(gt_token)

def get_language_model(**args):
    model_type = args["type"]
    if model_type == "roformer":
        from text2semantic.roformer.roformer import get_model
    elif model_type == "llama":
        from text2semantic.llama.llama import get_model
    else:
        raise ValueError(f" [x] Unknown Model: {model_type}")
    
    model = get_model(**args)
    return model