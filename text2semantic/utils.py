import os
import torch

def get_data_loaders(args, whole_audio = False, accelerate = None):
    model_type = args.model.text2semantic.type
    if model_type == "roformer":
        import text2semantic.roformer.dataloader as TextDataset
    else:
        raise ValueError(f" [x] Unknown Model: {model_type}")
    
    data_train = TextDataset(
        path_root = args.data.train_path,
        use_cache = args.model.text2semantic.train.cache_all_data,
        accelerate = accelerate
    )

    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.text2semantic.train.batch_size,
        shuffle=True,
        num_workers = args.model.text2semantic.train.num_workers if not args.model.text2semantic.train.cache_all_data else 0,
        persistent_workers=(args.model.text2semantic.train.num_workers > 0) if not args.model.text2semantic.train.cache_all_data else False,
        pin_memory=True if not args.model.text2semantic.train.cache_all_data else False
    )

    data_val = TextDataset(
        path_root = args.data.valid_path,
        use_cache = args.model.text2semantic.train.cache_all_data,
        accelerate = accelerate
    )

    loader_val = torch.utils.data.DataLoader(
        data_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    loader_train, loader_val

