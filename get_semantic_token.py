import torch
import cluster
import numpy as np 
import torch.multiprocessing as mp
import argparse
from logger import utils
import os
from logger.utils import traverse_dir
from tqdm import tqdm
# def _process_utterance(wav_path):
#         wav = np.load(wav_path)
#         token = cluster.get_cluster_result(model, wav)
#         out_path = out_dir / (wav_path.stem + ".npy")
#         np.save(out_path, token)


def preprocess_utterance(rank, units_path, model,in_dir, out_dir, num_workers):
    units_path = units_path[rank::num_workers]
    for unit_path in tqdm(units_path,position=rank):
        unit = np.load(os.path.join(in_dir, "units" , unit_path))
        token = cluster.get_cluster_result(model, unit)
        out_path = os.path.join(out_dir, unit_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, token)

def preprocess(in_dir, model, num_workers=1):
    """Preprocess the training set."""
    out_dir = os.path.join(in_dir, "semantic_token")
    os.makedirs(out_dir, exist_ok=True)

    units_dir = os.path.join(in_dir, "units")
    # list files
    filelist = traverse_dir(
        units_dir,
        extensions=["npy"],
        is_pure=True,
        is_sort=True,
        is_ext=True)
    
    mp.spawn(preprocess_utterance, args=(filelist, model,in_dir, out_dir, num_workers), nprocs=num_workers, join=True)
    
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
        "-n",
        "--num_workers",
        type=int,
        default= 8,
        help="num workers")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == "__main__":
    cmd = parse_args()
    args = utils.load_config(cmd.config)
    num_workers = cmd.num_workers

    model = cluster.get_cluster_model("pretrain/semantic_codebook.pt")
    
    # preprocess training set
    preprocess(args.data.train_path, model, num_workers=num_workers)
    # preprocess validation set
    preprocess(args.data.valid_path, model, num_workers=num_workers)