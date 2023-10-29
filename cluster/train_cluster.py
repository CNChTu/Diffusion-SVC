import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from kmeans import KMeansGPU
from sklearn.cluster import KMeans, MiniBatchKMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_cluster(in_dirs, n_clusters, use_minibatch=True, verbose=False,use_gpu=False):#gpu_minibatch真拉，虽然库支持但是也不考虑
    features = []
    for in_dir in in_dirs:
        logger.info(f"Loading features from {in_dir}")

        for path in tqdm.tqdm(in_dir.glob("*.wav.npy")):
        # for name in os.listdir(in_dir):
        #     path="%s/%s"%(in_dir,name)
            features.append(np.load(path))
            # print(features[-1].shape)
    features = np.concatenate(features, axis=0)
    print(features.nbytes/ 1024**2, "MB , shape:",features.shape, features.dtype)
    features = features.astype(np.float32)
    logger.info(f"Clustering features of shape: {features.shape}")
    t = time.time()
    if(use_gpu is False):
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters,verbose=verbose, batch_size=4096, max_iter=80).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters,verbose=verbose).fit(features)
    else:
            kmeans = KMeansGPU(n_clusters=n_clusters, mode='euclidean', verbose=2 if verbose else 0,max_iter=500,tol=1e-2)#
            features=torch.from_numpy(features)#.to(device)
            kmeans.fit_predict(features)#

    print(time.time()-t, "s")

    x = {
            "n_features_in_": kmeans.n_features_in_ if use_gpu is False else features.shape[1],
            "_n_threads": kmeans._n_threads if use_gpu is False else 4,
            "cluster_centers_": kmeans.cluster_centers_ if use_gpu is False else kmeans.centroids.cpu().numpy(),
    }
    print("end")

    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default="./data/train/units",
                        help='path of training data directory')
    parser.add_argument('--output', type=Path, default="pretrain",
                        help='path of model output directory')
    parser.add_argument('--gpu',action='store_true', default=False ,
                        help='to use GPU')
    parser.add_argument('--n_clusters', type=int, default=2048 ,
                        help='clusters number')

    args = parser.parse_args()

    checkpoint_dir = args.output
    dataset = args.dataset
    use_gpu = args.gpu
    n_clusters = args.n_clusters
    
    in_dirs = []
    for spk in os.listdir(dataset):
        if os.path.isdir(dataset/spk):
            print(f"found {spk}...")
            in_dir = dataset/spk
            in_dirs.append(in_dir)
    x = train_cluster(in_dirs, n_clusters,use_minibatch=False,verbose=False,use_gpu=use_gpu)

    checkpoint_path = checkpoint_dir / "semantic_codebook.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        x,
        checkpoint_path,
    )
    
