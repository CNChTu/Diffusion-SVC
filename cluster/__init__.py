import torch
from sklearn.cluster import KMeans


def get_cluster_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    km = KMeans(checkpoint["n_features_in_"])
    km.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
    km.__dict__["_n_threads"] = checkpoint["_n_threads"]
    km.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"]
    return km

def get_cluster_result(model, x):
    """
        x: np.array [t, 256]
        return cluster class result
    """
    return model.predict(x)

def get_cluster_center_result(model, x):
    """x: np.array [t, 256]"""
    predict = model.predict(x)
    return model.cluster_centers_[predict]

def get_center(model, token):
    return model.cluster_centers_[token]
