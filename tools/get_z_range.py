import os
import numpy as np
import yaml
import torch
import argparse


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


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
        "-m",
        "--model",
        type=str,
        default=None,
        required=False,
        help="val or train; default is train, it means get z range from train data path")
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    train_or_val = cmd.model
    if train_or_val is None:
        train_or_val = 'train'
    print(f'  [INFO] train_or_val: {train_or_val}')

    # load config
    args = load_config(cmd.config)
    print(f'  [INFO] args: {args}')
    print(f'  [INFO] config: {cmd.config}')
    print(f'  [INFO] args.data.{train_or_val}_data_path: {args.data.train_path}')

    # get all file path
    data_path = args.data.train_path if train_or_val == 'train' else args.data.valid_path
    path_srcdir = os.path.join(data_path, 'mel')
    filelist = traverse_dir(
        path_srcdir,
        extensions=['npy'],
        is_pure=True,
        is_sort=True,
        is_ext=True)
    # get z range
    z_max = None
    z_max_name = None
    z_min = None
    z_min_name = None
    z_total_std = 0.
    m_max = None
    m_max_name = None
    m_min = None
    m_min_name = None
    m_total_std = 0.
    logs_max = None
    logs_max_name = None
    logs_min = None
    logs_min_name = None
    logs_total_std = 0.
    for file in filelist:
        path_specfile = os.path.join(path_srcdir, file)
        # load spec
        spec = np.load(path_specfile, allow_pickle=True)
        spec = torch.from_numpy(spec).float()
        m = spec.transpose(-1, 0)[:1].transpose(-1, 0).squeeze(-1)
        logs = spec.transpose(-1, 0)[1:].transpose(-1, 0).squeeze(-1)
        z = m + torch.randn_like(m) * torch.exp(logs)

        z_total_std += z.std()
        m_total_std += m.std()
        logs_total_std += logs.std()

        if z_max is None:
            z_max = z.max()
            z_max_name = path_specfile
        else:
            z_max = max(z_max, z.max())
            z_max_name = path_specfile
        if z_min is None:
            z_min = z.min()
            z_min_name = path_specfile
        else:
            z_min = min(z_min, z.min())
            z_min_name = path_specfile

        if m_max is None:
            m_max = m.max()
            m_max_name = path_specfile
        else:
            m_max = max(m_max, m.max())
            m_max_name = path_specfile
        if m_min is None:
            m_min = m.min()
            m_min_name = path_specfile
        else:
            m_min = min(m_min, m.min())
            m_min_name = path_specfile

        if logs_max is None:
            logs_max = logs.max()
            logs_max_name = path_specfile
        else:
            logs_max = max(logs_max, logs.max())
            logs_max_name = path_specfile
        if logs_min is None:
            logs_min = logs.min()
            logs_min_name = path_specfile

        print(f"  [INFO] path/file: {path_specfile}")
        print(f"  [INFO]     >z: {z.max()}, {z.min()}; std: {z.std()}; mean: {z.mean()}")
        print(f"  [INFO]     >m: {m.max()}, {m.min()}; std: {m.std()}; mean: {m.mean()}")
        print(f"  [INFO]     >logs: {logs.max()}, {logs.min()}; std: {logs.std()}; mean: {logs.mean()}")
    print("END")
    print(f"  [INFO] z_max: {z_max}, z_max_name: {z_max_name}")
    print(f"  [INFO] z_min: {z_min}, z_min_name: {z_min_name}")
    print(f"  [INFO] z_total_std: {z_total_std / len(filelist)}")
    print(f"  [INFO] m_max: {m_max}, m_max_name: {m_max_name}")
    print(f"  [INFO] m_min: {m_min}, m_min_name: {m_min_name}")
    print(f"  [INFO] m_total_std: {m_total_std / len(filelist)}")
    print(f"  [INFO] logs_max: {logs_max}, logs_max_name: {logs_max_name}")
    print(f"  [INFO] logs_min: {logs_min}, logs_min_name: {logs_min_name}")
    print(f"  [INFO] logs_total_std: {logs_total_std / len(filelist)}")





