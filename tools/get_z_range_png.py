import os
import numpy as np
import tqdm
import yaml
import torch
import argparse
import tqdm


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
    parser.add_argument(
        "-min",
        "--min",
        type=float,
        default=-10.0,
        required=False,
        help="z_min")
    parser.add_argument(
        "-max",
        "--max",
        type=float,
        default=10.0,
        required=False,
        help="z_max")
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

    # get all file path
    data_path = args.data.train_path if train_or_val == 'train' else args.data.valid_path
    print(f'  [INFO] args.data.{train_or_val}_data_path: {data_path}')
    path_srcdir = os.path.join(data_path, 'mel')
    filelist = traverse_dir(
        path_srcdir,
        extensions=['npy'],
        is_pure=True,
        is_sort=True,
        is_ext=True)

    # 定义0.001一级，范围z的直方图的表，tensor
    z_min = cmd.min
    z_max = cmd.max
    bins = int((z_max - z_min) * 1000 + 1)
    hist = torch.zeros(bins)
    # 遍历所有文件
    for file in tqdm.tqdm(filelist):
        path_specfile = os.path.join(path_srcdir, file)
        # load spec
        spec = np.load(path_specfile, allow_pickle=True)
        spec = torch.from_numpy(spec).float()
        m = spec.transpose(-1, 0)[:1].transpose(-1, 0).squeeze(-1)
        logs = spec.transpose(-1, 0)[1:].transpose(-1, 0).squeeze(-1)
        z = m + torch.randn_like(m) * torch.exp(logs)
        # 计算直方图
        # clip将z限制在-10到10之间, 超出部分视为-10或10
        z_c = z.clamp(z_min, z_max)
        hist += torch.histc(z_c, bins=bins, min=z_min, max=z_max)

    # 计算直方图的累积分布函数
    # 从左到右累积
    cdf = torch.cumsum(hist, dim=0)
    # total count
    cdf_total = cdf[-1]
    # 找到0.001和0.999的位置
    z_find_min = z_min
    z_find_max = z_max
    for i in range(bins):
        if cdf[i] > (0.001 * cdf_total):
            z_find_min = i / 1000 - 10
            break
    for i in range(bins):
        if cdf[i] > (0.999 * cdf_total):
            z_find_max = i / 1000 - 10
            break
    print(f'  [INFO] z_min(0.001): {z_find_min}, z_max(0.009): {z_find_max}')
    # 刨去两端极值的数据占比
    _sum = (cdf[-2] - cdf[0])
    _sum = _sum / cdf_total
    print(f'  [INFO] sum(min > max): {_sum}')
    import matplotlib.pyplot as plt
    # 画图
    plt.figure()
    plt.plot(torch.arange(z_min, z_max + 0.001, 0.001), hist)
    plt.xlabel('z')
    plt.ylabel('count')
    plt.title('z range')
    plt.grid()
    plt.savefig(os.path.join(data_path, 'z_range.png'))
    plt.close()
