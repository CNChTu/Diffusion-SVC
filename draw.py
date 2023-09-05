import os
import argparse
import random
from tqdm import tqdm
import shutil

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--trainrate",
        type=int,
        default=0.7,
        help="train set rate")
    
    
    return parser.parse_args(args=args, namespace=namespace)

# 如果文件夹不存在创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    train_rate = cmd.trainrate
    test_rate = 1 - train_rate

    for i in list(filter(lambda x: os.path.isdir(f"./dataset_raw/{x}") ,os.listdir('./dataset_raw'))):
        speaker_path = os.path.join('./dataset_raw', i)
        # 读取dataset_raw文件夹下的所有.wav文件
        file_list = list(filter(lambda x: x.endswith('.wav') ,os.listdir(speaker_path)))
        file_list = list(map(lambda x: os.path.join("./dataset_raw",i,x), file_list))
        
        # 打乱文件列表
        random.shuffle(file_list)
        train_list = file_list[:int(len(file_list)*train_rate)]
        val_list = file_list[int(len(file_list)*train_rate):]
    
        train_path = f"./data/train/audio/{i}/"
        mkdir(train_path)
        val_path = f"./data/val/audio/{i}/"
        mkdir(val_path)

        for j in tqdm(train_list, desc=f"copying {i} train set"):
            shutil.copy(j, train_path)
        for j in tqdm(val_list, desc=f"copying {i} val set"):
            shutil.copy(j, val_path)
            