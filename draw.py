import os
import argparse
import random
from tqdm import tqdm

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

    for i in os.listdir('./dataset_raw'):
        speaker_path = os.path.join('./dataset_raw', i)
        # 读取dataset_raw文件夹下的所有.wav文件
        file_list = list(filter(lambda x: x.endswith('.wav') ,os.listdir('./dataset_raw')))
        file_list = list(map(lambda x: os.path.join("./dataset_raw",x), file_list))
        
        # 打乱文件列表
        file_list = random.shuffle(file_list)
    
        train_list = file_list[:int(len(file_list)*train_rate)]
        val_list = file_list[int(len(file_list)*train_rate):]
    
        train_path = f"./dataset/train/audio/{i}"
        mkdir(train_path)
        val_path = f"./dataset/val/audio/{i}"
        mkdir(val_path)

        for j in tqdm(train_list, prefix=f"copying {i} train set"):
            os.system(f"cp {j} {train_path}")
        for j in tqdm(val_list, prefix=f"copying {i} val set"):
            os.system(f"cp {j} {val_path}")
            
