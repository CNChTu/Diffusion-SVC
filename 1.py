root_path = "D:/Diffusion-SVC/data/train/audio"
import os
from train_log.utils import traverse_dir
file_list = traverse_dir(root_path, extensions=["wav"], is_pure=True, is_sort=True, is_ext=True)
print(file_list)
with open("D:/Diffusion-SVC/wav.txt", "w") as f:
    f.write(root_path + "\n")
    for file in file_list:
        f.write(file + "\n")