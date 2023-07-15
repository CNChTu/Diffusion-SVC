import os
import pathlib

def getModels(directory: pathlib.Path | str):

    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)

    subdirectories = []

    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")

    for path in directory.iterdir():
        if path.is_dir():
            subdirectories.append(path)

    return subdirectories
