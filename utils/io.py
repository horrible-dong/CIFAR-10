import os

import torch
import ujson as json
from PIL import Image


def pil_loader(path, format=None):
    image = Image.open(path)
    if format is not None:
        image = image.convert(format)
    return image


def json_loader(path):
    with open(path, "r") as f:
        json_dict = json.load(f)
    return json_dict


def json_saver(json_dict, path, mode=0o777, overwrite=True, **kwargs):
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, **kwargs)
    os.chmod(path, mode)


def weights_saver(model, path, save_pos=None, state_dict=True, delete_keys=(), data_parallel=False, mode=0o777,
                  rename=False, overwrite=True):
    if os.path.exists(path):
        if rename:
            while os.path.exists(path):
                split_path = os.path.splitext(path)
                path = split_path[0] + "(1)" + split_path[1]
        elif overwrite:
            os.remove(path)
        else:
            raise FileExistsError
    if data_parallel:
        model = model.module
    if save_pos is not None:
        save_pos = save_pos.split(".")
        for child in save_pos:
            model = getattr(model, child)

    weights_dict = model.state_dict() if state_dict else model

    for key in delete_keys:
        if weights_dict.get(key) is not None:
            del weights_dict[key]

    torch.save(weights_dict, path)
    os.chmod(path, mode)
