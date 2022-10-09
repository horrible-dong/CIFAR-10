import os

import datasets
from utils.common import flatten_list
from .transforms_getter import get_transforms


def get_dataset(data_root, dataset, mode=None, vis=False, extra=()):
    dataset_path = os.path.join(data_root, dataset)
    assert mode in ["train", "val", "test", None]

    train = mode == "train" and not vis
    val = mode == "val" and not vis
    test = mode == "test" and not vis

    if dataset in ["cifar10"]:
        scale_factor = 1.0
        custom = {
            "scale_factor": scale_factor,
            "image_size": (28, 28)
        }
        transforms = get_transforms(dataset, **custom)
        train_dataset = datasets.CIFAR10(root=dataset_path,
                                         mode="train",
                                         transforms=transforms,
                                         download=True) if train else None
        val_dataset = datasets.CIFAR10(root=dataset_path,
                                       mode="test",
                                       transforms=transforms,
                                       download=True) if train or val else None
        test_dataset = datasets.CIFAR10(root=dataset_path,
                                        mode="test",
                                        transforms=transforms,
                                        download=True) if test else None
        vis_dataset = datasets.CIFAR10(root=dataset_path,
                                       mode=mode,
                                       transforms=transforms,
                                       download=True) if vis else None
    else:
        raise ValueError(f"Dataset {dataset} is not registered.")

    if vis:
        mode = "vis"

    options = {
        "dataset": {
            "train": [train_dataset, val_dataset],
            "val": val_dataset,
            "test": test_dataset,
            "vis": vis_dataset
        },

        "extra": {
            "custom": custom,
            "transforms": transforms
        }
    }

    return_list = []

    if mode is not None:
        return_list.append(options["dataset"][mode])
    for item in extra:
        return_list.append(options["extra"][item])

    return_list = flatten_list(return_list)

    if len(return_list) == 0:
        raise Exception

    return return_list[0] if len(return_list) == 1 else return_list
