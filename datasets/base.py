import os.path

from torch.utils.data import Dataset

from utils.io import pil_loader

__all__ = ["BaseDataset", "BaseDatasetGroup"]


def default_loader(path, format="RGB"):
    return pil_loader(path, format)


class BaseDataset(Dataset):
    def __init__(self, root, mode, eval=False, loader=default_loader, transforms=None):
        self.root = os.path.expanduser(root)
        self.mode = mode
        self.eval = eval
        self.loader = loader
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError


class BaseDatasetGroup(Dataset):
    """
    root_mode_group (list): [[root_0, mode_0], [root_1, mode_1]... ]
    """

    def __init__(self, root_mode_group: list, loader=default_loader, transforms=None):
        self.root_mode_group = [[os.path.expanduser(root), mode] for root, mode in root_mode_group]
        self.num_datasets = len(root_mode_group)
        self.loader = loader
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError
