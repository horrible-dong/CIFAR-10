import numpy as np
import torch
from torchvision import transforms as tfs


class Cutout:
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_transforms(dataset, **kwargs):
    if dataset in ["cifar10", "cifar100"]:
        transforms = {
            "train": tfs.Compose([
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                Cutout(n_holes=1, length=16)
            ]),
            "test": tfs.Compose([
                tfs.Resize(32),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }

    else:
        raise ValueError

    return transforms
