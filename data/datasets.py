import math
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class OpenImages(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, image_dims, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self, scale=0, H=0, W=0):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
            transforms.ToTensor(),
            transforms.RandomCrop((self.im_height, self.im_width), pad_if_needed=True),
            # transforms.Resize((self.im_height, self.im_width)),
        ]

        # if self.normalize is True:
        #     transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, dataset_path):
        self.data_dir = dataset_path
        self.imgs = []
        self.imgs += glob(os.path.join(self.data_dir, '*.jpg'))
        self.imgs += glob(os.path.join(self.data_dir, '*.png'))
        self.imgs.sort()
        self.transform = transforms.Compose([
            # transforms.RandomCrop((384, 384), pad_if_needed=True),
            transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        img = self.transform(image)
        return img

    def __len__(self):
        return len(self.imgs)


def get_loader(train_dir, test_dir, num_workers, batch_size):
    train_dataset = OpenImages((3, 384, 384), train_dir)
    test_dataset = Datasets(test_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               batch_size=batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)

    return train_loader, test_loader


def get_dataset(train_dir, test_dir):
    train_dataset = OpenImages((3, 384, 384), train_dir)
    test_dataset = Datasets(test_dir)
    return train_dataset, test_dataset


def get_test_loader(test_dir):
    test_dataset = Datasets(test_dir)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False)
    return test_loader
