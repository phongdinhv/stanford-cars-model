import cv2
import torch
from scipy import io as mat_io
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from base import BaseDataLoader


class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, data_dir, metas, resize_width, resize_height, limit):

        self.data_dir = data_dir
        self.data = []
        self.target = []

        self.to_tensor = transforms.ToTensor()
        self.mode = 'train'

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break
            image = io.imread(data_dir + img_[5][0])

            if len(image.shape) == 2: # this is gray image
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            img_resized = cv2.resize(image, (resize_width, resize_height))

            self.data.append(img_resized)
            if self.mode == 'train':
                self.target.append(img_[4][0][0])

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
        ])

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.transform(self.data[idx]), torch.tensor(self.target[idx]-1, dtype=torch.long)
        elif self.mode == 'val' or self.mode == 'test':
            return self.to_tensor(self.data[idx]), torch.tensor(self.target[idx] - 1, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class CarsDataLoader(BaseDataLoader):
    """
    Cars data loading
    """
    def __init__(self, data_dir, metas, batch_size, resize_width,
                 resize_height, shuffle=True, validation_split=0.0,
                 num_workers=1, limit=None):

        self.dataset = CarsDataset(data_dir, metas, resize_width, resize_height, limit)

        super(CarsDataLoader, self).__init__(self.dataset, batch_size, shuffle,
                                             validation_split, num_workers)


