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
    def __init__(self, data_dir, train_metas, resize_width, resize_height, transform, limit):

        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.mode = 'train'

        if not isinstance(train_metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(train_metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break
            image = io.imread(data_dir + img_[5][0])

            if len(image.shape) == 2: # this is gray image
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            img_resized = cv2.resize(image, (resize_height, resize_width))

            self.data.append(img_resized)
            self.target.append(img_[4][0][0])

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
    def __init__(self, data_dir, train_metas, batch_size, resize_width,
                 resize_height, shuffle=True, validation_split=0.0,
                 num_workers=1, limit=None):
        trsfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomGrayscale(),
            transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.dataset = CarsDataset(data_dir, train_metas, resize_width, resize_height, trsfm, limit)

        super(CarsDataLoader, self).__init__(self.dataset, batch_size, shuffle,
                                             validation_split, num_workers)


