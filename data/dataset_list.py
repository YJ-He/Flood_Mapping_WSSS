import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''for 3 bands input data'''
rgb_mean = (0.4353, 0.4452, 0.4131)
rgb_std = (0.2044, 0.1924, 0.2013)

class MyDataset(Dataset):
    def __init__(self,
                 config,
                 args,
                 subset):
        super(MyDataset, self).__init__()
        assert subset == 'train' or subset == 'val' or subset == 'test'

        self.args = args
        self.config = config
        self.root = args.input
        self.subset = subset
        self.data = self.config.data_folder_name  # image
        self.target = self.config.target_folder_name  # label
        self.class_names = ['other', 'flood']
        self.mapping = {
            0: 0,
            255: 1,
        }

        self.data_list = []
        self.target_list = []
        # image
        with open(os.path.join(self.root, subset + '_image.txt'), 'r') as f:
            for line in f:
                if line.strip('\n') != '':
                    self.data_list.append(line.strip('\n'))
        # label
        if not self.args.only_prediction:
            with open(os.path.join(self.root, subset + '_label.txt'), 'r') as f:
                for line in f:
                    if line.strip('\n') != '':
                        self.target_list.append(line.strip('\n'))
            assert len(self.data_list) == len(self.target_list)

    def mask_to_class(self, mask):
        """
        Encode class to number
        """
        for k in self.mapping:
            mask[mask == k] = self.mapping[k]
        return mask

    def train_transforms(self, image, mask):
        """
        Preprocessing and augmentation on training data (image and label)
        """
        in_size = self.config.input_size
        train_transform = A.Compose(
            [
                A.Resize(in_size, in_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.RandomRotate90(p=0.8),
                A.Transpose(p=0.8),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = train_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        mask = mask.float()
        return image, mask

    def untrain_transforms(self, image, mask):
        """
        Preprocessing on val or test data (image and label)
        """
        untrain_transform = A.Compose(
            [
                A.Resize(self.config.eval_size, self.config.eval_size, interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = untrain_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        mask = mask.float()
        return image, mask

    def untrain_transforms1(self, image):
        """
        Preprocessing on untrain data (image)
        """
        untrain_transform = A.Compose(
            [
                A.Resize(self.config.eval_size, self.config.eval_size),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = untrain_transform(image=image)
        image = transformed["image"]
        return image

    def __getitem__(self, index):
        image = cv2.imread(self.data_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not self.args.only_prediction:
            mask = np.array(Image.open(self.target_list[index])).astype(np.uint8)

        if self.subset == 'train':
            if not self.args.is_test:
                t_datas, t_targets = self.train_transforms(image, mask)
            else:
                t_datas, t_targets = self.untrain_transforms(image, mask)
            return t_datas, t_targets, self.data_list[index]
        elif self.subset == 'val':
            t_datas, t_targets = self.untrain_transforms(image, mask)
            return t_datas, t_targets, self.data_list[index]
        elif self.subset == 'test':
            if not self.args.only_prediction:
                t_datas, t_targets = self.untrain_transforms(image, mask)
                return t_datas, t_targets, self.data_list[index]
            else:
                t_datas = self.untrain_transforms1(image)
                return t_datas, self.data_list[index]

    def __len__(self):

        return len(self.data_list)
