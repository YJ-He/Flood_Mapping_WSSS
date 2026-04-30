import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FloodDatasetTiff(Dataset):
    """Dataset for multi-band TIFF images with PNG labels.

    Reads from directory structure:
        {root}/{train,val,test}/{images,labels}/
    Images: multi-band .tif files loaded via tifffile
    Labels: single-channel .png files with values 0 (background) and 1 (flood)
    """

    def __init__(self, config, args, subset):
        super().__init__()
        assert subset in ['train', 'val', 'test']

        self.args = args
        self.config = config
        self.subset = subset
        self.class_names = ['background', 'flood']

        self.img_mean = self._parse_tuple(config.img_mean)
        self.img_std = self._parse_tuple(config.img_std)

        img_dir = os.path.join(args.input, subset, 'images')
        label_dir = os.path.join(args.input, subset, 'labels')

        img_exts = ('.tif', '.tiff')
        self.data_list = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(img_exts)
        ])

        self.target_list = []
        if not self.args.only_prediction:
            label_exts = ('.png', '.tif', '.tiff')
            self.target_list = sorted([
                os.path.join(label_dir, f) for f in os.listdir(label_dir)
                if f.lower().endswith(label_exts)
            ])
            assert len(self.data_list) == len(self.target_list), \
                f"Image/label count mismatch: {len(self.data_list)} vs {len(self.target_list)}"

        print(f"  [{subset}] loaded {len(self.data_list)} samples from {img_dir}")

    @staticmethod
    def _parse_tuple(s):
        if not s or s.strip() == '':
            return None
        return tuple(float(x.strip()) for x in s.split(','))

    @staticmethod
    def _load_tiff(filepath):
        img = tifffile.imread(filepath)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.ndim == 3 and img.shape[0] < img.shape[2]:
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.float32)
        return img

    def _get_transforms(self, is_train):
        tfms = []
        if is_train:
            size = self.config.input_size
            tfms.extend([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
            ])
        else:
            size = self.config.eval_size
            tfms.append(A.Resize(size, size))

        if self.img_mean is not None and self.img_std is not None:
            tfms.append(A.Normalize(
                mean=self.img_mean,
                std=self.img_std,
                max_pixel_value=1.0,
            ))

        tfms.append(ToTensorV2())
        return A.Compose(tfms)

    def __getitem__(self, index):
        image = self._load_tiff(self.data_list[index])

        is_train = (self.subset == 'train' and not self.args.is_test)
        transform = self._get_transforms(is_train)

        if not self.args.only_prediction:
            label_path = self.target_list[index]
            if label_path.lower().endswith('.png'):
                mask = np.array(Image.open(label_path)).astype(np.uint8)
            else:
                mask = tifffile.imread(label_path).astype(np.uint8)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            transformed = transform(image=image, mask=mask)
            return transformed['image'], transformed['mask'].float(), self.data_list[index]
        else:
            transformed = transform(image=image)
            return transformed['image'], self.data_list[index]

    def __len__(self):
        return len(self.data_list)
