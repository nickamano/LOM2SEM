import pdb

import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
import pickle
import torch
import random
import os


class ImagePairDataset(Dataset):
    """A dataset consists of pairs of images. Images in the same pair come from
    different folders with the same name.

    Args:
        root: Root directory path.
        size: The size of input after resizing.
        folder_A: Folder of image type A.
        folder_A: Folder of image type B.
        split: A text file listing the name of images.
    """
    def __init__(self, root, size, folder_A, folder_B, split):
        super(ImagePairDataset, self).__init__()
        if split is None:
            # list all tif files
            self.img_names = [f for f in os.listdir(f"{root}/{folder_A}") if f.endswith('.tif')]
        else:
            self.img_names = np.genfromtxt(f"{root}/split/{split}",
                                           dtype=str, delimiter='\n', ndmin=1)
        self.folder_A = f"{root}/{folder_A}"
        self.folder_B = f"{root}/{folder_B}"
        self.transform = T.Compose([
            T.Resize(size, antialias=True),
            # T.Pad(size[0]//4, padding_mode='reflect'),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # T.RandomRotation(degrees=20, interpolation=T.InterpolationMode.BILINEAR),
            # T.CenterCrop(448),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        with open(f"{root}/split/labels.pkl", 'rb') as f:
            self.labels = pickle.load(f)

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_A_path = f'{self.folder_A}/{img_name}'
        img_B_path = f'{self.folder_B}/{img_name}'
        with open(img_A_path, 'rb') as f:
            img_A = Image.open(f)
            img_A = img_A.convert('RGB')
            img_A = T.ToTensor()(img_A)
        with open(img_B_path, 'rb') as f:
            img_B = Image.open(f)
            img_B = img_B.convert('RGB')
            img_B = T.ToTensor()(img_B)
        both = torch.cat((img_A.unsqueeze(0), img_B.unsqueeze(0)), dim=0)
        both = self.transform(both)
        img_A, img_B = both[0], both[1]
        # label = self.labels[img_name]
        label = 0
        return img_A, img_B[0][None,:,:], label

    def __len__(self):
        return len(self.img_names)

class AdaINImagePairDataset(ImagePairDataset):
    """A dataset consists of pairs of images. Images in the same pair come from
    different folders with the same name.

    Args:
        root: Root directory path.
        size: The size of input after resizing.
        folder_A: Folder of image type A.
        folder_A: Folder of image type B.
        folder_style: Folder of style images
        split: A text file listing the name of images.
    """
    def __init__(self, root, size, folder_A, folder_B, style_folder, split):
        super(AdaINImagePairDataset, self).__init__(root, size, folder_A, folder_B, split)
        self.folder_style = f"{root}/{style_folder}"
        self.style_img_names = [f for f in os.listdir(self.folder_style) if f.endswith('.tif')]
        self.transform = T.Compose([
            T.Resize(224, antialias=True),
            # T.Pad(size[0]//4, padding_mode='reflect'),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # T.RandomRotation(degrees=20, interpolation=T.InterpolationMode.BILINEAR),
            # T.CenterCrop(448),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_style = random.choice(self.style_img_names)
        img_A_path = f'{self.folder_A}/{img_name}'
        img_B_path = f'{self.folder_B}/{img_name}'
        img_style_path = f'{self.folder_style}/{img_style}'
        with open(img_A_path, 'rb') as f:
            img_A = Image.open(f)
            img_A = img_A.convert('RGB')
            img_A = T.ToTensor()(img_A)
        with open(img_B_path, 'rb') as f:
            img_B = Image.open(f)
            img_B = img_B.convert('RGB')
            img_B = T.ToTensor()(img_B)
        with open(img_style_path, 'rb') as f:
            img_style = Image.open(f)
            img_style = img_style.convert('RGB')
            img_style = T.ToTensor()(img_style)
        both = torch.cat((img_A.unsqueeze(0), img_B.unsqueeze(0), img_style.unsqueeze(0)), dim=0)
        both = self.transform(both)
        img_A, img_B = both[0], both[1], both[2]
        # label = self.labels[img_name]
        label = 0
        return img_A, img_B[0][None,:,:], img_style[0][None,:,:], label

    def __len__(self):
        return len(self.img_names)

class MyRotateTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def get_list_of_ops(args):
    if args is None: return []
    ops = []
    for func_name, params in args:
        if func_name == 'MyRotateTransform':
            ops.append(MyRotateTransform([0, 90, 180, 270]))
            continue
        if func_name == 'RandomRotation':
            ops.append(T.RandomRotation(degrees=params['degrees'],
                                        interpolation=T.InterpolationMode.BILINEAR))
            continue
        func = getattr(T, func_name)
        ops.append(func(**params))
    return ops


class P2PHDDataset(ImagePairDataset):
    def __init__(self, root: str, size: tuple, folder_A: str, folder_B: str, split: str, args_aug: list):
        super(P2PHDDataset, self).__init__(root, size, folder_A, folder_B, split)
        args_aug_both = [(k, v) for k, v in args_aug if k not in ['Grayscale', 'ColorJitter']]
        args_aug_A_only = [(k, v) for k, v in args_aug if k in ['Grayscale', 'ColorJitter']]
        self.transform_both = T.Compose([
            T.Resize(size, antialias=True),
            *get_list_of_ops(args_aug_both),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_A_only = T.Compose([*get_list_of_ops(args_aug_A_only)])

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_A_path = f'{self.folder_A}/{img_name}'
        img_B_path = f'{self.folder_B}/{img_name}'
        with open(img_A_path, 'rb') as f:
            img_A = Image.open(f)
            img_A = img_A.convert('RGB')
            img_A = self.transform_A_only(img_A)
            img_A = T.ToTensor()(img_A)
        with open(img_B_path, 'rb') as f:
            img_B = Image.open(f)
            img_B = img_B.convert('RGB')
            img_B = T.ToTensor()(img_B)
        both = torch.cat((img_A.unsqueeze(0), img_B.unsqueeze(0)), dim=0)
        both = self.transform_both(both)
        img_A, img_B = both[0], both[1]
        # label = self.labels[img_name]
        label = 0
        return img_A, img_B[0][None,:,:], label

class AdaINDataset(ImagePairDataset):
    def __init__(self, root: str, size: tuple, folder_A: str, folder_B: str, split: str, args_aug: list):
        super(AdaINDataset, self).__init__(root, size, folder_A, folder_B, split)
        args_aug_both = [(k, v) for k, v in args_aug if k not in ['Grayscale', 'ColorJitter']]
        args_aug_A_only = [(k, v) for k, v in args_aug if k in ['Grayscale', 'ColorJitter']]
        self.transform_both = T.Compose([
            T.Resize(224, antialias=True),
            *get_list_of_ops(args_aug_both),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_A_only = T.Compose([*get_list_of_ops(args_aug_A_only)])

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_style = random.choice(self.img_names)
        img_A_path = f'{self.folder_A}/{img_name}'
        img_B_path = f'{self.folder_B}/{img_name}'
        img_style_path = f'{self.folder_B}/{img_style}'
        with open(img_A_path, 'rb') as f:
            img_A = Image.open(f)
            img_A = img_A.convert('RGB')
            img_A = self.transform_A_only(img_A)
            img_A = T.ToTensor()(img_A)
        with open(img_B_path, 'rb') as f:
            img_B = Image.open(f)
            img_B = img_B.convert('RGB')
            img_B = T.ToTensor()(img_B)
        with open(img_style_path, 'rb') as f:
            img_style = Image.open(f)
            img_style = img_style.convert('RGB')
            img_style = T.ToTensor()(img_style)
        both = torch.cat((img_A.unsqueeze(0), img_B.unsqueeze(0), img_style.unsqueeze(0)), dim=0)
        both = self.transform_both(both)
        img_A, img_B, img_style = both[0], both[1], both[2]
        # label = self.labels[img_name]
        label = 0
        return img_A, img_B, img_style, label

class SingleFullImageDataset(Dataset):
    def __init__(self, root: str, size: int, folder_A: str, folder_B: str, img_name: str, scale=4):
        super(SingleFullImageDataset, self).__init__()
        img_A_path = f'{root}/{folder_A}/{img_name}'
        img_B_path = f'{root}/{folder_B}/{img_name}'
        # transform = T.Compose([
        #     T.Grayscale(num_output_channels=3),
        #     T.ToTensor()])
        self.transform_A = T.Compose([
            # T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_B = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        with open(img_A_path, 'rb') as f:
            img_A = Image.open(f)
            img_A = img_A.resize((img_A.size[0]//scale, img_A.size[1]//scale))
            self.img_size = img_A.size
            img_A = img_A.convert('RGB')
            self.img_A = self.transform_A(img_A)
        with open(img_B_path, 'rb') as f:
            img_B = Image.open(f)
            img_B = img_B.convert('RGB')
            self.img_B = self.transform_B(img_B)
        self.n_row = img_A.size[1] // size + 1
        self.n_col = img_A.size[0] // size + 1
        self.row_indices = np.arange(self.n_row) * size
        self.col_indices = np.arange(self.n_col) * size
        self.row_indices[-1] = img_A.size[1]-size
        self.col_indices[-1] = img_A.size[0]-size
        self.crop_indices = np.array(np.meshgrid(self.row_indices, self.col_indices)).T.reshape(-1, 2)
        self.size = size

    def __getitem__(self, index: int):
        row, col = self.crop_indices[index]
        img_A = T.functional.crop(self.img_A, row, col, self.size, self.size)
        img_B = T.functional.crop(self.img_B, row, col, self.size, self.size)
        return img_A, img_B[0][None,:,:]

    def stitch(self, crops):
        print(self.img_size)
        out = np.zeros((self.img_size[1], self.img_size[0]))
        for crop, (row, col) in zip(crops, self.crop_indices):
            out[row:row+self.size, col:col+self.size] = crop
        return out

    def __len__(self):
        return len(self.crop_indices)