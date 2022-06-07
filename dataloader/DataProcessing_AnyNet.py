import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess
import Imath
import OpenEXR
import matplotlib.pyplot as plt

from time import time

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    # source: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)

    exr_file = OpenEXR.InputFile(path)
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    try:
        channel = exr_file.channel('R', PIXEL_TYPE)
    except TypeError:
        channel = exr_file.channel('Y', PIXEL_TYPE)

    # print(type(channel))
    # channel = channel.clone()
    depth_np = np.frombuffer(channel, dtype=np.float32)
    depth_np.shape = (height, width)
    depth_np = depth_np.copy()
    # depth = torch.frombuffer(channel, dtype=torch.float32)  # produced warnings
    # depth = depth.reshape(width, height)  # height, width ??
    depth = torch.from_numpy(depth_np)
    return depth.clone()


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader, with_gt=True):

        assert (training and not with_gt) or not training # training without GT is non-sense for supervised ML
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.with_gt = with_gt

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        if self.with_gt:
            disp_L = self.disp_L[index]

        # 0000107_img.webp -> 107
        basename = os.path.basename(left)
        fid_str, _ = basename.split("_")
        fid = int(fid_str)

        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.with_gt:
            dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            # left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            # right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL, dtype=np.float32)
            # dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL, fid
        else:
            w, h = left_img.size

            # left_img = left_img.crop((w - 1232, h - 368, w, h))
            # right_img = right_img.crop((w - 1232, h - 368, w, h))
            w1, h1 = left_img.size

            if self.with_gt:
                # dataL = dataL.crop((w - 1232, h - 368, w, h))
                dataL = np.ascontiguousarray(dataL, dtype=np.float32)

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            if self.with_gt:
                return left_img, right_img, dataL, fid
            else:
                return left_img, right_img, fid

    def __len__(self):
        return len(self.left)
