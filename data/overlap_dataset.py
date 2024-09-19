import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch


class OverlapDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.

    Instead of providing only one image from each domain, this dataset loads larger images (size 728x728) from each domain,
    crops out a patch of size (512-p squared) and splits the image into 4 overlapping patches of size 256x256, providing 4 images from each domain.
    Those images have an overlap of p pixels. This allows an additional consistency loss to be used in
    training, which is the L1 loss between the overlapping regions of fake/reconstructed images. Intuitively,
    this loss reflects the fact that generated images with spatial overlap should be consistent with each other.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        normalize = False if self.opt.stain_task else True
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1), A_or_B='A', normalize=normalize)
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1), A_or_B='B', normalize=normalize)
        self.overlap = opt.overlap

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches or self.opt.lambda_L1 > 0.0:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # crop out a patch of size (512-p squared) and split the image into 4 overlapping patches of size 256x256
        w, h = A_img.size
        crop_size = 512 - self.overlap
        crop_top_left = (random.randint(0, w - crop_size), random.randint(0, h - crop_size))
        big_A = A_img.crop((crop_top_left[0], crop_top_left[1], crop_top_left[0] + crop_size, crop_top_left[1] + crop_size))
        big_B = B_img.crop((crop_top_left[0], crop_top_left[1], crop_top_left[0] + crop_size, crop_top_left[1] + crop_size))
        ims_A, ims_B = [], []
        for x in [crop_top_left[0], crop_top_left[0] + 256 - self.overlap]:
            for y in [crop_top_left[1], crop_top_left[1] + 256 - self.overlap]:
                ims_A.append(self.transform_A(A_img.crop((x, y, x + 256, y + 256))))
                ims_B.append(self.transform_B(B_img.crop((x, y, x + 256, y + 256))))

        # stack along dim 0 to get a tensor of shape (4, 3, 256, 256)
        A = torch.stack(ims_A, dim=0)
        B = torch.stack(ims_B, dim=0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'big_A': self.transform_A(big_A), 'big_B': self.transform_B(big_B)}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)