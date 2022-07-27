from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
from utils.image_utils import randFlipStereoImage
import glob



class StereoDataset_new(Dataset):
    def __init__(self, stereo_dir_2012, stereo_dir_2015, isTrainingData=True, randomFlip=False, RandomCrop=False,
                 colorJitter=False, transform=transforms.ToTensor()):
        """
        KITTI Stereo Image Pairs dataset.

        Args:
            stereo_dir_2012 (string): Directory with stereo images from 2012 kitti dataset.
            stereo_dir_2015 (string): Directory with stereo images from 2015 kitti dataset.
                note1: inside each stereo_dir, expected sub folders: testing & training, and sub-sub folders image_2,image_3
                note2: matching images from the two folders(_2, _3) are assumed to have the same names.
                note3: assumes *png* images
            transform (optional): Optional transform to be applied on the images.
        """
        subFolder = 'training' if isTrainingData else 'testing'
        stereo2012_dir = os.path.join(stereo_dir_2012, subFolder, 'image_2')
        stereo2015_dir = os.path.join(stereo_dir_2015, subFolder, 'image_2')
        stereo2012_path_list = glob.glob(os.path.join(stereo2012_dir, '*png'))
        stereo2015_path_list = glob.glob(os.path.join(stereo2015_dir, '*png'))
        '''
        For the exact dataset, Ayzik and Avidan Github repo.
        '''
        self.stereo_image_2_path_list = stereo2012_path_list + stereo2015_path_list
        self.transform = transform
        self.randomFlip = randomFlip
        self.RandomCrop = RandomCrop
        self.colorJitter = colorJitter

    def __len__(self):
        return len(self.stereo_image_2_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_stereo1 = Image.open(self.stereo_image_2_path_list[idx])
        img_stereo2_name = self.stereo_image_2_path_list[idx].replace('image_2', 'image_3')
        try:
            img_stereo2 = Image.open(img_stereo2_name)
        except ValueError:
            raise ValueError("Error when reading stereo2-image. Image names in both folder should be the same.")

        if self.transform:
            img_stereo1 = self.transform(img_stereo1)
            img_stereo2 = self.transform(img_stereo2)

        if self.RandomCrop:
            i, j, h, w = transforms.RandomCrop.get_params(img_stereo1, output_size=(320, 960))  # multiplication of 32
            img_stereo1 = img_stereo1[:, i:i+h, j:j+w]
            img_stereo2 = img_stereo2[:, i:i+h, j:j+w]

        if self.colorJitter:
            tsfm = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)])
            img_stereo1 = tsfm(torch.cat((img_stereo1[None,:],img_stereo2[None,:]), 0))
            img_stereo2 = img_stereo1[1,:,:,:]
            img_stereo1 = img_stereo1[0, :, :, :]

        if self.randomFlip:
            # convert to numpy, do random flip, convert back to tensor
            im1_np = img_stereo1.permute(1, 2, 0).detach().cpu().numpy()
            im2_np = img_stereo2.permute(1, 2, 0).detach().cpu().numpy()
            img_stereo1, img_stereo2 = randFlipStereoImage(im1_np, im2_np)
            img_stereo1 = torch.tensor(img_stereo1.copy()).permute(2, 0, 1)
            img_stereo2 = torch.tensor(img_stereo2.copy()).permute(2, 0, 1)

        return img_stereo1, img_stereo2





