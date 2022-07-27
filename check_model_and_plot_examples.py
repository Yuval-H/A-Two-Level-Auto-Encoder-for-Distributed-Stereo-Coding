import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import PIL
import glob
import numpy as np
from model import DSC_stereo_compression
from dataset import StereoDataset_new
from utils.image_utils import msssim_x_vs_rec
from torch.utils.data import DataLoader


# Load model
pretrained_model_path = 'model_weights.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DSC_stereo_compression()

checkpoint = torch.load(pretrained_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

#Load test dataset
# should be paths to KITTI dataset folder
stereo_dir_2012 = '/home/access/disk/dev/data_sets/kitti/Sharons datasets/data_stereo_flow_multiview'
stereo_dir_2015 = '/home/access/disk/dev/data_sets/kitti/Sharons datasets/data_scene_flow_multiview'
val_data = StereoDataset_new(stereo_dir_2012, stereo_dir_2015, isTrainingData=False)
val_dataloader = DataLoader(val_data, batch_size=1)


avg_mse = 0
avg_psnr = 0
avg_l1 = 0
avg_msssim = 0
min_msssim = 1.1
max_msssim = 0
min_idx = 0
max_idx = 0
count = 0
M = 32
with torch.no_grad():
    for batch, data in enumerate(val_dataloader):
        # Get stereo pair
        images_cam1, images_cam2 = data
        # Cut to be multiple of 32 (M)
        shape = images_cam1.size()
        images_cam1 = images_cam1[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
        images_cam2 = images_cam2[:, :, :M * (shape[2] // M), :M * (shape[3] // M)]
        images_cam1 = images_cam1.to(device)
        images_cam2 = images_cam2.to(device)

        # get model outputs
        _, mse_2, img1_recon, _ = model(images_cam1, images_cam2)
        numpy_input_image = torch.squeeze(images_cam1).permute(1, 2, 0).cpu().detach().numpy()
        numpy_output_image = torch.squeeze(img1_recon).permute(1, 2, 0).cpu().detach().numpy()
        l1 = np.mean(np.abs(numpy_input_image - numpy_output_image))
        mse = np.mean(np.square(numpy_input_image - numpy_output_image))  # * 255**2   #mse_loss.item()/2
        psnr = -20 * np.log10(np.sqrt(mse))
        msssim = msssim_x_vs_rec((numpy_input_image * 255), (numpy_output_image * 255))

        print(psnr, msssim)
        avg_msssim += msssim.item()
        avg_l1 = avg_l1 + l1
        avg_mse = avg_mse + mse
        avg_psnr += psnr
        count = count + 1
        msssim_score = msssim.item()
        if msssim_score > max_msssim:
            max_msssim = msssim_score
            img_rec_max_msssim = numpy_output_image
            img_orig_max_msssim = numpy_input_image
        if msssim_score < min_msssim:
            min_msssim = msssim_score
            img_rec_min_msssim = numpy_output_image
            img_orig_min_msssim = numpy_input_image




avg_mse = avg_mse / count
avg_psnr = avg_psnr / count
avg_l1 = avg_l1 / count
avg_msssim = avg_msssim / count
rms = np.sqrt(avg_mse)
print('min  MS-SSIM = ', min_msssim)
print('max MS-SSIM = ', max_msssim)
print('average MSE: ', avg_mse,',  ', avg_mse*255**2)
print('average RMS: ', rms,', ', rms*255)
print('average PSNR: ', avg_psnr)
print('average MS-SSIM: ', avg_msssim)



plot_best_and_worst = True
if plot_best_and_worst:
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Best MS-SSIM reconstruction = ' + str(max_msssim))
    ax1.imshow(img_orig_max_msssim)
    ax2.imshow(img_rec_max_msssim)
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Worst MS-SSIM reconstruction = ' + str(min_msssim))
    ax1.imshow(img_orig_min_msssim)
    ax2.imshow(img_rec_min_msssim)
    fig.tight_layout()

    plt.show()
