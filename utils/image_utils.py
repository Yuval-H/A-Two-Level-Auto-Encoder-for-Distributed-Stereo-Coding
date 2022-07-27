import numpy as np
from utils import ms_ssim_np_imgcomp


def randFlipStereoImage(img1, img2):
    # Determine which kind of augmentation takes place according to probabilities
    random_chooser_lr = 0#np.random.rand()
    random_chooser_ud = np.random.rand()
    if random_chooser_lr > 0.5:
        img1 = np.fliplr(img1)
        img2 = np.fliplr(img2)
    if random_chooser_ud > 0.5:
        img1 = np.flipud(img1)
        img2 = np.flipud(img2)
    return img1, img2

def msssim_x_vs_rec(x, x_rec):
    if x.ndim < 4:
        x = np.expand_dims(x, axis=-1)
    if x_rec.ndim < 4:
        x_rec = np.expand_dims(x_rec, axis=-1)
    return ms_ssim_np_imgcomp._calc_msssim_orig(x, x_rec)