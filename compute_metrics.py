import skimage.io as sio
import numpy as np
from pathlib import Path
from natsort import natsorted
from skimage.measure import compare_psnr, compare_ssim

if __name__ == '__main__':
    gt_folder = Path('./testing_set')
    pred_folder = Path('./testing_res/srn-celeba')

    gt_image_paths = natsorted(list(gt_folder.glob('*.png')))
    pred_image_paths = natsorted(list(pred_folder.glob('*.png')))

    psnrs = []
    ssims = []
    for gt_image_path, pred_image_path in zip(gt_image_paths, pred_image_paths):
        gt_image = sio.imread(str(gt_image_path))
        pred_image = sio.imread(str(pred_image_path))

        psnrs.append(compare_psnr(gt_image, pred_image))
        ssims.append(compare_ssim(gt_image, pred_image, multichannel=True))

    print('Average PSNR = ', np.mean(psnrs))
    print('Average SSIM = ', np.mean(ssims))


