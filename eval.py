import argparse
from cleanfid import fid
from glob import glob
import os

from torcheval.metrics.functional import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from core.dataset import TestImagePairDataset
from core.utils import inception_score

from Palette.core.base_dataset import BaseDataset
import numpy as np

from skimage.metrics import structural_similarity as ssim
from PIL import Image 
import matplotlib.pyplot as mpl

def MPSNR(true_image_folder, pred_image_folder, root):
    """
    Calculates the mean squared error of the true and predicted images 
    """
    dataset = TestImagePairDataset(size=512, folder_A=true_image_folder, folder_B=pred_image_folder)
    dataloader = DataLoader(dataset, batch_size= 1)
    PSNRs = []

    for true, pred in dataloader:
        PSNRs.append(peak_signal_noise_ratio(target = true, input = pred))
    return PSNRs

def SSIM(true_image_folder, pred_image_folder, root):
    """
    Calculates the mean squared error of the true and predicted images 
    """
    dataset = TestImagePairDataset(size=512, folder_A=true_image_folder, folder_B=pred_image_folder)
    dataloader = DataLoader(dataset, batch_size= 1)
    SSIMs = []

    for true, pred in dataloader:
        true = true.numpy().squeeze()
        pred = pred.numpy().squeeze()
        concat = np.concatenate((true,pred))
        SSIMs.append(ssim(im1 = true, im2 = pred, channel_axis = 0, gaussian_weights = True, 
        sigma = 1.5, use_sample_covariance = False, data_range= (concat.max() - concat.min())))
    return SSIMs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tru', type=str, help='Ground truth images directory')
    parser.add_argument('-g', '--gen', type=str, help='Generate images directory')
    parser.add_argument('-d', '--dst', type=str, help="Destination of results", required = False)
    parser.add_argument('-b', '--batch', type=bool, help="If the process is a batch computation", required = False)
   
    ''' parser configs '''
    args = parser.parse_args()

    if args.batch:
        fid_scores = []
        is_means, is_stds  = ([], [])
        PSNRs = []


        dirs = glob(args.gen, recursive = True)
        for dir in dirs:
            fid_scores.append(fid.compute_fid(args.tru, f'{args.gen}/{dir}/'))
            is_mean, is_std = inception_score(BaseDataset(dir), cuda=True, batch_size=8, resize=True, splits=10)
            is_means.append(is_mean)
            is_stds.append(is_std)
            PSNRs.append(MPSNR(args.tru, dir, args.gen))

        if args.dst:
            # Writing to file
            with open(args.dst + "/output.txt", "w+") as f:
                f.write(f'dir,FID,IS Mean,IS STD,PSNR')
                for i,dir in enumerate(dirs):
                    # Writing data to a file
                    f.write(f'{dir},{fid_scores[i]},{is_means[i]}, {is_stds[i]}, {PSNRs[i]}')
        else:
            print(f'dir,FID,IS Mean,IS STD,PSNR')
            for i,dir in enumerate(dirs):
                    # Writing data to a file
                    print(f'{dir},{fid_scores[i]},{is_means[i]}, {is_stds[i]}, {PSNRs[i]}')
        

    else:
        fid_score = fid.compute_fid(args.tru, args.gen)
        is_mean, is_std = inception_score(BaseDataset(args.gen), cuda=True, batch_size=8, resize=True, splits=10)
        ssim = SSIM(args.tru, args.gen, '')
        PSNR = MPSNR(args.tru, args.gen, '')
        
        if args.dst:
            # Writing to file
            os.makedirs(args.dst, exist_ok=True)
            np.save(os.path.join(args.dst,"ssim.npy"), ssim)
            np.save(os.path.join(args.dst,"psnr.npy"), PSNR)
            
            with open(os.path.join(args.dst,"output.txt"), "w+") as f:
                # Writing data to a file
                f.write(f'FID: {fid_score}, IS:{is_mean} {is_std}, SSIM:{np.mean(ssim)} {np.std(ssim)}, PSNR: {np.mean(PSNR)} {np.std(PSNR)}')
        else:
            # print(f'FID: {fid_score}')
            print(f'IS:{is_mean} {is_std}')
            # print(f'SSIM:{np.mean(ssim)} {np.std(ssim)}')
            # print(f'PSNR: {np.mean(PSNR)} {np.std(PSNR)}')