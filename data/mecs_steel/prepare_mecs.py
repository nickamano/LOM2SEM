from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
import numpy as np
import os

image_size = 512

out_train_dir_LOM = f'./data/mecs_steel/train_LOM{image_size}'
out_train_dir_SEM = f'./data/mecs_steel/train_SEM{image_size}'
out_test_dir_LOM = f'./data/mecs_steel/test_LOM{image_size}'
out_test_dir_SEM = f'./data/mecs_steel/test_SEM{image_size}'
for out_dir in [out_train_dir_LOM, out_train_dir_SEM, out_test_dir_LOM, out_test_dir_SEM]:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

raw_imgs_LOM = ['1318 _neu_Reg_ 0.25.tif', '60510 _neu_Reg_ 0.75.tif', 'Z518B _neu_Reg_ 0.25.tif']
raw_imgs_SEM = ['1318 _neu_Reg_ 0.25.tif', '60510 _neu_Reg_ 0.75.tif', 'Z518B _neu_Reg_ 0.25.tif']


n_train = 100
n_test = 20
for raw_img_LOM, raw_img_SEM in zip(raw_imgs_LOM, raw_imgs_SEM):
    raw_LOM = Image.open(f'./data/mecs_steel/LOM/{raw_img_LOM}')
    raw_SEM = Image.open(f'./data/mecs_steel/SEM/{raw_img_SEM}')
    width, height = raw_LOM.size
    n_rows, n_cols = height // image_size, width // image_size
    print(f"n_rows: {n_rows}, n_cols: {n_cols}")
    count = 1
    for i in range(n_rows):
        for j in range(n_cols):
            if count > n_train + n_test:
                continue
            left, upper = j * image_size, i * image_size
            right, lower = left + image_size, upper + image_size
            crop_LOM = raw_LOM.crop((left, upper, right, lower))
            crop_SEM = raw_SEM.crop((left, upper, right, lower))
            if count <= n_train:
                crop_LOM.save(f'{out_train_dir_LOM}/{raw_img_LOM[:-4]}_{i+1}_{j+1}.tif')
                crop_SEM.save(f'{out_train_dir_SEM}/{raw_img_SEM[:-4]}_{i+1}_{j+1}.tif')
            else:
                crop_LOM.save(f'{out_test_dir_LOM}/{raw_img_LOM[:-4]}_{i+1}_{j+1}.tif')
                crop_SEM.save(f'{out_test_dir_SEM}/{raw_img_SEM[:-4]}_{i+1}_{j+1}.tif')
            count += 1