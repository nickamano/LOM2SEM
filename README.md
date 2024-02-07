# LOM2SEM

This repository contains the code for generating SEM images from LOM images for bi-phase steels.

## Pix2Pix
Pix2Pix is a conditional image generation model based on GAN. 
To train a Pix2PixGAN model, run the following command:
```bash
python -m Pix2Pix.train_p2phd --config Pix2Pix/configs/p2phd_aug.yaml
```

## Palette
Palette is a conditional image generation model based on diffusion models.
To train a Palette model, run the following command:
```bash
python -m Palette.run -p train -c ./Palette/config/lom2sem_4layers_attn16.json
```