# LOM2SEM

This repository contains the code for generating SEM images from LOM images for bi-phase steels.

## Pix2Pix
Pix2Pix is a conditional image generation model with a GAN architecture and standard MLP models. 
To train a Pix2PixGAN model, run the following command:
```bash
python -m Pix2Pix.train_p2phd --config Pix2Pix/configs/p2phd_aug.yaml
```

## Palette
Palette is a conditional image generation framework with a GAN architecture and using diffusion models.
To train a Palette model, run the following command:
```bash
python -m Palette.run -p train -c ./Palette/config/lom2sem_4layers_attn16.json
```

## AdaIn
AdaIn is a style mixing generation model based on an Encoder-Decoder structure.
To train a Palette model, run the following command:
```bash
python -m Palette.run -p train -c ./Palette/config/lom2sem_4layers_attn16.json
```

## Eval Models
To evaluate the models use the `eval.py` file to extract the IOU, IS, FID.
```bash
python eval.py -t [ground image path] -g [generated image path]
```