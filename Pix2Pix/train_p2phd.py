import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pickle
import time
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

from core.base_config import update_cfg
from core.log import config_writer
from core.dataset import P2PHDDataset, ImagePairDataset
from core.utils import show_tensor_images, ScoreMeter, postprocess
from Pix2Pix.config import cfg
from Pix2Pix.pix2pixHD import GlobalGenerator, MultiscaleDiscriminator, Loss


def weights_init(m):
    ''' Function for initializing all model weights '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)

def run(cfg):
    writer = config_writer(cfg)
    train_set = P2PHDDataset(cfg.data.directory, cfg.train.input_size,
                             cfg.data.LOM_folder, cfg.data.SEM_folder, cfg.data.split, cfg.train.augmentations)
    test_set = ImagePairDataset(cfg.data.directory, cfg.train.input_size,
                                cfg.data.test_LOM_folder, cfg.data.test_SEM_folder, split=None)
    print(f'train set size: {len(train_set)}, test set size: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)

    loss_fn = Loss(device=cfg.device)

    gen = GlobalGenerator(in_channels=3, out_channels=1).to(cfg.device).apply(weights_init)
    disc = MultiscaleDiscriminator(in_channels=4, n_layers=cfg.p2phd.n_disc_layers,
                                    n_discriminators=cfg.p2phd.n_discriminators).to(cfg.device).apply(weights_init)

    train(cfg, train_loader, test_set, [gen, disc], loss_fn, writer, cfg.device)


def train(cfg, dataloader, test_set, models, loss_fn, writer, device):
    generator, discriminator = models
    g_optimizer = optim.Adam(generator.parameters(), lr=cfg.train.gen_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.train.disc_lr)
    g_scheduler = optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=np.exp(np.log(cfg.train.lr_decay_factor) / (cfg.train.max_steps - cfg.train.lr_decay_start)))
    d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=np.exp(np.log(cfg.train.lr_decay_factor) / (cfg.train.max_steps - cfg.train.lr_decay_start)))

    curr_step = 0
    score_meter = ScoreMeter(('gen_loss', 'gen_adv', 'gen_recon', 'gen_vgg', 'd_loss'))
    start = time.time()

    for epoch in range(cfg.train.epochs):
        # Training epoch

        for real_A, real_B, _ in dataloader:
            curr_step += 1
            if curr_step > cfg.train.lr_decay_start:
                g_scheduler.step()
                d_scheduler.step()
            if curr_step > cfg.train.max_steps:
                break
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            generator.train()
            discriminator.train()
            g_loss, d_loss, fake_B, g_adv_loss, g_fm_loss, g_vgg_loss = loss_fn(
                real_A, real_B, generator, discriminator
            )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            score_meter.update([g_loss.item(), g_adv_loss, g_fm_loss, g_vgg_loss, d_loss.item()], n=cfg.train.batch_size)
            if curr_step % cfg.log.display_freq == 0:
                image_tensor = torch.cat([real_A[:cfg.log.n_row_imgs],
                                          torch.cat([real_B]*3, dim=1)[:cfg.log.n_row_imgs],
                                          torch.cat([fake_B]*3, dim=1)[:cfg.log.n_row_imgs]])
                image_tensor = (image_tensor + 1) / 2
                title = f"step {curr_step}"
                save_path = f"{cfg.log.gen_dir}/step{curr_step}.png"
                image_grid = show_tensor_images(image_tensor, title, nrow=cfg.log.n_row_imgs,
                                                num_images=cfg.log.n_row_imgs * 3, save_path=save_path)
                writer.add_image('train/gen', image_grid, curr_step)
                eval(cfg, test_set, generator, curr_step, writer)

            if curr_step % cfg.log.log_freq == 0:
                print(f"epoch {epoch} | step {curr_step} | {score_meter.stats_string()} | "
                      f"lr {g_optimizer.param_groups[0]['lr']:.2e} | "
                      f"time {time.time() - start:.2f} | "
                      f"memory {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.0f}MiB", flush=True)
                for k, v in score_meter.stats_dict().items():
                    writer.add_scalar(f'train/{k}', v, curr_step)
                score_meter.reset()

            torch.save({'gen': generator.state_dict(),
                        'gen_opt': g_optimizer.state_dict(),
                        'disc': discriminator.state_dict(),
                        'disc_opt': d_optimizer.state_dict(),
                        }, f"{cfg.checkpoint_dir}/model.pt")


@torch.no_grad()
def eval(cfg, dset, generator, curr_step, writer, n_imgs=8):
    generator.eval()
    indices = torch.randint(len(dset), (n_imgs,))
    real_A, real_B = [], []
    for idx in indices:
        real_A.append(dset[idx][0])
        real_B.append(dset[idx][1])
    real_A = torch.stack(real_A)
    real_B = torch.stack(real_B)
    real_A, real_B = real_A.to(cfg.device), real_B.to(cfg.device)
    fake_B = generator(real_A)

    image_tensor = torch.cat([real_A,
                              torch.cat([real_B] * 3, dim=1),
                              torch.cat([fake_B] * 3, dim=1)])
    image_tensor = (image_tensor + 1) / 2
    title = f"test step {curr_step}"
    save_path = f"{cfg.log.gen_dir}/test_step{curr_step}.png"
    image_grid = show_tensor_images(image_tensor, title, nrow=n_imgs,
                                    num_images=n_imgs * 3, save_path=save_path)
    writer.add_image('test/gen', image_grid, curr_step)

@torch.no_grad()
def test_eval(cfg, dset, generator, curr_step, batch_size=4):
    generator.eval()
    for i, (real_A, real_B, img_names) in enumerate(dset, 0): 
        real_A, real_B = real_A.to(cfg.device), real_B.to(cfg.device)
        fake_B = generator(real_A)

        print(f"test step {i}")
        save_path = f"{cfg.log.gen_dir}/test"
        os.makedirs(save_path, exist_ok=True)

        for j, img_name in enumerate(img_names, 0):
            # print(os.path.join(save_path, img_name))
            np_img = postprocess(fake_B[j].cpu())
            print(np_img[0].shape)
            Image.fromarray(np_img[0]).save(os.path.join(save_path, img_name))

def test(cfg):
    writer = config_writer(cfg)

    test_set = ImagePairDataset(cfg.data.directory, cfg.train.input_size,
                                cfg.data.test_LOM_folder, cfg.data.test_SEM_folder, split=None)
    print(f'test set size: {len(test_set)}')

    test_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2)

    print(f"loading weights from {cfg.data.gen_checkpoint}")

    print('pickle worked')
    weights = torch.load(cfg.data.gen_checkpoint)

    gen = GlobalGenerator(in_channels=3, out_channels=1).to(cfg.device)
    gen.load_state_dict(weights['gen'])

    test_eval(cfg, test_loader, gen, 0 , writer)

if __name__ == '__main__':
    cfg.merge_from_file('./Pix2Pix/configs/default_p2phd.yaml')
    cfg = update_cfg(cfg)
    if cfg.test:
        test(cfg)
    else:
        run(cfg)

