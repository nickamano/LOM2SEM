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
from core.dataset import AdaINDataset, AdaINImagePairDataset
from core.utils import show_tensor_images, ScoreMeter, postprocess
from AdaIN.config import cfg
from AdaIN.models import StyleNet, Loss


def weights_init(m):
    ''' Function for initializing all model weights '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)

def run(cfg):
    writer = config_writer(cfg)
    train_set = AdaINDataset(cfg.data.directory, cfg.train.input_size,
                             cfg.data.LOM_folder, cfg.data.SEM_folder, cfg.data.split, cfg.train.augmentations)
    test_set = AdaINImagePairDataset(cfg.data.directory, cfg.train.input_size,
                                cfg.data.test_LOM_folder, cfg.data.test_SEM_folder, 
                                cfg.data.SEM_folder, split=None)
    print(f'train set size: {len(train_set)}, test set size: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)

    model = StyleNet(cfg.data.desc_checkpoint).to(cfg.device)

    loss_fn = Loss(cfg.train.wt_s, model)

    train(cfg, train_loader, test_set, model, loss_fn, writer, cfg.device)


def train(cfg, dataloader, test_set, model, loss_fn, writer, device):
    model = model
    optimizer = optim.Adam(model.decoder.parameters(), lr=cfg.train.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(cfg.train.lr_decay_factor) / (cfg.train.max_steps - cfg.train.lr_decay_start)))

    curr_step = 0
    score_meter = ScoreMeter(('loss'))
    start = time.time()

    for epoch in range(cfg.train.epochs):
        # Training epoch

        for real_A, real_B, style, _ in dataloader:
            # print(real_A.shape, real_B.shape, style.shape)
            curr_step += 1
            if curr_step > cfg.train.lr_decay_start:
                scheduler.step()
            if curr_step > cfg.train.max_steps:
                break
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            style = style.to(device)

            fake_B, t = model(
                real_A, style, return_t=True
            )

            loss = loss_fn(
                fake_B, real_B, t
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score_meter.update([loss.item()], n=cfg.train.batch_size)
            if curr_step % cfg.log.display_freq == 0:
                image_tensor = torch.cat([real_A[:cfg.log.n_row_imgs],
                                          real_B[:cfg.log.n_row_imgs],
                                          fake_B[:cfg.log.n_row_imgs]])
                image_tensor = (image_tensor + 1) / 2
                title = f"step {curr_step}"
                save_path = f"{cfg.log.gen_dir}/step{curr_step}.png"
                image_grid = show_tensor_images(image_tensor, title, nrow=cfg.log.n_row_imgs,
                                                num_images=cfg.log.n_row_imgs * 3, save_path=save_path)
                writer.add_image('train/gen', image_grid, curr_step)
                eval(cfg, test_set, model, curr_step, writer)

            if curr_step % cfg.log.log_freq == 0:
                print(f"epoch {epoch} | step {curr_step} | {score_meter.stats_string()} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                      f"time {time.time() - start:.2f} | "
                      f"memory {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.0f}MiB", flush=True)
                for k, v in score_meter.stats_dict().items():
                    writer.add_scalar(f'train/{k}', v, curr_step)
                score_meter.reset()

            torch.save(model.decoder.state_dict(), f"{cfg.checkpoint_dir}/model.pt")


@torch.no_grad()
def eval(cfg, dset, generator, curr_step, writer, n_imgs=8):
    generator.eval()
    indices = torch.randint(len(dset), (n_imgs,))
    real_A, real_B, style = [], [], []
    for idx in indices:
        real_A.append(dset[idx][0])
        real_B.append(dset[idx][1])
        style.append(dset[idx][2])
    real_A = torch.stack(real_A)
    real_B = torch.stack(real_B)
    style = torch.stack(style)
    real_A, real_B, style = real_A.to(cfg.device), real_B.to(cfg.device), style.to(cfg.device)
    fake_B = generator(real_A, style)

    image_tensor = torch.cat([real_A,
                              real_B,
                              style,
                              fake_B])
    image_tensor = (image_tensor + 1) / 4
    title = f"test step {curr_step}"
    save_path = f"{cfg.log.gen_dir}/test_step{curr_step}.png"
    image_grid = show_tensor_images(image_tensor, title, nrow=n_imgs,
                                    num_images=n_imgs * 4, save_path=save_path)
    writer.add_image('test/gen', image_grid, curr_step)

@torch.no_grad()
def test_eval(cfg, dset, generator, curr_step, batch_size=4):
    generator.eval()
    for i, (real_A, real_B, style, img_names) in enumerate(dset, 0): 
        real_A, real_B, style = real_A.to(cfg.device), real_B.to(cfg.device), style.to(cfg.device)
        fake_B = generator(real_A, style)

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
    test_set = AdaINImagePairDataset(cfg.data.directory, cfg.train.input_size,
                                cfg.data.test_LOM_folder, cfg.data.test_SEM_folder, 
                                cfg.data.SEM_folder, split=None)
    print(f'test set size: {len(test_set)}')

    model = StyleNet(cfg.data.desc_checkpoint).to(cfg.device)

    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    test_eval(cfg, test_loader, model, 0 , writer)

if __name__ == '__main__':
    cfg.merge_from_file('./AdaIN/configs/default_AdaIN.yaml')
    cfg = update_cfg(cfg)
    if cfg.test:
        test(cfg)
    else:
        run(cfg)

