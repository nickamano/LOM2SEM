from yacs.config import CfgNode as CN

from core.base_config import cfg

cfg.model.name = 'pix2pix'

cfg.pix2pix = CN()
cfg.pix2pix.lambda_recon = 50
cfg.pix2pix.patch_level = 3
cfg.pix2pix.adv_loss_type = 'LS'

cfg.p2phd = CN()
cfg.p2phd.n_disc_layers = 3
cfg.p2phd.n_discriminators = 3

cfg.train.gen_lr = 2e-4
cfg.train.disc_lr = 1e-4
cfg.train.lr_decay_start = 300
cfg.train.lr_decay_factor = 0.01

cfg.data.gen_checkpoint = None