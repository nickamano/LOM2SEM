from yacs.config import CfgNode as CN

from core.base_config import cfg

cfg.model.name = 'AdaIN'

cfg.train.lr = 2e-4
cfg.train.lr_decay_start = 300
cfg.train.lr_decay_factor = 0.01
cfg.train.wt_s = 10

cfg.data.desc_checkpoint = None

