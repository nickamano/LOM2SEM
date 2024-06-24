import os
import argparse
from yacs.config import CfgNode as CN
import torch

cfg = CN()
# Basic options
cfg.device = 0 if torch.cuda.is_available() else 'cpu'
cfg.seed = 42


# Dataset options
cfg.data = CN()
cfg.data.directory = "./data/mecs_steel"
cfg.data.LOM_folder = 'LOM640'
cfg.data.SEM_folder = 'SEM640'
cfg.data.split = 'all.txt'
cfg.data.test_LOM_folder = None
cfg.data.test_SEM_folder = None


# Training options
cfg.train = CN()
cfg.train.input_size = [256, 256]
cfg.train.batch_size = 4
cfg.train.epochs = 200
cfg.train.max_steps = 100000
cfg.train.lr = 1e-4
cfg.train.scheduler = None
cfg.train.weight_decay = 0.
cfg.train.dropout = 0.
cfg.train.optimizer = 'Adam'
cfg.train.augmentations = []
cfg.train.num_workers = 4


# Logging options
cfg.log = CN()
cfg.log.directory = None
cfg.log.name = None
cfg.log.display_freq = 500
cfg.log.log_freq = 50
cfg.log.n_row_imgs = 4


# Model options, for different models, the options are different
cfg.model = CN()

# Testing options
cfg.test = CN()
cfg.test.save_path = None


def update_cfg(cfg):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    parser.add_argument('--debug', action='store_true', help="Whether to enter debug mode")
    parser.add_argument('--test', action='store_true', help="Whether to enter test mode")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    # parse from command line
    args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()
    cfg.debug = args.debug
    cfg.test = args.test

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)
        cfg.config_file = os.path.basename(args.config).split('.')[0]
    elif args.config == "":
        cfg.config_file = 'default'
        print(f"No config file specified, use default configs.")
    else:
        raise ValueError(f"Config file {args.config} not found!")

    # Update from command line
    cfg.merge_from_list(args.opts)
    if cfg.debug:
        cfg.log.display_freq = 10
        cfg.log.log_freq = 10
    return cfg