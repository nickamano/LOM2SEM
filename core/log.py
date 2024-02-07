import os
import sys
import shutil
import datetime
from torch.utils.tensorboard import SummaryWriter
import pytz

def config_writer(cfg):
    # generate logfile name
    if cfg.log.directory is None:
        OUT_DIR = "results/"
    else:
        OUT_DIR = cfg.log.directory
    os.makedirs(OUT_DIR, exist_ok=True)
    if cfg.log.name is None:
        timezone = pytz.timezone('US/Pacific')
        cfg.log.name = f'{cfg.config_file}_{datetime.datetime.now(timezone).strftime("%Y%m%d_%H%M%S")}'
        if cfg.debug:
            cfg.log.name = 'debug_' + cfg.log.name
    cfg.log.dir = os.path.join(OUT_DIR, cfg.model.name, cfg.log.name)
    cfg.checkpoint_dir = os.path.join(cfg.log.dir, 'checkpoint') # used for saving model
    cfg.log.gen_dir = os.path.join(cfg.log.dir, 'gen') # used for saving generated images
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log.gen_dir, exist_ok=True)

    # setup tensorboard writer
    writer_folder = os.path.join(cfg.log.dir, 'tensorboard')
    if os.path.isdir(writer_folder):
        shutil.rmtree(writer_folder)  # reset the folder, can also not reset
    writer = SummaryWriter(writer_folder)

    # redirect stdout print, better for large scale experiments
    if not cfg.debug:
        sys.stdout = open(f'{cfg.log.dir}/log.txt', 'w')
        sys.stderr = open(f'{cfg.log.dir}/log.txt', 'w')

    # log configuration
    print("-"*50)
    print(cfg)
    print("-"*50)
    print('Time:', datetime.datetime.now().strftime("%Y/%m/%d - %H:%M"))

    return writer