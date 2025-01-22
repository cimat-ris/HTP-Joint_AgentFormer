import os
import sys
import glob
import argparse

import torch
torch.set_default_dtype(torch.float32)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from data.datamodule import AgentFormerDataModule
from utils.config import Config
from utils.utils import get_timestring
from trainer import AgentFormerTrainer


def main(args):
    # initialize AgentFormer config from full config filename
    cfg = Config(args.cfg)

    # Set global random seed
    pl.seed_everything(cfg.seed)

    if args.no_gpu:
        accelerator = None
        plugin = None
    else:
        if not args.mode == 'test':
            plugin = DDPStrategy(find_unused_parameters=args.find_unused_params)
            args.devices = torch.cuda.device_count()
            accelerator = 'gpu'
        else:
            plugin = None
            args.devices = 1
            accelerator = 'gpu'

    print(f"using {args.devices} gpus")
    sanity_val_steps = 1

    # load checkpoint model
    default_root_dir = args.default_root_dir = os.path.join(args.logs_root, cfg.id)
    if args.mode == 'train':
        # specify checkpoint to resume
        resume_cfg = cfg.get('resume_cfg', None)
        if resume_cfg is not None:
            models = sorted(glob.glob(os.path.join(args.logs_root, resume_cfg, '*epoch=*.ckpt')))
            print("tuning via resuming from best checkpoint of other model")
        else:  # resume from last checkpoint
            models = sorted(glob.glob(os.path.join(default_root_dir, 'last-v*.ckpt')))
            if len(models) == 0:
                models = sorted(glob.glob(os.path.join(default_root_dir, 'last*.ckpt')))
    elif args.mode == 'test' or args.mode == 'val':
        if args.checkpoint_str is not None:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*{args.checkpoint_str}*.ckpt')))
        else:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*epoch=*.ckpt')))
        if len(models) == 0:
            raise FileNotFoundError(f'If testing, must exist model in {default_root_dir}')
    else:
        raise NotImplementedError
    print("models:", models)

    if args.checkpoint_path is not None and args.resume:
        resume_from_checkpoint = args.checkpoint_path
        print("LOADING from custom checkpoint:", resume_from_checkpoint)
    elif len(models) > 0 and os.path.isfile(models[-1]) and args.resume:
        resume_from_checkpoint = models[-1]
        print("LOADING from default model directory:", resume_from_checkpoint)
    else:
        resume_from_checkpoint = None
        print("STARTING new run from scratch")

    # initialize DataModule and Trainer
    dm = AgentFormerDataModule(cfg, args)
    model = AgentFormerTrainer(cfg, args)
    if args.log_graph:
        model.set_example_input_array(next(iter(dm.train_dataloader())))
    model.model.set_device(model.device)

    # initialize logging and checkpointing and other training utils
    if args.mode == 'train':
        logger = TensorBoardLogger(args.logs_root, name=cfg.id, log_graph=args.log_graph)
    else:
        logger = None
    early_stop_cb = EarlyStopping(patience=20, verbose=True, monitor='val/ADE_joint')
    checkpoint_callback = ModelCheckpoint(monitor='val/ADE_joint', mode='min', save_last=True,
                                          every_n_epochs=1, dirpath=default_root_dir, filename='{epoch:04d}')
    tqdm = TQDMProgressBar(refresh_rate=args.tqdm_rate)

    callbacks = [tqdm, checkpoint_callback, early_stop_cb]

    print("LOGGING TO:", default_root_dir)
    print("\n\n")
    trainer = pl.Trainer(check_val_every_n_epoch=args.val_every, num_sanity_val_steps=sanity_val_steps,
                         devices=args.devices, strategy=plugin, accelerator=accelerator,
                         log_every_n_steps=50, max_epochs=cfg.num_epochs, default_root_dir=default_root_dir,
                         logger=logger, callbacks=callbacks,)

    if 'train' in args.mode:
        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
        trainer = pl.Trainer(devices=1, accelerator=accelerator, default_root_dir=default_root_dir)
        args.save_traj = True
        args.save_viz = True
        model.update_args(args)
        trainer.test(model, datamodule=dm)
    elif 'test' in args.mode or 'val' in args.mode:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='eth_agentformer_sfm_pre8-2')
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--no_gpu', '-ng', action='store_true', default=False)
    parser.add_argument('--dont_resume', '-dr', '-nc', dest='resume', action='store_false', default=True)
    parser.add_argument('--checkpoint_path', '-cp', default=None)
    parser.add_argument('--checkpoint_str', '-c', default=None)
    parser.add_argument('--no_mp', '-nmp', dest='mp', action='store_false', default=True)
    parser.add_argument('--save_viz', '-v', action='store_true', default=False)
    parser.add_argument('--save_num', '-vn', type=int, default=10, help='number of visualizations to save per eval')
    parser.add_argument('--logs_root', '-lr', default='results', help='where to save checkpoints and tb logs')
    parser.add_argument('--save_traj', '-s', action='store_true', default=False)
    parser.add_argument('--log_graph', '-g', action='store_true', default=False)
    parser.add_argument('--find_unused_params', '-f', action='store_true', default=False)
    parser.add_argument('--tqdm_rate', '-tq', type=int, default=20)
    parser.add_argument('--val_every', '-ve', type=int, default=5)
    args = parser.parse_args()

    time_str = get_timestring()
    print(f"time str: {time_str}")
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print(f"torch version : {torch.__version__}")
    print(f"cudnn version : {torch.backends.cudnn.version()}")

    main(args)
