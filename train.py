import os
import sys  
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import dis_est_utils
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from rich import print

from dis_est_utils.utils_utils import get_optimizer, save_checkpoint, create_logger
from settings import config as cfg

from ik_net_develop.datasets import IKDataset, HOT3DLoader as TargetDataset
from models.ik_network import IKNet

from core.loss import PoseLoss 
from core.function import train, validate, test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = IKNet(cfg)
    model = torch.nn.DataParallel(model).to(device)
    
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, TargetDataset.__name__, 'train')

    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # define loss function (criterion) and optimizer
    criterions = {}
    criterions['orient_loss'] = PoseLoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).to(device)
    criterions['pose_loss'] = PoseLoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).to(device)

    batch_size = cfg.BATCH_SIZE * cfg.N_GPUS
    n_workers = 0 #if cfg.DEBUG.NO_MP else min(16, batch_size)
    prefetch_factor = 16 if n_workers > 0 else None
    persistent_workers = False

    print(f"BATCH_SIZE: {batch_size}")
    print(f"N_WORKERS: {n_workers}")
    print(f"N_GPUS: {cfg.N_GPUS}")
    print(f'IMAGE_SIZE: {cfg.MODEL.IMAGE_SIZE}')
    print(f'HEATMAP_SIZE: {cfg.MODEL.HEATMAP_SIZE}')
    
    sequence_length = cfg.DATASET.SEQUENCE_LENGTH  # Add sequence length from config


    train_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='train'), sequence_length=sequence_length)
    # valid_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='val'), sequence_length=sequence_length)
    valid_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='val'), sequence_length=sequence_length)
    test_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='val'), sequence_length=sequence_length)
    
    # train_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='train'), sequence_length=sequence_length)
    # valid_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='train'), sequence_length=sequence_length)
    # test_dataset = IKDataset(cfg, TargetDataset(cfg.DATASET.ROOT, get_camera=True, split='train'), sequence_length=sequence_length)
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers ,
        pin_memory=False,
        persistent_workers=persistent_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        drop_last=True,
        prefetch_factor=prefetch_factor)

    best_perf = 1e6
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, cfg.MODEL.CHECKPOINT_PATH
    )
    # checkpoint_file = '/netscratch/asodariya/Experiments/Aria/ik_net_develop/logs/tensorboard/EgoHPE/HOT3DLoader_2025-03-18-10-09/model_best.pth'
    
    if os.path.isfile(checkpoint_file) and checkpoint_file.endswith('.pth'):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        last_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in trange(begin_epoch, cfg.TRAIN.END_EPOCH, desc='Epoch'):
        try:
            print('In Train Loop')            
            train(cfg, train_loader, train_dataset, model, criterions, optimizer, epoch, final_output_dir, tb_log_dir, writer_dict, device)
            
            if epoch % 10 == 0:
                # evaluate on validation set
                perf_indicator = validate(
                    cfg, valid_loader, valid_dataset, model, criterions,
                    final_output_dir, tb_log_dir, writer_dict, device
                )

                test(cfg, None, test_dataset, model, tb_log_dir, writer_dict, device)

            if epoch % 10 != 0:
                continue

            lr_scheduler.step()

        except KeyboardInterrupt as e:
            perf_indicator = 1e6
            
        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint(epoch + 1, {
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, tb_log_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

if __name__ == '__main__':
    main()