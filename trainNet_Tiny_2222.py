import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss, HeatmapLoss
from core.function import train, trainNet
from core.function import validate, validateNet
from models.myModel import LiteHRNet
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset.FishKeyDataset
import dataset
import models

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='experiments/myHRNet/model_256x256_VMHRNet18_2222_tiny.yaml',
                        type=str)

    
    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)
    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    # HRNet
    if config.MODEL.NAME == 'LiteHRNet':
        base_dim = 96
        extra = dict(
            in_channels=3,
            extra=dict(
                stem=dict(
                    stem_channels=32,
                    out_channels=32,
                    expand_ratio=1),
                num_stages=3,
                stages_spec=dict(
                    num_branches=(2, 3, 4),
                    num_depths=(
                        (2, 2),
                        (2, 2, 2),
                        (2, 2, 2, 2)
                    ),
                    module_type=('VSS', 'VSS', 'VSS'),
                    with_fuse=(True, True, True),
                    reduce_ratios=(8, 8, 8),
                    num_dims=(
                        (base_dim, base_dim * 2),
                        (base_dim, base_dim * 2, base_dim * 4),
                        (base_dim, base_dim * 2, base_dim * 4, base_dim * 8),
                    )),
                with_head=True,
            ))
        model = LiteHRNet(**extra).cuda()
        weigth_path = 'models/pertrain/imagenet_1k/vssm_tiny_0230_ckpt_epoch_262.pth'
        checkpoint = torch.load(weigth_path)
        state_dict = model.state_dict()

        def change_parameter_names(state_dict):
            def change_name(src_pattern, dst_pattern):
                for src_key in list(state_dict.keys()):
                    if src_key.startswith(src_pattern):
                        dst_key = src_key.replace(src_pattern, dst_pattern, 1)
                        state_dict[dst_key] = state_dict.pop(src_key)

            # 执行需要的参数名称更改
            for i in range(100):
                for j in range(100):
                    change_name(f"layers.{i}.blocks.{j}.norm.", f"0.layers.{i}.{j}.ln_1.")
                    change_name(f"layers.{i}.blocks.{j}.op", f"0.layers.{i}.{j}.self_attention")

        change_parameter_names(checkpoint['model'])
        for key, param in state_dict.items():
            for per_key, per_param in checkpoint['model'].items():
                if per_key in key:
                    param.data = per_param.data
    # if config.MODEL.NAME == 'HRNet':
    #     model = HRNet(config.MODEL.HRNet, 20, 0.1).cuda()
    # elif config.MODEL.NAME == 'PoseResNet':
    #     model = get_pose_net(config, is_train = True).cuda()

    # copy model file
    this_dir = os.path.dirname(__file__)
    # shutil.copy2(
    #     os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
    #     final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)

    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=1e-3, max_lr=4e-1, step_size_up=10, step_size_down=5, mode="triangular2", cycle_momentum=False
    # )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = dataset.FishKeyDataset.FishKeyPoseDataset(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = dataset.FishKeyDataset.FishKeyPoseDataset(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        # # train for one epoch
        trainNet(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        lr_scheduler.step()

        # evaluate on validation set
        perf_indicator = validateNet(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    # torch.save(model.module.state_dict(), final_model_state_file)
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
