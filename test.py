import argparse
import os
import time
import numpy as np
import json
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
# from models.Lite_HRNet import LiteHRNet
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

from core.inference import get_final_preds
import dataset.FishKeyDataset
import dataset
import models
from models.posenet import HgNet, VNet
from models.HRNet import HRNet
from models.pose_resnet import get_pose_net

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

kp_dic = ["M", "Eh", "Et", "H", "G", "A1", "A2", "W1", "W2", "W3", "T"]
skeleton_dic = [[1, 2], [2, 3], [1, 4], [4, 5], [4, 6], [4, 7], [4, 10], [8, 10], [9, 10], [10, 11]]
saveJSON = 'output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_22F2_small/results/keypoints_test_results.json'


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='experiments/myHRNet/model_256x256_VMHRNet18_22F2_small.yaml',
                        # default='experiments/SimpleBaseline/model_256×256_resnet50.yaml',
                        # default='experiments/hourglass/model_256×256_4stack_hourglass.yaml',
                        # default='experiments/HRNet/model_256×256_LiteHRNet18.yaml',
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

    # logger, final_output_dir, tb_log_dir = create_logger(
    #     config, args.cfg, 'train')
    #
    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))
    #
    # # cudnn related setting
    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    # # HRNet
    # if config.MODEL.NAME == 'LiteHRNet':
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
                    (2, 2, 15),
                    (2, 2, 15, 2)
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

    # model = get_pose_net(config, is_train = False).cuda()

    # model = eval(config.MODEL.NAME)(
    #     nstack=config.MODEL.NUM_STACKS, inp_dim=256, oup_dim=11
    # ).cuda()

    # base_channel = 40
    # extra = dict(
    #     in_channels=3,
    #     extra=dict(
    #         stem=dict(
    #             stem_channels=32,
    #             out_channels=32,
    #             expand_ratio=1),
    #         num_stages=3,
    #         stages_spec=dict(
    #             num_modules=(2, 4, 2),
    #             num_branches=(2, 3, 4),
    #             num_blocks=(2, 2, 2),
    #             module_type=('LITE', 'LITE', 'LITE'),
    #             with_fuse=(True, True, True),
    #             reduce_ratios=(8, 8, 8),
    #             num_channels=(
    #                 (base_channel, base_channel * 2),
    #                 (base_channel, base_channel * 2, base_channel * 4),
    #                 (base_channel, base_channel * 2, base_channel * 4, base_channel * 8),
    #             )),
    #         with_head=True,
    #     ))
    # model = LiteHRNet(**extra).cuda()


    weigth_path = 'output/ModelFishPose/LiteHRNet_50/model_256x256_VMHRNet18_22F2_small/model_best.pth.tar'
    # weigth_path = 'output/ModelFishPose/PoseResNet_50/model_256×256_resnet50/model_best.pth.tar'
    # weigth_path = 'output/ModelFishPose/HgNet_50/model_256×256_4stack_hourglass/model_best.pth.tar'
    # weigth_path = 'output/ModelFishPose/LiteHRNet_50/model_256×256_LiteHRNet18/model_best.pth.tar'
    model.load_state_dict(torch.load(weigth_path))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([transforms.ToTensor(), normalize,])
    # input_image = transform(image).unsqueeze(0)

    config.DATASET.TEST_SET = 'test'
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

    gpus = [int(i) for i in config.GPUS.split(',')]

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    model.eval()

    num_samples = len(valid_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))

    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    keypoints_test = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(valid_loader):
            # compute output
            input = input.cuda()
            num_images = input.size(0)
            print(num_images, idx)

            output = model(input).cuda()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)
            print(preds.shape[0], idx, idx + num_images)

            all_preds[idx:(idx + num_images), :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:(idx + num_images), :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:(idx + num_images), 0:2] = c[:, 0:2]
            all_boxes[idx:(idx + num_images), 2:4] = s[:, 0:2]
            all_boxes[idx:(idx + num_images), 4] = np.prod(s * 200, 1)
            all_boxes[idx:(idx + num_images), 5] = score

            for i in range(num_images):
                kp_pred = all_preds[idx + i].reshape(-1).tolist()
                image_name = os.path.basename(meta['image'][i])
                print(image_name)
                count = 0
                data = {
                    'image_name': image_name,
                    'categories': [
                        {"id": idx + i + 1,
                         "name": "person",
                         "supercategory": "person",
                         "keypoints": kp_dic,
                         "skeleton": skeleton_dic
                         }],
                    'keypoints': kp_pred
                }
                keypoints_test.append(data)

            idx += num_images


        # print(keypoints_test)
        data_str = json.dumps(keypoints_test)
        with open(saveJSON, 'w') as fp:
            fp.write(data_str)
        print(f"Keypoint detection completed.")




if __name__ == '__main__':
    main()
