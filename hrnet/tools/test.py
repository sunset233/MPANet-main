# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from tqdm import tqdm
from torch.nn import functional as F
import _init_paths
import hrnet.lib.models
import hrnet.lib.datasets as datasets
from hrnet.lib.config import config
from hrnet.lib.config import update_config
from hrnet.lib.core.function import testval, test
from hrnet.lib.utils.modelsummary import get_model_summary
from hrnet.lib.utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('--cfg',
                        help='experiment confngigure file name',
                        default='/home/lxz/lph/MPANet-main/hrnet/experiments/market1501/seg_hrnet_w48_market1501.yaml',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('hrnet.lib.models.seg_hrnet.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        # model_state_file = config.TEST.MODEL_FILE
        model_state_file = '/home/lxz/lph/Human-Parsing-Network-human/pretrained_models/best.pth'
    else:
        model_state_file = os.path.join(final_output_dir,
                                        '/home/lxz/lph/Human-Parsing-Network-human/pretrained_models/final_state.pth')
    # model_state_file = os.path.join('../pretrained_models/best.pth')
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_samples=None,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TEST.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    start = timeit.default_timer()
    if 'val' in config.DATASET.TEST_SET:
        mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config,
                                                           test_dataset,
                                                           testloader,
                                                           model)

        msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU,
            pixel_acc, mean_acc)
        # logging.info(msg)
        # logging.info(IoU_array)
    # elif 'test' in config.DATASET.TEST_SET:
    else:
        # test(config,
        #      test_dataset,
        #      testloader,
        #      model,
        #      sv_dir=final_output_dir)
        sv_pred = True
        sv_dir = final_output_dir
        model.eval()
        transforms = torchvision.transforms.Grayscale(3)
        with torch.no_grad():
            for batch in tqdm(testloader):
                # for batch in testloader:
                image, _, size, name = batch
                image = transforms(image)
                size = size[0]
                # pred = test_dataset.multi_scale_inference(
                #     model,
                #     image,
                #     scales=config.TEST.SCALE_LIST,
                #     flip=config.TEST.FLIP_TEST)

                size = image.size()
                b, c, h, w = image.shape
                print("image.shape", size)

                pred = model(image)
                pred = F.upsample(input=pred,
                                  size=(size[-2], size[-1]),
                                  mode='bilinear')
                if config.TEST.FLIP_TEST:
                    flip_img = image.numpy()[:, :, :, ::-1]
                    flip_output = model(torch.from_numpy(flip_img.copy()))
                    flip_output = F.upsample(input=flip_output,
                                             size=(size[-2], size[-1]),
                                             mode='bilinear')
                    flip_pred = flip_output.cpu().numpy().copy()
                    flip_pred[:, 14, :, :] = flip_output[:, 15, :, :]
                    flip_pred[:, 15, :, :] = flip_output[:, 14, :, :]
                    flip_pred[:, 16, :, :] = flip_output[:, 17, :, :]
                    flip_pred[:, 17, :, :] = flip_output[:, 16, :, :]
                    flip_pred[:, 18, :, :] = flip_output[:, 19, :, :]
                    flip_pred[:, 19, :, :] = flip_output[:, 18, :, :]
                    flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
                    pred += flip_pred
                    pred = pred * 0.5

                if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                    pred = F.upsample(pred, (size[-2], size[-1]),
                                      mode='bilinear')

                if sv_pred:
                    sv_path = os.path.join(sv_dir, 'test_results')
                    if not os.path.exists(sv_path):
                        os.mkdir(sv_path)
                    print(sv_path)
                    test_dataset.save_pred(pred, sv_path, name)

    end = timeit.default_timer()
    # logger.info('Mins: %d' % np.int((end-start)/60))
    logger.info('Done')


if __name__ == '__main__':
    main()
