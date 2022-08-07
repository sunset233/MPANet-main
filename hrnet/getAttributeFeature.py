import argparse
import os

import numpy as np
import torch
import torchvision.transforms
from torch import nn
import hrnet.lib.models
from hrnet.lib import config
from hrnet.lib.config import config, update_config
from hrnet.lib.utils.utils import create_logger
import torch.nn.functional as F


def parse_args():
    parser1 = argparse.ArgumentParser(description='Test segmentation network')

    parser1.add_argument('--cfg',
                        help='experiment confngigure file name',
                        default='/home/lxz/lph/MPANet-main/hrnet/experiments/market1501/seg_hrnet_w48_market1501.yaml',
                        type=str)
    parser1.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args1 = parser1.parse_args()
    update_config(config, args1)

    return args1

def load_model():
    args1 = parse_args()
    # build model
    model = eval('hrnet.lib.models.seg_hrnet.get_seg_model')(config)


    model_state_file = '/home/lxz/lph/Human-Parsing-Network-human/pretrained_models/best.pth'

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = nn.DataParallel(model).cuda()

    return model

def combine_label(img):
    img[img == 6] = 1
    img[img == 7] = 1
    img[img == 5] = 2
    img[img == 8] = 2
    img[img == 9] = 3
    img[img == 4] = 3
    return img

def pred_imgs(imgs, model):
    # pred = model(imgs)
    # image, _, size, name = batch
    # size = size[0]
    # pred = test_dataset.multi_scale_inference(
    #     model,
    #     image,
    #     scales=config.TEST.SCALE_LIST,
    #     flip=config.TEST.FLIP_TEST)

    size = imgs.size()
    # transforms = torchvision.transforms.Grayscale(3)
    # imgs = transforms(imgs)
    b, c, h, w = imgs.shape
    # print("image.shape", size)
    preds = model(imgs)
    preds = F.upsample(input=preds,
                      size=(size[-2], size[-1]),
                      mode='bilinear')
    if config.TEST.FLIP_TEST:
        flip_img = imgs.numpy()[:, :, :, ::-1]
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
        preds += flip_pred
        preds = preds * 0.5
    preds = preds.exp()

    if preds.size()[-2] != size[0] or preds.size()[-1] != size[1]:
        preds = F.upsample(preds, (size[-2], size[-1]),
                          mode='bilinear')
    preds = preds.cpu().detach().numpy().copy()
    preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
    mask = []
    mask_1 = []
    mask_2 = []
    mask_3 = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = combine_label(pred)
        mask.append(pred)
        mask_1.append(np.where(pred == 1, 1, 0))
        mask_2.append(np.where(pred == 2, 1, 0))
        mask_3.append(np.where(pred == 3, 1, 0))
    mask = torch.from_numpy(np.array(mask).copy()).cuda()
    mask_1 = torch.from_numpy(np.array(mask_1).copy()).cuda()
    mask_2 = torch.from_numpy(np.array(mask_2.copy())).cuda()
    mask_3 = torch.from_numpy(np.array(mask_3.copy())).cuda()
    return mask, mask_1, mask_2, mask_3

if __name__ == '__main__':
    imgs = torch.randn((40, 3, 384, 128))
    model = load_model()
    mask_1, mask_2, mask_3 = pred_imgs(imgs, model)
    print(mask_1.shape)