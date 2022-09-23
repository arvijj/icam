# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import argparse
import torch
import numpy as np
import random
torch.manual_seed(1) # cpu
torch.cuda.manual_seed(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time
import os

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback
from torch.utils.data import DataLoader
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from utils.DenseCRF import dense_crf
from utils.test_utils import single_gpu_test
from utils.imutils import onehot

def ClassLogSoftMax(f, category):
    exp = torch.exp(f)
    exp_norm = exp/torch.sum(exp*category, dim=1, keepdim=True)
    softmax = exp_norm*category
    logsoftmax = torch.log(exp_norm)*category
    return softmax, logsoftmax

def test_net(args, cfg):
    dataset = generate_dataset(cfg, period=args.period, transform='none')
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    dataloader = DataLoader(dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=cfg.DATA_WORKERS,
                worker_init_fn = worker_init_fn)

    #net = generate_net(cfg, batchnorm=nn.BatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
    net = generate_net(cfg, batchnorm=nn.BatchNorm2d)
    print('net initialize')

    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
    print('start loading model %s'%cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT)
    net.load_state_dict(model_dict, strict=False)

    print('Use %d GPU'%cfg.GPUS)
    assert torch.cuda.device_count() == cfg.GPUS
    device = torch.device('cuda')
    net.to(device)
    net.eval()

    def prepare_func(sample):	
        image_msf = []
        for rate in cfg.TEST_MULTISCALE:
            inputs_batched = sample['image_%f'%rate]
            image_msf.append(inputs_batched)
            if cfg.TEST_FLIP:
                image_msf.append(torch.flip(inputs_batched,[3]))
        return image_msf

    def inference_func(model, img):
        seg, _ = model(img)
        return seg

    def collect_func(result_list, sample):
        [batch, channel, height, width] = sample['image'].size()
        for i in range(len(result_list)):
            result_seg = F.interpolate(result_list[i], (height, width), mode='bilinear', align_corners=True)	
            if cfg.TEST_FLIP and i % 2 == 1:
                result_seg = torch.flip(result_seg, [3])
            result_list[i] = result_seg
        prob_seg = torch.cat(result_list, dim=0)
        prob_seg = F.softmax(torch.mean(prob_seg, dim=0, keepdim=True),dim=1)[0]


        if cfg.TEST_CRF:
            prob = prob_seg.cpu().numpy()
            img_batched = img_denorm(sample['image'][0].cpu().numpy()).astype(np.uint8)
            prob = dense_crf(prob, img_batched, n_classes=cfg.MODEL_NUM_CLASSES, n_iters=1)
            prob_seg = torch.from_numpy(prob.astype(np.float32))

        result = torch.argmax(prob_seg, dim=0, keepdim=False).cpu().numpy()
        return result

    def save_step_func(result_sample):
        dataset.save_result([result_sample], cfg.MODEL_NAME)

    result_list = single_gpu_test(net, dataloader, prepare_func=prepare_func, inference_func=inference_func, collect_func=collect_func, save_step_func=save_step_func)
    resultlog = dataset.do_python_eval(cfg.MODEL_NAME)
    print('Test finished')
    writelog(cfg, args.period, metric=resultlog)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default='val', type=str)
    parser.add_argument("--exp-dir", default=None, type=str)
    parser.add_argument("--data-pseudo-gt", default=None, type=str)
    parser.add_argument("--test-ckpt", default=None, type=str)
    args = parser.parse_args()

    if args.exp_dir is not None:
        config_dict['EXP_DIR'] = args.exp_dir
        config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['EXP_DIR'], 'ckpts')
        config_dict['LOG_DIR'] = os.path.join(config_dict['EXP_DIR'], 'logs', 'final')
        config_dict['TEST_CKPT'] = os.path.join(config_dict['MODEL_SAVE_DIR'], 'final.pth')
    if args.data_pseudo_gt is not None:
        config_dict['DATA_PSEUDO_GT'] = args.data_pseudo_gt
    if args.test_ckpt is not None:
        config_dict['TEST_CKPT'] = args.test_ckpt
    
    cfg = Configuration(config_dict, False)
    cfg.DATA_AUG = False
    test_net(args, cfg)


