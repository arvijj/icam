import argparse
import cv2
import importlib
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from tool import pyutils, imutils, torchutils, visualization, probutils

def main(args):

    if args.dataset == 'voc2012':
        num_classes = 20
        train_list = 'voc/train_aug.txt'
        args.data_root += '/VOCdevkit/VOC2012'
        from voc.data import ClsDataset
    elif args.dataset == 'coco2014':
        num_classes = 80
        train_list = 'coco/train2014.txt'
        args.data_root += '/coco'
        from coco.data import ClsDataset
    elif args.dataset == 'coco2017':
        num_classes = 80
        train_list = 'coco/train2017.txt'
        args.data_root += '/coco'
        from coco.data import ClsDataset
    else:
        raise ValueError('Dataset "{}" not implemented'.format(args.dataset))

    model = getattr(importlib.import_module(args.network), 'Net')(num_classes)
    summary_writer = SummaryWriter(args.tblog_dir)
    
    # Create dataset
    train_dataset = ClsDataset(
        train_list, data_root=args.data_root,
        transform=transforms.Compose([
            imutils.RandomResizeLong(448, 768),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1),
            np.asarray,
            model.normalize,
            imutils.RandomCrop(args.crop_size),
            imutils.HWC_to_CHW,
            torch.from_numpy]))
    
    # Create data loader
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn)
    
    # Create optimizer
    max_step = len(train_dataset) // args.batch_size * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer(
        [{'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
         {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
         {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
         {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}],
        lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    # Load weights
    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)
    
    # Create model
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    
    # Metrics and timer
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_fs')
    timer = pyutils.Timer("Session started: ")
    
    # Training loop
    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):
            
            # Construct input and label
            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            N, C, H, W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            
            # Feed forward: compute CAMs and revised CAMs
            pred1, cam1, cam1_logit, cam1_rv, cam1_rv_logit = model(img1)
            pred2, cam2, cam2_logit, cam2_rv, cam2_rv_logit = model(img2)
            
            # Compute classification loss for revised CAMs
            assert args.rv_pooling in ['avg', 'max']
            if args.rv_pooling == 'avg':
                pred1_rv = F.adaptive_avg_pool2d(cam1_rv_logit, (1, 1))
                pred2_rv = F.adaptive_avg_pool2d(cam2_rv_logit, (1, 1))
                loss_cls1_rv = F.multilabel_soft_margin_loss(pred1_rv[:,1:,:,:], label[:,1:,:,:])
                loss_cls2_rv = F.multilabel_soft_margin_loss(pred2_rv[:,1:,:,:], label[:,1:,:,:])
            elif args.rv_pooling == 'max':
                pred1_rv = probutils.probs_to_logits(F.adaptive_max_pool2d(cam1_rv, (1, 1)))
                pred2_rv = probutils.probs_to_logits(F.adaptive_max_pool2d(cam2_rv, (1, 1)))
                loss_cls1_rv = F.multilabel_soft_margin_loss(pred1_rv, label)
                loss_cls2_rv = F.multilabel_soft_margin_loss(pred2_rv, label)
            
            # Compute classification loss based on pooling
            assert args.pooling in ['avg', 'max']
            if args.loss_weight != 1:
                if args.pooling == 'avg':
                    pred1_cls = F.adaptive_avg_pool2d(cam1_logit, (1, 1))
                    pred2_cls = F.adaptive_avg_pool2d(cam2_logit, (1, 1))
                    loss_cls1 = F.multilabel_soft_margin_loss(pred1_cls[:,1:,:,:], label[:,1:,:,:])
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2_cls[:,1:,:,:], label[:,1:,:,:])
                elif args.pooling == 'max':
                    loss_cls1 = F.multilabel_soft_margin_loss(pred1, label)
                    loss_cls2 = F.multilabel_soft_margin_loss(pred2, label)
            
            # Compute classification loss based on importance sampling
            if args.loss_weight != 0:
                pred1_is = probutils.sample_probs(cam1, num_samples=args.num_samples)
                pred2_is = probutils.sample_probs(cam2, num_samples=args.num_samples)
                pred1_is = probutils.probs_to_logits(pred1_is)
                pred2_is = probutils.probs_to_logits(pred2_is)
                loss_cls1_is = F.multilabel_soft_margin_loss(pred1_is[:,1:,:], label[:,1:,:,:].squeeze(-1))
                loss_cls2_is = F.multilabel_soft_margin_loss(pred2_is[:,1:,:], label[:,1:,:,:].squeeze(-1))
            
            # Compute feature similarity loss
            if args.fsl:
                n, c, h, w = cam1.size()
                img1_s = F.interpolate(img1, (h, w), mode='bilinear', align_corners=True).cuda(non_blocking=True)
                mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1_s.device)
                std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1_s.device)
                img1_s = img1_s*std_t + mean_t
                loss_fs = probutils.feature_similarity_loss(img1_s, cam1, args.fsl_mu, args.fsl_sigma)
            else:
                loss_fs = torch.tensor(0, dtype=torch.float32)
            
            # Downsample CAM1 to match CAM2
            _, _, hs, ws = cam2.size()
            cam1 = F.interpolate(cam1, (hs, ws), mode='bilinear', align_corners=True)
            cam1_rv = F.interpolate(cam1_rv, (hs, ws), mode='bilinear', align_corners=True)
            
            # Compute ER loss
            loss_er = torch.mean(torch.abs(cam1 - cam2)) + \
                      torch.mean(torch.abs(cam1_rv - cam2_rv)) + \
                      torch.mean(torch.abs(cam1 - cam1_rv)) + \
                      torch.mean(torch.abs(cam2 - cam2_rv))
            
            # Compute ECR loss
            loss_ecr = torch.mean(torch.abs(cam1 - cam2_rv)) + \
                       torch.mean(torch.abs(cam2 - cam1_rv))
            
            # Compute the final loss
            if args.loss_weight == 0:
                loss_cls = loss_cls1 + loss_cls2
            elif args.loss_weight == 1:
                loss_cls = loss_cls1_is + loss_cls2_is
            else:
                loss_cls = (1 - args.loss_weight) * (loss_cls1 + loss_cls2) + \
                           args.loss_weight * (loss_cls1_is + loss_cls2_is)
            loss_cls += 1.0*(loss_cls1_rv + loss_cls2_rv)
            loss = loss_cls + loss_er + loss_ecr + 0.5*loss_fs
            
            # Backpropagate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss averages
            avg_meter.add({
                'loss': loss.item(), 'loss_cls': loss_cls.item(),
                'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item(),
                'loss_fs': loss_fs.item()})
        
            # Print after each batch
            if args.verbose and (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_fs'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
                avg_meter.pop()
        
        # Print and update tensorboard after each epoch
        if args.verbose:
            avg_meter.add({
                'loss': loss.item(), 'loss_cls': loss_cls.item(),
                'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item(),
                'loss_fs': loss_fs.item()})
        timer.update_progress(optimizer.global_step / max_step)
        print('Epoch:%2d/%2d' % (ep+1, args.max_epoches),
              'loss:%.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_fs'),
              'imps:%.1f' % (len(train_dataset) / timer.get_stage_elapsed()),
              'Fin:%s' % (timer.str_est_finish()),
              'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
        if args.verbose:
            print()
        avg_meter.pop()
        loss_dict = {'loss': loss.item(),
                     'loss_cls': loss_cls.item(),
                     'loss_er': loss_er.item(),
                     'loss_ecr': loss_ecr.item(),
                     'loss_fs': loss_fs.item()}
        summary_writer.add_scalars('loss', loss_dict, ep+1)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep+1)
        timer.reset_stage()
    
    # Save the model when training is finished
    torch.save(model.module.state_dict(), args.session_name + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_cam", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="resnet38_cam", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--loss_weight", default=0.8, type=float)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--pooling", default='avg', type=str)
    parser.add_argument("--rv_pooling", default='max', type=str)
    parser.add_argument("--fsl_mu", default=2.5, type=float)
    parser.add_argument("--fsl_sigma", default=5.0, type=float)
    parser.add_argument('--fsl', dest='fsl', action='store_true')
    parser.add_argument('--no-fsl', dest='fsl', action='store_false')
    parser.set_defaults(fsl=True)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(args)
