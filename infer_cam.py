import argparse
import cv2
import importlib
import numpy as np
import os
import pandas as pd
import scipy.misc
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from tool import imutils, pyutils, visualization, probutils

def main(args):

    if args.dataset == 'voc2012':
        num_classes = 20
        args.data_root += '/VOCdevkit/VOC2012'
        from voc.data import ClsDatasetMSF, get_img_path
    elif args.dataset == 'coco2014':
        num_classes = 80
        args.data_root += '/coco'
        from coco.data import ClsDatasetMSF, get_img_path
    elif args.dataset == 'coco2017':
        num_classes = 80
        args.data_root += '/coco'
        from coco.data import ClsDatasetMSF, get_img_path
    else:
        raise ValueError('Dataset "{}" not implemented'.format(args.dataset))
    
    # Load model
    model = getattr(importlib.import_module(args.network), 'Net')(num_classes)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.cuda()
    
    # Create data loader
    infer_dataset = ClsDatasetMSF(
        args.infer_list, data_root=args.data_root, scales=[0.5, 1.0, 1.5, 2.0],
        inter_transform=torchvision.transforms.Compose(
            [np.asarray, model.normalize, imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Run inference with one model per GPU
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

    # Run inference
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        if args.verbose and iter % 100 == 0:
            print('Iter: ' + str(iter) + '/' + str(len(infer_data_loader)), flush=True)
        
        # Read image
        img_name = img_name[0]; label = label[0]
        img_path = get_img_path(img_name, args.data_root)
        orig_img = np.asarray(Image.open(img_path).convert("RGB"))
        orig_img_size = orig_img.shape[:2]

        # Define inference work method
        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i%n_gpus):
                    _, cam, _, cam_rv, _ = model_replicas[i%n_gpus](img.cuda())
                    cam = 0.5*cam + 0.5*cam_rv
                    lbl = torch.cat([torch.tensor([1.]), label]).view(1, num_classes+1, 1, 1).cuda()
                    cam = probutils.label_cond(cam, lbl)[:, 1:, :, :]
                    cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy()
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        # Perform inference
        thread_pool = pyutils.BatchThreader(
            _work, list(enumerate(img_list)), batch_size=12,
            prefetch_size=0, processes=args.num_workers)
        cams = thread_pool.pop_results()
        norm_cam = np.mean(cams, axis=0)
        norm_cam[norm_cam != norm_cam.max(axis=0, keepdims=True)] = 0
        norm_cam = norm_cam / (norm_cam.max(axis=(1, 2), keepdims=True) + 1e-5)

        # Keep CAMs of present classes
        cam_dict = {}
        for i in range(num_classes):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        # Save CAMs to disk
        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)
        
        crf_alpha = [2, 4]
        
        # Define CRF function
        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            if len(v.shape) != 3:
                return {0: np.ones((orig_img.shape[0], orig_img.shape[1]), dtype=np.float32)}
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])
            n_crf_al = dict()
            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]
            return n_crf_al

        # Perform CRF
        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t)
                folder = os.path.join(args.out_crf, 'alpha_%.1f'%t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, img_name + '.npy'), crf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_cam", type=str)
    parser.add_argument("--infer_list", required=True, type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    main(args)
