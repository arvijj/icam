# ----------------------------------------
# Written by Yude Wang
# Modified by Arvi Jonnarth
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *
from utils.imutils import *
from utils.registry import DATASETS
from datasets.BaseDataset import BaseDataset

@DATASETS.register_module
class COCODataset(BaseDataset):
    def __init__(self, cfg, period, transform='none'):
        super(COCODataset, self).__init__(cfg, period, transform)
        self.dataset_name = 'coco%d'%cfg.DATA_YEAR
        self.data_dir = os.path.join(cfg.DATA_DIR)
        self.dataset_dir = os.path.join(self.data_dir,self.dataset_name)
        self.rst_dir = os.path.join(cfg.EXP_DIR,'results',period)
        self.eval_dir = os.path.join(cfg.EXP_DIR,'eval_result')
        self.img_dir = os.path.join(self.dataset_dir, 'Images')
        #self.ann_dir = os.path.join(self.dataset_dir, 'annotations')
        self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass')
        self.set_dir = os.path.join(self.dataset_dir)
        if cfg.DATA_PSEUDO_GT:
            self.pseudo_gt_dir = cfg.DATA_PSEUDO_GT
        else:
            #self.pseudo_gt_dir = os.path.join(self.data_dir,'pseudo_gt',self.dataset_name,'Segmentation')
            pass

        file_name = None
        if cfg.DATA_AUG and 'train' in self.period:
            file_name = self.set_dir+'/'+period+'aug.txt'
        else:
            file_name = self.set_dir+'/'+period+'.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        if self.dataset_name == 'coco2017':
            self.categories = \
                ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
                'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            #self.coco2voc = [[0],[5],[2],[16],[9],[44],[6],[3],[17],[62],
            #                 [21],[67],[18],[19],[4],[1],[64],[20],[63],[7],[72]]

            self.num_categories = len(self.categories)+1
            self.cmap = self.__colormap(len(self.categories)+1)
        
        if self.dataset_name == 'coco2014':
            self.categories = \
                ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
                'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            #self.coco2voc = [[0],[5],[2],[16],[9],[44],[6],[3],[17],[62],
            #                 [21],[67],[18],[19],[4],[1],[64],[20],[63],[7],[72]]

            self.num_categories = len(self.categories)+1
            self.cmap = self.__colormap(len(self.categories)+1)

    def __len__(self):
        return len(self.name_list)

    def load_name(self, idx):
        name = self.name_list[idx]
        return name

    def load_image(self, idx):
        name = str(self.name_list[idx]).zfill(12)
        img_file = self.img_dir + '/' + name + '.jpg'
        image = cv2.imread(img_file)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def load_segmentation(self, idx):
        name = str(self.name_list[idx]).zfill(12)
        seg_file = self.seg_dir + '/' + name + '.png'
        segmentation = np.array(Image.open(seg_file))
        return segmentation

    def load_pseudo_segmentation(self, idx):
        name = str(self.name_list[idx]).zfill(12)
        seg_file = self.pseudo_gt_dir + '/' + name + '.png'
        segmentation = np.array(Image.open(seg_file))
        return segmentation

    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap

    def load_ranked_namelist(self):
        df = self.read_rank_result()
        self.name_list = df['filename'].values

    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        cmap[m==255] = [255,255,255]
        return cmap

    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        if not os.path.exists(self.rst_dir):
            os.makedirs(self.rst_dir)

        for sample in result_list:
            if isinstance(sample['name'], torch.Tensor):
                file_path = os.path.join(self.rst_dir, '%s.png'%str(sample['name'].item()).zfill(12))
            else:
                file_path = os.path.join(self.rst_dir, '%s.png'%str(sample['name']).zfill(12))
            cv2.imwrite(file_path, sample['predict'])

    def save_pseudo_gt(self, result_list, folder_path=None):
        """Save pseudo gt

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = self.pseudo_gt_dir if folder_path is None else folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s.png'%(sample['name']))
            cv2.imwrite(file_path, sample['predict'])
            i+=1

    #def do_matlab_eval(self, model_id):
    #    import subprocess
    #    path = os.path.join(self.data_dir, 'VOCcode')
    #    eval_filename = os.path.join(self.eval_dir,'%s_result.mat'%model_id)
    #    cmd = 'cd {} && '.format(path)
    #    cmd += 'matlab -nodisplay -nodesktop '
    #    cmd += '-r "dbstop if error; VOCinit; '
    #    cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
    #    cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
    #    cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
    #    cmd += 'quit;"'
    #
    #    print('start subprocess for matlab evaluation...')
    #    print(cmd)
    #    subprocess.call(cmd, shell=True)

    def do_python_eval(self, model_id):
        gt_folder = self.seg_dir
        TP = []
        P = []
        T = []
        for i in range(self.num_categories):
            TP.append(multiprocessing.Value('d', 0, lock=True))
            P.append(multiprocessing.Value('d', 0, lock=True))
            T.append(multiprocessing.Value('d', 0, lock=True))

        def compare(start,step,TP,P,T):
            for idx in range(start,len(self.name_list),step):
                #print('%d/%d'%(idx,len(self.name_list)))
                name = self.name_list[idx]
                predict_file = os.path.join(self.rst_dir,'%s.png'%str(name).zfill(12))
                gt_file = os.path.join(gt_folder,'%s.png'%str(name).zfill(12))
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                gt = np.array(Image.open(gt_file))
                cal = gt<255
                mask = (predict==gt) * cal

                for i in range(self.num_categories):
                    P[i].acquire()
                    P[i].value += np.sum((predict==i)*cal) / 100000 # avoid overflow
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt==i)*cal) / 100000 # avoid overflow
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt==i)*mask) / 100000 # avoid overflow
                    TP[i].release()
        p_list = []
        for i in range(32):
            p = multiprocessing.Process(target=compare, args=(i,32,TP,P,T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.num_categories):
            IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        loglist = {}
        for i in range(self.num_categories):
            if i == 0:
                print('%11s:%7.3f%%'%('background',IoU[i]*100),end='\t')
                loglist['background'] = IoU[i] * 100 
            else:
                if i%2 != 1:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100),end='\t')
                else:
                    print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
                loglist[self.categories[i-1]] = IoU[i] * 100

        miou = np.mean(np.array(IoU))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        loglist['mIoU'] = miou * 100
        return loglist

    #def __coco2voc(self, m):
    #    r,c = m.shape
    #    result = np.zeros((r,c),dtype=np.uint8)
    #    for i in range(0,21):
    #        for j in self.coco2voc[i]:
    #            result[m==j] = i
    #    return result


