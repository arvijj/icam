#!/bin/bash

dt=$(date +'%Y-%m-%d_%H%M%S')
EXPDIR="experiments/${dt}"
DATADIR="data/VOCdevkit/VOC2012"
DATASET="voc"
DATASETYEAR="2012"

# Create necessary directories
mkdir -p ${EXPDIR}/{ckpts,logs/{cam,aff,final},plabels/{train,val}/{cam,crf,rw}}

echo Starting CAM training
python3 train_cam.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --data_root data \
    --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --session_name ${EXPDIR}/ckpts/cam \
    --tblog_dir ${EXPDIR}/logs/cam

echo Starting CAM inference on validation set
python3 infer_cam.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --weights ${EXPDIR}/ckpts/cam.pth \
    --data_root data \
    --infer_list ${DATASET}/val.txt \
    --out_cam ${EXPDIR}/plabels/val/cam \
    --out_crf ${EXPDIR}/plabels/val/crf

echo Starting CAM evaluation on validation set
python3 eval_cam.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --eval_list ${DATADIR}/ImageSets/Segmentation/val.txt \
    --predict_dir ${EXPDIR}/plabels/val \
    --gt_dir ${DATADIR}/SegmentationClass

echo Computing CAM contour F-score on validation set
python3 contour_fscore.py \
    --pred_path ${EXPDIR}/plabels/val/cam \
    --img_list ${DATADIR}/ImageSets/Segmentation/val.txt \
    --gt_path ${DATADIR}/SegmentationClass \
    --is_cam \
    --num_classes 20

echo Starting CAM inference on training set
python3 infer_cam.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --weights ${EXPDIR}/ckpts/cam.pth \
    --data_root data \
    --infer_list ${DATASET}/train_aug.txt \
    --out_cam ${EXPDIR}/plabels/train/cam \
    --out_crf ${EXPDIR}/plabels/train/crf

echo Starting CAM evaluation on training set
python3 eval_cam.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --eval_list ${DATADIR}/ImageSets/Segmentation/train.txt \
    --predict_dir ${EXPDIR}/plabels/train/ \
    --gt_dir ${DATADIR}/SegmentationClass

echo Computing CAM contour F-score on training set
python3 contour_fscore.py \
    --pred_path ${EXPDIR}/plabels/train/cam \
    --img_list ${DATADIR}/ImageSets/Segmentation/train.txt \
    --gt_path ${DATADIR}/SegmentationClass \
    --is_cam \
    --num_classes 20

echo Starting AffinityNet training
python3 train_aff.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --weights pretrained/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --data_root data \
    --la_crf_dir ${EXPDIR}/plabels/train/crf/alpha_2.0 \
    --ha_crf_dir ${EXPDIR}/plabels/train/crf/alpha_4.0 \
    --session_name ${EXPDIR}/ckpts/aff

echo Starting AffinityNet inference on validation set
python3 infer_aff.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --weights ${EXPDIR}/ckpts/aff.pth \
    --infer_list ${DATASET}/val.txt \
    --cam_dir ${EXPDIR}/plabels/val/cam \
    --data_root data \
    --out_rw ${EXPDIR}/plabels/val/rw

echo Starting AffinityNet evaluation on validation set
python3 eval_aff.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --eval_list ${DATADIR}/ImageSets/Segmentation/val.txt \
    --predict_dir ${EXPDIR}/plabels/val/rw \
    --gt_dir ${DATADIR}/SegmentationClass

echo Computing AffinityNet contour F-score on validation set
python3 contour_fscore.py \
    --pred_path ${EXPDIR}/plabels/val/rw \
    --img_list ${DATADIR}/ImageSets/Segmentation/val.txt \
    --gt_path ${DATADIR}/SegmentationClass \
    --no-is_cam \
    --num_classes 20

echo Starting AffinityNet inference on training set
python3 infer_aff.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --weights ${EXPDIR}/ckpts/aff.pth \
    --infer_list ${DATASET}/train_aug.txt \
    --cam_dir ${EXPDIR}/plabels/train/cam \
    --data_root data \
    --out_rw ${EXPDIR}/plabels/train/rw

echo Starting AffinityNet evaluation on training set
python3 eval_aff.py \
    --dataset ${DATASET}${DATASETYEAR} \
    --eval_list ${DATADIR}/ImageSets/Segmentation/train.txt \
    --predict_dir ${EXPDIR}/plabels/train/rw \
    --gt_dir ${DATADIR}/SegmentationClass

echo Computing AffinityNet contour F-score on training set
python3 contour_fscore.py \
    --pred_path ${EXPDIR}/plabels/train/rw \
    --img_list ${DATADIR}/ImageSets/Segmentation/train.txt \
    --gt_path ${DATADIR}/SegmentationClass \
    --no-is_cam \
    --num_classes 20

# Rename the relevant config file to 'config.py'
cp ${DATASET}/config_${DATASET}${DATASETYEAR}.py config.py

echo Starting final model training
python3 train_final.py \
    --exp-dir ${EXPDIR} \
    --data-pseudo-gt ${EXPDIR}/plabels/train/rw

echo Starting final model evaluation on validation set
python3 eval_final.py \
    --period val \
    --exp-dir ${EXPDIR} \
    --data-pseudo-gt ${EXPDIR}/plabels/val/rw

echo Computing final model contour F-score on validation set
python3 contour_fscore.py \
    --pred_path ${EXPDIR}/results/val \
    --img_list ${DATADIR}/ImageSets/Segmentation/val.txt \
    --gt_path ${DATADIR}/SegmentationClass \
    --no-is_cam \
    --num_classes 20

echo Starting final model evaluation on training set
python3 eval_final.py \
    --period train \
    --exp-dir ${EXPDIR} \
    --data-pseudo-gt ${EXPDIR}/plabels/train/rw

echo Computing final model contour F-score on training set
python3 contour_fscore.py \
    --pred_path ${EXPDIR}/results/train \
    --img_list ${DATADIR}/ImageSets/Segmentation/train.txt \
    --gt_path ${DATADIR}/SegmentationClass \
    --no-is_cam \
    --num_classes 20