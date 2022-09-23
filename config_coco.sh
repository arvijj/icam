#!/usr/bin/env bash

# Download data
echo Downloading training set
wget -nc http://images.cocodataset.org/zips/train2017.zip -P ./data/
echo Downloading validation set
wget -nc http://images.cocodataset.org/zips/val2017.zip -P ./data/
echo Downloading annotations
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./data/
wget -nc http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./data/

# Unzip data
echo Unzipping training set
unzip -n -q ./data/train2017.zip -d ./data/coco/
echo Unzipping validation set
unzip -n -q ./data/val2017.zip -d ./data/coco/
echo Unzipping annotations
unzip -n -q ./data/annotations_trainval2017.zip -d ./data/coco/
unzip -n -q ./data/annotations_trainval2014.zip -d ./data/coco/

# Create a single folder for the images
mv data/coco/train2017 data/coco/Images
mv data/coco/val2017/* data/coco/Images/
rm -r data/coco/val2017

# Compile COCO API
echo Compiling COCO API
rm -rf cocoapi/
cp -a ./cocoapi_compile/linux/. ./cocoapi/
cd cocoapi/PythonAPI/
make
cd -

# Process data
echo Processing data
python3 process_coco.py --data_type train2014 --no-gen_seg_masks
python3 process_coco.py --data_type train2017
python3 process_coco.py --data_type val2014 --no-gen_seg_masks
python3 process_coco.py --data_type val2017

# Create classification labels
python3 make_cls_labels.py \
    --dataset coco \
    --train_list coco/train2017.txt \
    --val_list coco/val2017.txt \
    --out data/coco/cls_labels.npy \
    --data_root data/coco

# Tar everything into one file
#tar -cf data/coco.tar -C data coco