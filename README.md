# ICAM

This repository contains the implementation of **Importance sampling CAMs for Weakly-Supervised Segmentation with Highly Accurate Contours**, [arXiv](https://arxiv.org/pdf/2203.12459.pdf), 2022.

## Abstract

Classification networks have been used in weakly-supervised semantic segmentation (WSSS) to segment objects by means of class activation maps (CAMs). However, without pixel-level annotations, they are known to (1) mainly focus on discriminative regions, and (2) to produce diffuse CAMs without well-defined prediction contours. In this work, we alleviate both problems by improving CAM learning. First, we incorporate importance sampling based on the class-wise probability mass function induced by the CAMs to produce stochastic image-level class predictions. This results in segmentations that cover a larger extent of the objects, as shown in our empirical studies. Second, we formulate a feature similarity loss term, which further improves the alignment of predicted contours with edges in the image. Furthermore, we shed new light onto the problem of WSSS by measuring the contour F-score as a complement to the common area mIoU metric. We show that our method significantly outperforms previous methods in terms of contour quality, while matching state-of-the-art on region similarity.

## Installation

Install with pip:
- `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

Install on a singularity container:
- [Install singularity](https://sylabs.io/guides/3.0/user-guide/quick_start.html)
- `singularity build --fakeroot pytorch.sif pytorch.def`

## Data preparation

Download and configure data:
- VOC: `./config_voc.sh`
- COCO: `./config_coco.sh`

## Download pretrained weights

Download pretrained weights from [here](https://drive.google.com/drive/folders/1y3WjoQLgqbR4q7u9KWPjj4enlLvbn_-P?usp=sharing) and put them in a new folder named `pretrained`. The folder contains the following weights:
- `ilsvrc-cls_rna-a1_cls1000_ep-0001.params` - ImageNet pre-trained weights for ResNet-38
- `cam.pth` - CAM network weights
- `aff.pth` - AffinityNet weights
- `final.pth` - Final DeepLab-v1 model weights
- `final_et_res38.pth` - Final model weights with extended training
- `final_et_r2n.pth` - Final model weights with extended training, Res2Net-101 backbone

## Training and evaluation

Train/infer/evaluate on all three stages CAM/AffinityNet/final:
- VOC 2012: `./run_all_voc.sh`
- COCO 2014/2017: `./run_all_coco.sh`

Parameter configurations are handled by `argparse`. For the final stage, config files are also used to set some of the parameters, except for the ImageNet pretrained model path which needs to be set manually in `lib/net/backbone/resnet38d.py`. The config files are:
- `voc/config_voc2012.py`
- `coco/config_coco2014.py`
- `coco/config_coco2017.py`

For training on multiple GPUs, update the `GPUS` field in the config file to match the number of available GPUs on your system.

For training with the extended training setting from PMM \[4\], see [WSSS_MMSeg](https://github.com/Eli-YiLi/WSSS_MMSeg).

## Acknowledgements

This code was based on the following repositories:
- [jiwoon-ahn/irn](https://github.com/jiwoon-ahn/irn) \[1\]
- [YudeWang/SEAM](https://github.com/YudeWang/SEAM) \[2\]
- [YudeWang/semantic-segmentation-codebase](https://github.com/YudeWang/semantic-segmentation-codebase) \[2\]
- [davisvideochallenge/davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation/) \[3\]
- [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)

## References

\[1\] Jiwoon Ahn and Suha Kwak. Learning Pixel-Level Semantic Affinity with Image-Level Supervision for Weakly Supervised Semantic Segmentation. CVPR, 2018.

\[2\] Yude Wang, Jie Zhang, Meina Kan, Shiguang Shan, and Xilin Chen. Self-Supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation. CVPR, 2020.

\[3\] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. CVPR, 2016.

\[4\] Yi Li, Zhanghui Kuang, Liyang Liu, Yimin Chen, and Wayne Zhang. Pseudo-mask Matters in Weakly-supervised Semantic Segmentation. ICCV, 2021.
