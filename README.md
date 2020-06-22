# Mnemonics Training

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8?style=flat-square)](https://pytorch.org/)

This repository contains the PyTorch implementation for [CVPR 2020](http://cvpr2020.thecvf.com/) Paper "[Mnemonics Training: Multi-Class Incremental Learning without Forgetting](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Mnemonics_Training_Multi-Class_Incremental_Learning_Without_Forgetting_CVPR_2020_paper.pdf)". If you have any questions on this repository or the related paper, feel free to create an issue or send me an email. 

#### Summary

* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Running Experiments](#running-experiments)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Introduction

Multi-Class Incremental Learning (MCIL) aims to learn new concepts by incrementally updating a model trained on previous concepts. However, there is an inherent trade-off to effectively learning new concepts without catastrophic forgetting of previous ones. To alleviate this issue, it has been proposed to keep around a few examples of the previous concepts but the effectiveness of this approach heavily depends on the representativeness of these examples. This paper proposes a novel and automatic framework we call ***mnemonics***, where we parameterize exemplars and make them optimizable in an end-to-end manner. We train the framework through bilevel optimizations, i.e., model-level and exemplar-level. We conduct extensive experiments on three MCIL benchmarks, CIFAR-100, ImageNet-Subset and ImageNet, and show that using ***mnemonics*** exemplars can surpass the state-of-the-art by a large margin. Interestingly and quite intriguingly, the ***mnemonics*** exemplars tend to be on the boundaries between classes.


<p align="center">
    <img src="https://yyliu.net/images/misc/mnemonics.png" width="600"/>
</p>

> Figure: The t-SNE results of three exemplar methods in two phases. The original data of 5 colored classes occur in the early phase. In each colored class, deep-color points are exemplars, and light-color ones show the original data as reference of the real data distribution. Gray crosses represent other participating classes, and each cross for one class. We have two main observations. (1) Our approach results in much clearer separation in the data, than random (where exemplars are randomly sampled in the early phase) and herding (where exemplars are nearest neighbors of the mean sample in the early phase). (2) Our learned exemplars mostly locate on the boundaries between classes.

## Dependencies

- [Python](https://www.python.org/) >= 3.6
- [PyTorch](https://pytorch.org/) >= 0.4.0
- [Pillow](https://pillow.readthedocs.io/en/stable/) >= 6.2.1
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [scipy](https://www.scipy.org/)


## Running Experiments


### Clone this repository

```bash
cd ~
git clone git@github.com:yaoyao-liu/mnemonics-training.git
```

### Processing the datasets

Download the [`ImageNet (ILSVRC2012)`](http://www.image-net.org/) source file.

Process `ImageNet-Sub` and `ImageNet`:
```bash
cd ~/mnemonics-training/eval/process_imagenet
python generate_imagenet_subset.py
python generate_imagenet.py
```

[`CIFAR-100`](https://www.cs.toronto.edu/~kriz/cifar.html) will be downloaded automatically.

### Download models

Download the models for CIFAR-100, ImageNet-Sub and ImageNet:
```bash
cd ~/mnemonics-training/eval
sh ./script/download_ckpt.sh
```

You may also download the checkpoints on [Google Drive](https://drive.google.com/file/d/1sKO2BOssWgTFBNZbM50qDzgk6wqg4_l8/view?usp=sharing).

### Evaluation on our models

Run evaluation code with our models：
```bash
cd ~/mnemonics-training/eval
sh run_eval.sh
```

### Training from scratch

Run the experiment using v1.0:
```bash
cd ~/mnemonics-training/train_v1.0
python run_exp.py
```

Run the experiment using v2.0 (with new features):
```bash
cd ~/mnemonics-training/train_v2.0
python main.py
```

## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{liu2020mnemonics,
author    = {Liu, Yaoyao and Su, Yuting and Liu, An{-}An and Schiele, Bernt and Sun, Qianru},
title     = {Mnemonics Training: Multi-Class Incremental Learning without Forgetting},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
pages     = {12245--12254},
year      = {2020}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

* [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)

* [Dataset Distillation](https://github.com/SsnL/dataset-distillation)

* [Generative Teaching Networks](https://github.com/uber-research/GTN)
