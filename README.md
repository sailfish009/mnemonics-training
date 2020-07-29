# Mnemonics Training

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/mnemonics/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-0.4.0-%237732a8?style=flat-square)](https://pytorch.org/)

This repository contains the PyTorch implementation for "Mnemonics Training: Multi-Class Incremental Learning without Forgetting". If you have any questions on this repository or the related paper, feel free to create an issue or send me an email. 

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

### Running experiments for baselines

```bash
cd 1_train
python main.py --nb_cl_fg=50 --method=baseline --nb_cl=10
python main.py --nb_cl_fg=50 --method=baseline --nb_cl=5
python main.py --nb_cl_fg=50 --method=baseline --nb_cl=2
```

### Running experiments for our method

```bash
cd 1_train
python main.py --nb_cl_fg=50 --method=mnemonics --nb_cl=10
python main.py --nb_cl_fg=50 --method=mnemonics --nb_cl=5
python main.py --nb_cl_fg=50 --method=mnemonics --nb_cl=2
```

### Performance

#### Average accuracy (%)

| Method          | Dataset   | 5-phase     | 10-phase     | 25-phase    | 
| ----------      | --------- | ----------  | ----------   |------------ |
| [LwF](https://arxiv.org/abs/1606.09282)  | CIFAR-100 | 52.44  | 48.47   | 45.75 |
| [LwF](https://arxiv.org/abs/1606.09282) w/ ours  | CIFAR-100 | 54.21  | 52.72   | 51.59 |
| [iCaRL](https://arxiv.org/abs/1611.07725)  | CIFAR-100 | 58.03  | 53.01  | 48.47 |
| [iCaRL](https://arxiv.org/abs/1611.07725) w/ ours | CIFAR-100 | 60.01  | 57.37   | 54.13 |

#### Forgetting rate (%, lower is better)

| Method          | Dataset   | 5-phase     | 10-phase     | 25-phase    | 
| ----------      | --------- | ----------  | ----------   |------------ |
| [LwF](https://arxiv.org/abs/1606.09282)  | CIFAR-100 | 45.02  | 42.50   | 39.86 |
| [LwF](https://arxiv.org/abs/1606.09282) w/ ours  | CIFAR-100 | 40.00  | 36.50   | 34.25 |
| [iCaRL](https://arxiv.org/abs/1611.07725)  | CIFAR-100 | 32.87  | 32.98 | 36.32 |
| [iCaRL](https://arxiv.org/abs/1611.07725) w/ ours | CIFAR-100 | 25.93  | 26.92   | 28.92 |

> We find some bugs in the code for LUCIR w/ ours. <br /> We'll update it later after we've fixed this issue.

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
