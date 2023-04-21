## Quantization Steps Aware Method
Official PyTorch implementation of **Picking Up Quantization Steps for Compression Image Classification**.

## 1.The sensitivity of deep neural networks to compressed images.
![a](https://github.com/LiMaPKU/QSAM/blob/main/imgs/ImageNet_Samples/Q75.png)
More samples are shown in `imgs/ImageNet_samples/Q100-10.png`

## 2. Requirements
### Environments
Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- tensorboard 2.5.0+

## 3. Training & Evaluation
To train the models in paper, run this command:
```train and evaluation
python main.py --base_lr 0.1 --meta_lr 0.004 --base_epochs 200 --meta_epochs 150 --dataset cifar100
```

## 4. Results
Main results on CIFAR-10-J and CIFAR-100-J.

| ![2](https://github.com/LiMaPKU/QSAM/blob/main/imgs/results/CIFAR-10-J.png)   | ![z](https://github.com/LiMaPKU/QSAM/blob/main/imgs/results/CIFAR-100-J.png) |
| ------------------------------ | ---------------------------- |
| ![2](https://github.com/LiMaPKU/QSAM/blob/main/imgs/results/CIFAR-10-J-100-70.png)   | ![z](https://github.com/LiMaPKU/QSAM/blob/main/imgs/results/CIFAR-100-J-100-70.png) |
| ------------------------------ | ---------------------------- |

## 5. Citation
If you find QSAM useful in your work, please cite the following source:

```
@ARTICLE{ma2023qsam,
title={Picking Up Quantization Steps for Compressed Image Classification},
author={Ma, Li and Peng, Peixi and Chen, Guangyao and Zhao, Yifan and Dong, Siwei and Tian, Yonghong},
journal={IEEE Transactions on Circuits and Systems for Video Technology},
year={2023},
volume={33},
number={4},
pages={1884-1898},
}
```
