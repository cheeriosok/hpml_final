# DRCT: Upscaling the Dense Residual Connection Transformer

High Performance Machine Learning, New York University

## Overview

Raw Training Data can be viewed in the experiments directory as Experiment_[i].out where: 

Experiment 1 = Pre-Norm DRCT 4x Upscaling on DIV2K  
Experiment 2 = Post-Norm DRCT 4x Upscaling on DIV2K  
Experiment 3 = Post-Norm + Cos + DRCT 4x Upscaling on DIV2K  

![download](https://github.com/oscarabreu/hpml_final/assets/99779654/c7314bc3-740c-4b33-b589-e0d13f256655)

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8!!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 

### Installation
```
git clone 
conda create --name drctv2 python=3.11 -y
conda activate drct
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop
```

## How To Test
- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- Then run the following codes (taking `DRCT_SRx4.pth` as an example):
```
python drct/test.py -opt options/test/DRCT_SRx4.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/DRCT_SRx4.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/DRCT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Validation data can be download at [this page](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 drct/train.py -opt options/train/train_DRCT_SRx2_from_scratch.yml --launcher pytorch
```
The training logs and weights will be saved in the `./experiments` folder.

## Citations
#### BibTeX
    @misc{hsu2024drct,
      title={DRCT: Saving Image Super-resolution away from Information Bottleneck}, 
      author={Chih-Chung Hsu and Chia-Ming Lee and Yi-Shiuan Chou},
      year={2024},
      eprint={2404.00722},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

