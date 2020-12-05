# Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors

PyTorch implementation of the paper 
```
"Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors". 
Peri Akiva, Kristin Dana, Peter Oudemans, Michael Mars. CVPRW2020. 
```
[[Link to paper]](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Akiva_Finding_Berries_Segmentation_and_Counting_of_Cranberries_Using_Point_Supervision_CVPRW_2020_paper.pdf "Link"),  [[Link to project site]](https://periakiva.github.io/finding_berries/). This repository also features a docker container setup for training, evaluation and testing. 

## Citing this work

```
@InProceedings{Akiva_2020_CVPR_Workshops,
author = {Akiva, Peri and Dana, Kristin and Oudemans, Peter and Mars, Michael},
title = {Finding Berries: Segmentation and Counting of Cranberries Using Point Supervision and Shape Priors},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
} 
```

## Dataset
We use the CRAID (CRanberry Aerial Imagery Dataset) in our experiments. The dataset can be downloaded from:

  * [CRAID](https://forms.gle/gAonKq5PMhmifgxr9)

## Setting up the environment
We can set up an environment in 2 ways: locally, or in a docker container.

### Locally

We use Python 3.6. We recommend using Anaconda ([Available here](https://www.anaconda.com/)) to manage and install libraries. Once anaconda is installed, run those commands:

```
conda create --name finding_berries --file environment.txt
conda activate finding_berries
pip install .
```

### Docker

This project can also be ran in a docker container. First install docker and nvidia-docker on your machine ([docker](https://docs.docker.com/get-docker/), [docker-nvidia](https://github.com/NVIDIA/nvidia-docker)). Then:

```
git clone git@github.com:periakiva/finding_berries.git
cd finding_berries
mkdir data
cd data
mkdir checkpoints
cd checkpoints
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Y1FxmQmiypD145I327G8o82EgAL_q2eo' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Y1FxmQmiypD145I327G8o82EgAL_q2eo" -O floodfill_epoch_43_mean_iou_0.6254406128186686_best_mae_13.459821428571429_mioumao_ratio_4.646722960908185_time_2020-03-09-13:23:56.pth && rm -rf /tmp/cookies.txt
```


Download and save data to the data direcotory. The data should be in 2 or 3 directories (depending on functionality): images, masks, back-masks. Generally, only the images and masks directories are necessary where masks have both berry and background points. 

```
sudo docker build -t finding_berries .
sudo docker run --gpus all -it -p80:3000 finding_berries
```
Then you can run any of the scripts on the docker container. Note that you will need to set up the config file with respect to which approach you use. Sample config file can be seen below.

In the case you use

## Usage
In order to run this code, adjustments should be made to the config file in training and evaluation code. Sample config file:

```
# Ours

location: docker #[local, docker]
data:
  image_size: 456x608 # size of image. if image size is different than this, need to train from scratch, and not use pretrained model.
  local:
    type: semantic_seg # type of dataset to use. [semantic_seg,instance_seg,points]
    model_save_dir: /path/to/directory/where/model/is/saved
    train_dir: /path/to/train/data/
    test_dir: /path/to/test/data/
    val_dir: /path/to/val/data/
  docker:
    type: semantic_seg
    model_save_dir: /app/finding_berries/models/
    train_dir: /app/finding_berries/data/instance_points/
    test_dir: /app/finding_berries/data/semantic_seg/
    val_dir:
  
training: 
  learning_rate: 0.001 
  resume: False # if true, insert path to the model to be resumed. Otherwise this will train from scratch

  train_val_test_split: [0.9,0.05,0.05] #split of the data
  epochs: 200
  batch_size: 1 # what batch size to use
  optimizer: adam # which optimizer to use
  num_workers: # number of workers

  loss_weights: {seg: 3.0, instance: 3.0, convexity: 10.0, circularity: 10.0, count: 0.2}
  class_weights: {seg: [1,3,'mean'], instance: [6,1,'mean']}

  losses_to_use: ["instance","circularity", "count_detect"]
  test_with_full_supervision: 1

use_cuda: True

```
After config file is modified, simply run the train/evaluate script that is in the same directory. Some paths might be coded into the script, so error may occur. TODO: this should be fixed in the docker approach




#### Requirements:

```
ballpark=1.4.0
imageio=2.6.1
joblib=0.14.0
munch=2.5.0
ninja=1.9.0
numpy=1.17.3
opencv=4.1.0
pandas=0.25.3
parse=1.12.1
peterpy=1.0.1
pillow=6.2.1
pyqt=5.6.0
python=3.6.5
pytorch=1.3.1
pyyaml=5.1.2
requests=2.22.0
scikit-learn=0.22
scipy=1.3.2
setuptools=42.0.2
torchvision=0.4.2
tqdm=4.40.0
urllib3=1.25.7
yaml=0.1.7
```


