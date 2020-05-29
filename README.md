# Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors

PyTorch implementation of the paper 
```
"Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors". 
Peri Akiva, Kristin Dana, Peter Oudemans, Michael Mars. CVPRW2020. 
```
[[Link to paper]](https://arxiv.org/pdf/2004.08501.pdf "Link"),  [[Link to project site]](https://periakiva.github.io/finding_berries/). This repository also features a docker container setup for training, evaluation and testing. 

## Citing this work

```
@article{akiva2020finding,
  title={Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors},
  author={Peri Akiva and Kristin Dana and Peter Oudemans and Michael Mars},
  journal={Proceedings of the Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month={June},
  year={2020}
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

## Usage
In order to run this code, adjustments should be made to the config file in training and evaluation code. Sample config file:

```
# Ours
data:
  type: semantic_seg # type of dataset to use. [semantic_seg,instance_seg,points]
  model_save_dir: /path/to/directory/where/model/is/saved
  train_dir: /path/to/train/data/
  test_dir: /path/to/test/data/
  val_dir: /path/to/val/data/
  image_size: 456x608 # size of image. if image size is different than this, need to train from scratch, and not use pretrained model.

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

### Docker




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


