# Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors

PyTorch implementation of the paper 
```
"Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors". 
Peri Akiva, Kristin Dana, Peter Oudomous, Michael Mars. CVPRW2020. 
```
[This](https://arxiv.org/pdf/2004.08501.pdf "Link") is the link to the paper. This repository also features a docker container setup for training, evaluation and testing. 

## Citing this work

```
@article{akiva2020finding,
  title={Finding Berries: Segmentation and Counting of Cranberries using Point Supervision and Shape Priors},
  author={Peri Akiva and Kristin Dana and Peter Oudomous and Michael Mars},
  journal={Proceedings of the Computer Vision and Pattern Recognition Workshops(CVPRW)},
  month={June},
  year={2020}
}
```

## Dataset
We use the CRAID (CRanberry Aerial Imagery Dataset) in our experiments. The dataset can be downloaded from:

\item [CRAID](https://forms.gle/zfFCKy1pyDD4WNro7)

## Setting up the environment
We can set up an environment in 2 ways: locally, or in a docker container.

### Locally

We use Python 3.6. We recommend using Anaconda ([Available here](https://www.anaconda.com/)) to manage and install libraries. Once anaconda is installed, run this command:

```
conda create --name finding_berries --file environment.txt
```




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

##### Can install requirements using:
```
$ conda create --name cranberry_counting --file requirements.txt
```# finding_berries
# finding_berries
