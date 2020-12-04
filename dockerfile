FROM nvidia/cuda:10.2-base-ubuntu18.04
ARG PYTHON_VERSION=3.7

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt install libgl1-mesa-glx -y

# Create a working directory
RUN mkdir /app
WORKDIR /app

RUN set -xe \
    && apt-get update -y \
    && apt-get -y install python-pip

RUN apt-get install vim -y

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
USER root

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user



# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==${PYTHON_VERSION} numpy \
 && conda clean -ya

# CUDA 10.2-specific steps
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# RUN conda install -y -c pytorch \
#     cudatoolkit=10.2 \
#     "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \
#     "torchvision=0.6.0=py38_cu102" \
#  && conda clean -ya

RUN conda install -y -c anaconda pandas
RUN conda install -c conda-forge configargparse 
RUN conda install -c conda-forge matplotlib 
RUN conda install -c conda-forge pyyaml 
RUN conda install -c conda-forge munch 
RUN conda install -c conda-forge parse 
RUN conda install -c anaconda scipy 
RUN conda install -c conda-forge scikit-learn 
RUN conda install -c anaconda scikit-image 
RUN conda install -c conda-forge opencv 
RUN conda install -c anaconda pip
RUN pip install comet-ml
RUN pip install pretrainedmodels
RUN pip install efficientnet-pytorch
RUN pip install peterpy
# RUN conda install -y -c comet_ml comet_ml
# Set the default command to python3
# CMD ["python3"]

ADD . finding_berries/

WORKDIR /app/finding_berries/

# CMD ["python","train.py"]

EXPOSE 3000



