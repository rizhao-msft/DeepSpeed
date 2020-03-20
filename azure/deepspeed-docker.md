## Use these commands to push a container image to Azure Container Registry
## Create registry with username rizhaoRegistry
# az acr create --resource-group BrainwaveAI --name rizhaoRegistry --sku Basic
## Build and push image
# az acr build --image megatron-deepspeed --registry rizhaoRegistry --file deepspeed-docker.md .
## Test
# az acr run --registry rizhaoRegistry --cmd 'rizhaoregistry.azurecr.io/megatron-deepspeed:latest' /dev/null

FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu16.04

# NCCL 2.4 does not work with PyTorch, uninstall
RUN apt-get update && apt-get --purge remove libnccl2 -y --allow-change-held-packages

RUN apt-get -y update && apt-get -y install --no-install-recommends libnccl2=2.4.7-1+cuda10.0 libnccl-dev=2.4.7-1+cuda10.0

RUN [ "/bin/bash", "-c", "conda create -n deepspeed Python=3.6.2 && source activate deepspeed && conda install pip"]

RUN ldconfig /usr/local/cuda/lib64/stubs && \
    # Install GPUtil
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir GPUtil && \
    # Install AzureML SDK
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir azureml-defaults && \
    # Install PyTorch
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir torch==1.2 && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir torchvision==0.4.0 && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir pillow==6.2.2 && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir mkl==2018.0.3 && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir tqdm && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir boto3 && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir requests && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir regex && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir psutil && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir nltk && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir sentencepiece && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir pytest && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir pytest-forked && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir pre-commit && \
    ldconfig

RUN /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir tensorboardX==1.8 && \
    /opt/miniconda/envs/deepspeed/bin/pip install --no-cache-dir tensorflow-gpu==1.15.2 && \
    ldconfig

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    tmux \
    unzip \
    htop \
    ninja-build

# Enable 'conda activate'
# RUN sudo ln -s /opt/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Install deepspeed and apex
RUN mkdir -p /tmp && \
    cd /tmp && \
    git clone --branch rizhao/zero-optimizer-fp32 https://github.com/rizhao-msft/DeepSpeed.git && \
    cd DeepSpeed && \
    export PIP='/opt/miniconda/envs/deepspeed/bin/pip' && \
    export PYTHON='/opt/miniconda/envs/deepspeed/bin/python' && \
    ./install.sh -l

# For running on NC24_Promo (Tesla K80 GPU), install NVIDIA driver 410.129
# Download and run NVIDIA-Linux-x86_64-410.129-diagnostic.run