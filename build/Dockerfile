FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop

RUN pip install torch torchvision segmentation-models-pytorch albumentations dash dash-core-components pandas ipython plotly matplotlib pytorch_warmup

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code
