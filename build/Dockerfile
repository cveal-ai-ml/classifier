FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt update && apt install -y tmux vim htop

RUN pip install dash dash-core-components pandas ipython plotly matplotlib scikit-image
RUN pip install torch torchvision segmentation-models-pytorch pytorch_warmup
RUN pip install --no-dependencies qudida albumentations

WORKDIR /develop/results
WORKDIR /develop/data
WORKDIR /develop/code
