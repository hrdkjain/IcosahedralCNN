from pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y --no-install-recommends python3.6

RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3
RUN rm -f /opt/conda/bin/python3 && ln -s /usr/bin/python3.6 /opt/conda/bin/python3

RUN apt-get update && apt-get install -y --no-install-recommends \
python3-pip \
python3-setuptools \
python3-dev \
python3-wheel \
emacs \
git \
graphviz \
nano \
wget

RUN python3 -m pip install \
matplotlib==2.0.2 \
natsort==7.0.1 \
numpy==1.16.4 \
pillow==7.1.1 \
plotly==4.6.0 \
plyfile==0.7.2 \
requests==2.23.0 \
scikit-image==0.16.2 \
scikit-learn==0.23.2 \
scipy==1.4.1 \
setuptools==46.1.3 \
tensorboard==2.0.0 \
tensorflow-gpu==1.14 \
the1owl==0.0.8 \
thebrainfuck==0.0.3 \
torch==1.4.0 \
torchvision==0.4.2 \
torchviz==0.0.1 \
tqdm==4.45.0 \
visdom==0.1.8.9 \
wheel==0.26.0

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends python3.6-tk
ENV MPLBACKEND "Agg"

# ugly hack for python cache
RUN mkdir /.cache && chmod 777 /.cache

ARG USER
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER
