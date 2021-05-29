#!/bin/bash

set -euxo pipefail

apt-get update
apt install --no-install-recommends \
  terminator \
  tmux \
  vim \
  gedit \
  git \
  openssh-client \
  unzip \
  htop \
  libopenni-dev \
  apt-utils \
  usbutils \
  dialog \
  python-pip \
  python-dev \
  ffmpeg \
  nvidia-settings \
  cmake-curses-gui \
  libyaml-dev

pip install --upgrade pip==9.0.3
pip install -U setuptools

apt-get -y install ipython
apt-get -y install ipython-notebook
pip install \
  qtconsole==4.7.7\
  pyrsistent==0.16.1\
  imageio==2.6.0\
  jupyter \
  opencv-python==4.2.0.32 \
  plyfile \
  pandas \
  tensorflow \
  future \
  typing
