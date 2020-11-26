#!/bin/bash

sudo apt-get install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake
cd ~ || exit
# clone mxnet for deepglint
git clone -b anxiang --recursive https://github.com/anxiangsir/mxnet.git mxnet_glint
cd mxnet_glint || exit
#
git reset 4384839a95403055af30b94b3b48c1255cd2c132
#
cp make/config.mk .
make -j48
cd python || exit
pip install -e .

