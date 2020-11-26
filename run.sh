#!/bin/bash
#/home/face/miniconda3/bin/horovodrun \

/home/ubuntu/miniconda3/envs/mxnet/bin/horovodrun \
  -np 8 \
  -H localhost:8 \
  bash config.sh