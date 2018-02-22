#! /usr/bin/env bash


docker run -it \
--privileged \
--name cs344 \
-v ${HOME}/Documents/learning_cuda/cs344:/home/nvidia/cs344 \
nvidia/cuda:opencv \
/bin/bash