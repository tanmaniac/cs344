#! /usr/bin/env bash

docker build --build-arg UID=$(id -u) \
             --build-arg GID=$(id -g) \
             --tag nvidia/cudagl:opencv .