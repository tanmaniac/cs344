#! /usr/bin/env bash

NAME="cs344"

if [ ! "$(docker ps -q -f name=${NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${NAME})" ]; then
        # cleanup
        docker rm ${NAME}
    fi
    docker run -it \
           --runtime=nvidia \
           --privileged \
           --name ${NAME} \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v ${HOME}/Documents/learning_cuda/cs344:/home/nvidia/cs344 \
           nvidia/cudagl:opencv \
           /bin/bash -c "konsole --profile /home/nvidia/.local/share/konsole/nvidia.profile; /bin/bash"
fi
