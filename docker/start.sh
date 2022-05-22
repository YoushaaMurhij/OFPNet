#!/bin/bash

cd "$(dirname "$0")"
cd ..

workspace_dir=$PWD
dataset_dir="/datasets/Waymo_Motion/"


if [ "$(docker ps -aq -f status=exited -f name=ofp)" ]; then
    docker rm ofp;
fi

docker run -it -d --rm \
    --gpus '"device=0, 1"' \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="40g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name ofp \
    -v $workspace_dir/:/home/workspace/OFPNet:rw \
    -v ${dataset_dir}:/home/workspace/OFPNet/data/Waymo_Motion:rw \
    x64/ofp:latest

docker exec -it ofp /bin/bash -c \
    "export PYTHONPATH=\"${PYTHONPATH}:/home/workspace/OFPNet\";
    cd /home/workspace/OFPNet;
    nvidia-smi;"


