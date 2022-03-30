#!/bin/bash

cd "$(dirname "$0")"
cd ..
workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=ofp)" ]; then
    docker rm ofp;
fi

docker run -it -d --rm \
    --gpus '"device=0,1"' \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="40g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name ofp \
    -v $workspace_dir/:/home/workspace/Occ_Flow_Pred:rw \
    -v /datasets/Waymo_Motion/:/home/workspace/Occ_Flow_Pred/data/Waymo_Motion:rw \
    x64/ofp:latest

docker exec -it ofp /bin/bash -c \
    "export PYTHONPATH=\"${PYTHONPATH}:/home/workspace/Occ_Flow_Pred\";
    cd /home/workspace/Occ_Flow_Pred;"


