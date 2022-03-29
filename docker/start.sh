#!/bin/bash

cd "$(dirname "$0")"
cd ..
workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=occupancy_flow)" ]; then
    docker rm occupancy_flow;
fi

docker run -it -d --rm \
    --gpus '"device=0"' \
    --net host \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    -e "DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    --shm-size="40g" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --name occupancy_flow \
    -v $workspace_dir/:/home/docker_occupancy_flow/workspace/Occ_Flow_Pred:rw \
    -v /media/hdd/benchmarks/Waymo_Motion/:/home/docker_occupancy_flow/workspace/Occ_Flow_Pred/data/Waymo_Motion:rw \
    x64/occupancy_flow:latest

docker exec -it occupancy_flow /bin/bash -c \
    "export PYTHONPATH=\"${PYTHONPATH}:/home/docker_occupancy_flow/workspace/Occ_Flow_Pred\";
    cd /home/docker_occupancy_flow/workspace/Occ_Flow_Pred;"


