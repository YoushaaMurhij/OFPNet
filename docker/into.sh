docker exec -it occupancy_flow /bin/bash -c \
    "export PYTHONPATH=\"${PYTHONPATH}:/home/docker_occupancy_flow/workspace/Occ_Flow_Pred\";
    cd /home/docker_occupancy_flow/workspace/Occ_Flow_Pred;
    /bin/bash;" 