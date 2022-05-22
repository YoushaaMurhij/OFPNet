docker exec -it ofp /bin/bash -c \
    "export PYTHONPATH=\"${PYTHONPATH}:/home/workspace/OFPNet\";
    cd /home/workspace/OFPNet;
    /bin/bash;" 