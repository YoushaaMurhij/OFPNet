#!/bin/bash

method=$1
description=$2
split=$3

echo "Running inference script on ${split} set"
python3 ./tools/submit.py --method $method --description $description --split $split

echo "Compressing ${split} results for submission ..."
tar czvf /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/${method}.tar.gz  \
      -C /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/${split}/${method} .
echo "Deleting raw prediction protos ..."
rm -r /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/${split}/${method}
echo "Compression is done. Good luck!"
