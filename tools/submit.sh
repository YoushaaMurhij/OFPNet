#!/bin/bash

method=$1
description=$2

echo "Running inference script on Val/Test set"
python3 ./tools/submit.py --method $method --description $description

echo "Compressing validation results for submission ..."
tar czvf /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/${method}.tar.gz  \
      -C /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/validation/${method} .
echo "Deleting raw prediction protos ..."
rm -r /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/validation/${method}
echo "Compression is done. Good luck!"
