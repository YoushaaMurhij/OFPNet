#!/bin/bash
echo "Compressing validation results for submission ..."
tar czvf /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/submit_validation.tar.gz  \
      -C /home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed/occupancy_flow_challenge/validation .
echo "Compression is done. Good luck!"
