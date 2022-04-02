"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2022.03.26
# email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Cutsom Dataset loading script for Occupancy and Flow Prediction
"""
import os
import torch
import numpy as np 
import pickle as pkl
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset
from collections import defaultdict
from core.utils.io import make_model_inputs

import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class WaymoOccupancyFlowDataset(Dataset):
    def __init__(self, data_dir, gpu) -> None:
        super().__init__()
        
        self.gpu = gpu
        self.data_dir = data_dir
        self.scenes = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
        self.config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        text_format.Parse(open('./configs/config.txt').read(), self.config)

    def __len__(self):
        return len(self.scenes)   #487002 # for training || 44920 for testing  

    def __getitem__(self, idx):

        scene = self.scenes[idx]
        inputs_pkl = open(self.data_dir + scene, 'rb')
        inputs = pkl.load(inputs_pkl)

        # ID = inputs['scenario/id'].numpy()[0].decode("utf-8")
        inputs = occupancy_flow_data.add_sdc_fields(inputs)

        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(inputs=inputs, config=self.config)
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=self.config)
        vis_grids      = occupancy_flow_grids.create_ground_truth_vis_grids(inputs=inputs, timestep_grids=timestep_grids, config=self.config)

        model_inputs = make_model_inputs(timestep_grids, vis_grids).numpy()

        grid = torch.tensor(model_inputs[0])
        grid = torch.permute(grid, (2, 0, 1)).cuda(self.gpu)

        waypoint = defaultdict(dict)
        waypoint['vehicles']['observed_occupancy']    = [torch.tensor(wp[0].numpy()).cuda(self.gpu) for wp in true_waypoints.vehicles.observed_occupancy]    # (1, 256, 256, 1) * 8
        waypoint['vehicles']['occluded_occupancy']    = [torch.tensor(wp[0].numpy()).cuda(self.gpu) for wp in true_waypoints.vehicles.occluded_occupancy]    # (1, 256, 256, 1) * 8
        waypoint['vehicles']['flow']                  = [torch.tensor(wp[0].numpy()).cuda(self.gpu) for wp in true_waypoints.vehicles.flow]                  # (1, 256, 256, 1) * 8
        waypoint['vehicles']['flow_origin_occupancy'] = [torch.tensor(wp[0].numpy()).cuda(self.gpu) for wp in true_waypoints.vehicles.flow_origin_occupancy] # (1, 256, 256, 1) * 8

        sample = {'grids': grid, 'waypoints': waypoint} # 'index': idx 'scenario/id': ID

        return sample

