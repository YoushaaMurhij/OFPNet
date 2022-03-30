"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2022.03.26
# email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Cutsom Dataset loading script for Occupancy and Flow Prediction
"""
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from core.utils.io import make_model_inputs
from configs import config

import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2

class WaymoOccupancyFlowDataset(Dataset):
    def __init__(self, FILES) -> None:
        super().__init__()
        
        filenames = tf.io.matching_files(FILES)
        dataset = tf.data.TFRecordDataset(filenames, compression_type='')
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        dataset = dataset.batch(1)
    
        self.it = iter(dataset)
        self.config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        text_format.Parse(open('./configs/config.txt').read(), self.config)

        print("------------------------------------------")
        print("Occupency and Flow Prediction Parameters")
        print("------------------------------------------")
        print(self.config)
    
    def __len__(self):
        return 44920  # TODO WTF I need to calc it  44920

    def __getitem__(self, idx):

        inputs = next(self.it)

        ID = inputs['scenario/id'].numpy()[0].decode("utf-8")
        inputs = occupancy_flow_data.add_sdc_fields(inputs)

        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(inputs=inputs, config=self.config)
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=self.config)
        vis_grids      = occupancy_flow_grids.create_ground_truth_vis_grids(inputs=inputs, timestep_grids=timestep_grids, config=self.config)

        model_inputs = make_model_inputs(timestep_grids, vis_grids).numpy()

        grid = torch.tensor(model_inputs[0])
        grid = torch.permute(grid, (2, 0, 1))

        waypoint = defaultdict(dict)
        waypoint['vehicles']['observed_occupancy']    = [wp.numpy() for wp in true_waypoints.vehicles.observed_occupancy]
        waypoint['vehicles']['occluded_occupancy']    = [wp.numpy() for wp in true_waypoints.vehicles.occluded_occupancy]
        waypoint['vehicles']['flow']                  = [wp.numpy() for wp in true_waypoints.vehicles.flow]
        waypoint['vehicles']['flow_origin_occupancy'] = [wp.numpy() for wp in true_waypoints.vehicles.flow_origin_occupancy]

        

        sample = {'grids': grid, 'waypoints': waypoint, 'index': idx, 'scenario/id': ID}

        return sample

