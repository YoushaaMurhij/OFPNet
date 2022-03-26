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
# from torch._C import device
from torch.utils.data import Dataset
import numpy as np
from os import listdir
from os.path import isfile, join

class WaymoOccupancyFlowDataset(Dataset):
    def __init__(self, grids_dir, waypoints_dir, device) -> None:
        super().__init__()
        
        self.device = device
        self.grids_dir = grids_dir
        self.waypoints_dir = waypoints_dir
        self.grid_files = [f for f in listdir(self.grids_dir) if isfile(join(self.grids_dir, f))]
        self.waypoint_files = [f for f in listdir(self.waypoints_dir) if isfile(join(self.waypoints_dir, f))]
    
    def __len__(self):
        return len(self.grid_files)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        grid_name = os.path.join(self.grids_dir, self.grid_files[idx] + '.bin.pt')
        grid = torch.load(grid_name, map_location=self.device)

        waypoint_name = os.path.join(self.waypoints_dir, self.waypoint_files[idx] + '.txt')
        waypoint = np.loadtxt(waypoint_name, dtype=int, delimiter=',')

        sample = {'grid': grid, 'waypoint': waypoint, 'index': idx}

        return sample

