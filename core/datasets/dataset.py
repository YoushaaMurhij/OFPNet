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

class WaymoOccupancyFlowDataset(Dataset):
    def __init__(self, grids_dir, waypoints_dir, device) -> None:
        super().__init__()
        
        self.device = device
        self.grids_dir = grids_dir
        self.waypoints_dir = waypoints_dir
        self.grid_files = [f for f in listdir(self.grids_dir) if isfile(join(self.grids_dir, f))]
    
    def __len__(self):
        return len(self.grid_files)

    def __getitem__(self, idx):

        grid_path = os.path.join(self.grids_dir, self.grid_files[idx])
        grid_file = open(grid_path, 'rb')
        grid = pkl.load(grid_file)[0]
        
        grid = torch.tensor(grid).to(self.device)
        grid = torch.permute(grid, (2, 0, 1))

        waypoint_path = os.path.join(self.waypoints_dir, self.grid_files[idx])  # both have the same name [senario id] with diffrendt directory
        waypoint_file = open(waypoint_path, 'rb')
        waypoint = pkl.load(waypoint_file)


        sample = {'grids': grid, 'waypoints': waypoint, 'index': idx}

        return sample

