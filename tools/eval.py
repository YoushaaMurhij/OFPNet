import argparse
import numpy as np
from tqdm import tqdm
from time import sleep
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from core.datasets.dataset import WaymoOccupancyFlowDataset
from core.models.unet import UNet
from configs import config

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib

import tensorflow as tf
import tensorflow_graphics.image.transformer as tfg_transformer

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_metrics
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis

from core.utils.submission import *

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument("--ckpt", default="seg_head.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./logs/train_data/', help='path where to save output models and logs')
    args = parser.parse_args()
    return args

def get_pred_waypoint_logits(model_outputs):
    """Slices model predictions into occupancy and flow grids."""

    pred_waypoint_logits = defaultdict(dict)
    model_outputs = torch.permute(model_outputs, (0, 2, 3, 1))  

    pred_waypoint_logits['vehicles']['observed_occupancy'] = []
    pred_waypoint_logits['vehicles']['occluded_occupancy'] = []
    pred_waypoint_logits['vehicles']['flow'] = []

    # Slice channels into output predictions.
    for k in range(config.NUM_WAYPOINTS):
        index = k * config.NUM_PRED_CHANNELS
        waypoint_channels = model_outputs[:, :, :, index:index + config.NUM_PRED_CHANNELS]
        pred_observed_occupancy = waypoint_channels[:, :, :, :1]
        pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
        pred_flow = waypoint_channels[:, :, :, 2:]
        pred_waypoint_logits['vehicles']['observed_occupancy'].append(pred_observed_occupancy)
        pred_waypoint_logits['vehicles']['occluded_occupancy'].append(pred_occluded_occupancy)
        pred_waypoint_logits['vehicles']['flow'].append(pred_flow)

    return pred_waypoint_logits



def main(args):

    now = datetime.now()
    tag = "val debug"
    save_str = '.' + args.save_dir + now.strftime("%d-%m-%Y-%H:%M:%S") + tag
    device = torch.device(args.device)
    print(f'cuda device is: {device}')

    dataset = WaymoOccupancyFlowDataset(grids_dir=config.GRIDS_DIR, waypoints_dir=config.WAYPOINTS_DIR, device=device) 
    valid_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    model = UNet(config.INPUT_SIZE, config.NUM_CLASSES).to(device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint)

    model.eval()

    with tqdm(valid_loader, unit = "batch") as tepoch:
        for i, data in enumerate(tepoch):
            print(f'Creating submission for test shard {test_shard_path}...')
            submission = make_submission_proto()
            _generate_predictions_for_one_test_shard(
                submission=submission,
                test_dataset=test_dataset,
                test_scenario_ids=test_scenario_ids,
                shard_message=f'{i + 1} of {len(test_shard_paths)}')
            _save_submission_to_file(
                submission=submission, test_shard_path=test_shard_path)

            if i == 0:
                print('Sample scenario prediction:\n')
                print(submission.scenario_predictions[-1])

            
        with tqdm(valid_loader, unit = "batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                grids = data['grids']
                true_waypoints = data['waypoints']
                true_waypoints = true_waypoints

                model_outputs = model(grids)
                pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)
               
                sleep(0.01)

    print('Finished validation. Model Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

