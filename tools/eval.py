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

def main(args):

    now = datetime.now()
    tag = "val debug"
    save_str = '.' + args.save_dir + now.strftime("%d-%m-%Y-%H:%M:%S") + tag
    device = torch.device(args.device)
    print(f'cuda device is: {device}')

    dataset = WaymoOccupancyFlowDataset(FILES=config.SAMPLE_FILES, device=device)
    valid_loader = DataLoader(dataset, batch_size=config.VAL_BATCH_SIZE)

    model = UNet(config.INPUT_SIZE, config.NUM_CLASSES).to(device)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint)

    model.eval()

    with tqdm(valid_loader, unit = "batch") as tepoch:
        for i, data in enumerate(tepoch):

            submission = make_submission_proto()
            generate_predictions_for_one_test_shard(inputs=data['grids'], model=model,
                submission=submission,
                test_scenario_ids=test_scenario_ids)
            save_submission_to_file(
                submission=submission, tfrecord_id=data['tfrecord_id'])

            if i == 0:
                print('Sample scenario prediction:\n')
                print(submission.scenario_predictions[-1])


    print('Finished validation. Model Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

