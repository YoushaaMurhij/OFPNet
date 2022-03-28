import pathlib
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids

from configs import config
import torch
from collections import defaultdict

def run_model_on_inputs(model_inputs, model):

    """Preprocesses inputs and runs model on one batch."""

    model_outputs = model(model_inputs)
    pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)

    return pred_waypoint_logits

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

def make_submission_proto() -> occupancy_flow_submission_pb2.ChallengeSubmission:

    """Makes a submission proto to store predictions for one shard."""

    submission = occupancy_flow_submission_pb2.ChallengeSubmission()
    submission.account_name = 'yosha.morheg@phystech.edu'
    submission.unique_method_name = 'Occupancy to the moon'
    submission.authors.extend(['Youshaa Murhij', 'Dmitry Yudin'])
    submission.description = 'Encoder-Decoder based prediction'
    submission.method_link = 'https://github.com/YoushaaMurhij/Occ_Flow_Pred'
    return submission

def add_waypoints_to_scenario_prediction(pred_waypoints, scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,) -> None:

    """Add predictions for all waypoints to scenario_prediction message."""

    for k in range(config.NUM_WAYPOINTS):
        waypoint_message = scenario_prediction.waypoints.add()
        # Observed occupancy.
        obs_occupancy = pred_waypoints['vehicles']['observed_occupancy'][k].numpy()
        obs_occupancy_quantized = np.round(obs_occupancy * 255).astype(np.uint8)
        obs_occupancy_bytes = zlib.compress(obs_occupancy_quantized.tobytes())
        waypoint_message.observed_vehicles_occupancy = obs_occupancy_bytes
        # Occluded occupancy.
        occ_occupancy = pred_waypoints['vehicles']['occluded_occupancy'][k].numpy()
        occ_occupancy_quantized = np.round(occ_occupancy * 255).astype(np.uint8)
        occ_occupancy_bytes = zlib.compress(occ_occupancy_quantized.tobytes())
        waypoint_message.occluded_vehicles_occupancy = occ_occupancy_bytes
        # Flow.
        flow = pred_waypoints['vehicles']['flow'][k].numpy()
        flow_quantized = np.clip(np.round(flow), -128, 127).astype(np.int8)
        flow_bytes = zlib.compress(flow_quantized.tobytes())
        waypoint_message.all_vehicles_flow = flow_bytes

def apply_sigmoid_to_occupancy_logits(pred_waypoint_logits):

    """Converts occupancy logits with probabilities."""

    pred_waypoints =  defaultdict(dict)
    pred_waypoints['vehicles']['observed_occupancy'] = [torch.nn.Sigmoid(x) for x in pred_waypoint_logits['vehicles']['observed_occupancy']]
    pred_waypoints['vehicles']['occluded_occupancy'] = [torch.nn.Sigmoid(x) for x in pred_waypoint_logits['vehicles']['occluded_occupancy']]
    pred_waypoints['vehicles']['flow'] = pred_waypoint_logits['vehicles']['flow']
    return pred_waypoints

def generate_predictions_for_one_test_shard(inputs, model,
    submission: occupancy_flow_submission_pb2.ChallengeSubmission,
    test_scenario_ids: Sequence[str]) -> None:

    """Iterate over all test examples in one shard and generate predictions."""

    if inputs['scenario/id'] in test_scenario_ids:
        # Run inference.
        pred_waypoint_logits = run_model_on_inputs(inputs=inputs, model=model)
        pred_waypoints = apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

        # Make new scenario prediction message.
        scenario_prediction = submission.scenario_predictions.add()
        scenario_prediction.scenario_id = inputs['scenario/id'].numpy()[0]

        # Add all waypoints.
        add_waypoints_to_scenario_prediction(
            pred_waypoints=pred_waypoints,
            scenario_prediction=scenario_prediction,
            config=config)

def save_submission_to_file(
    submission: occupancy_flow_submission_pb2.ChallengeSubmission,
    tfrecord_id: str) -> None:

    """Save predictions for one test shard as a binary protobuf."""

    save_folder = os.path.join(pathlib.Path.home(),
                                'occupancy_flow_challenge/testing')
    os.makedirs(save_folder, exist_ok=True)
    submission_basename = 'occupancy_flow_submission.binproto-' + tfrecord_id + '-of-00150'

    submission_shard_file_path = os.path.join(save_folder, submission_basename)
    num_scenario_predictions = len(submission.scenario_predictions)
    print(f'Saving {num_scenario_predictions} scenario predictions to '
            f'{submission_shard_file_path}...\n')
    f = open(submission_shard_file_path, 'wb')
    f.write(submission.SerializeToString())
    f.close()

def make_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
    ) -> tf.Tensor:

    """Concatenates all occupancy grids over past, current to a single tensor."""

    model_inputs = tf.concat(
        [
            vis_grids.roadgraph,
            timestep_grids.vehicles.past_occupancy,
            timestep_grids.vehicles.current_occupancy,
            tf.clip_by_value(
                timestep_grids.pedestrians.past_occupancy +
                timestep_grids.cyclists.past_occupancy, 0, 1),
            tf.clip_by_value(
                timestep_grids.pedestrians.current_occupancy +
                timestep_grids.cyclists.current_occupancy, 0, 1),
        ],
        axis=-1,
    )
    return model_inputs