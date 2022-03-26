import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
import collections
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

DATASET_FOLDER = '/media/hdd/benchmarks/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed'
DATASET_PKL_FOLDER = '/media/hdd/benchmarks/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/pkls/sample'
SAMPLE_FILES = f'{DATASET_FOLDER}/tf_example/sample/training_tfexample.tfrecord*'
# Text files containing validation and test scenario IDs for this challenge.
VAL_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
TEST_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

NUM_PRED_CHANNELS = 4

def _make_model_inputs(
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

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    filenames = tf.io.matching_files(SAMPLE_FILES)
    dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    dataset = dataset.repeat()
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(1)
    it = iter(dataset)

    config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
    config_text = """
    num_past_steps: 10
    num_future_steps: 80
    num_waypoints: 8
    cumulative_waypoints: true
    normalize_sdc_yaw: true
    grid_height_cells: 256
    grid_width_cells: 256
    sdc_y_in_grid: 192
    sdc_x_in_grid: 128
    pixels_per_meter: 3.2
    agent_points_per_side_length: 48
    agent_points_per_side_width: 16
    """
    text_format.Parse(config_text, config)
    print("Used configs:")
    print(config)
    print("Started Converting to numpy pkls...")
    for i in range(len(filenames)):
        inputs = next(it)
        print('processing scene #' + str(i) + ' with id: ' + inputs['scenario/id'].numpy()[0].decode("utf-8"))

        inputs = occupancy_flow_data.add_sdc_fields(inputs)

        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
            inputs=inputs, config=config)

        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
            timestep_grids=timestep_grids, config=config)

        vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
            inputs=inputs, timestep_grids=timestep_grids, config=config)

        model_inputs = _make_model_inputs(timestep_grids, vis_grids)

        with open(DATASET_PKL_FOLDER + '/grids/' + inputs['scenario/id'].numpy()[0].decode("utf-8") + '.pkl','wb') as f: 
            pickle.dump(model_inputs.numpy, f)

        true_waypoints_dict = collections.defaultdict(dict)
        true_waypoints_dict['vehicles']['observed_occupancy']= true_waypoints.vehicles.observed_occupancy
        true_waypoints_dict['vehicles']['occluded_occupancy']= true_waypoints.vehicles.occluded_occupancy
        true_waypoints_dict['vehicles']['flow']= true_waypoints.vehicles.flow
        true_waypoints_dict['vehicles']['flow_origin_occupancy']= true_waypoints.vehicles.flow_origin_occupancy

        true_waypoints_dict['pedestrians']['observed_occupancy']= true_waypoints.pedestrians.observed_occupancy
        true_waypoints_dict['pedestrians']['occluded_occupancy']= true_waypoints.pedestrians.occluded_occupancy
        true_waypoints_dict['pedestrians']['flow']= true_waypoints.pedestrians.flow
        true_waypoints_dict['pedestrians']['flow_origin_occupancy']= true_waypoints.pedestrians.flow_origin_occupancy

        true_waypoints_dict['cyclists']['observed_occupancy']= true_waypoints.cyclists.observed_occupancy
        true_waypoints_dict['cyclists']['occluded_occupancy']= true_waypoints.cyclists.occluded_occupancy
        true_waypoints_dict['cyclists']['flow']= true_waypoints.cyclists.flow
        true_waypoints_dict['cyclists']['flow_origin_occupancy']= true_waypoints.cyclists.flow_origin_occupancy

        with open(DATASET_PKL_FOLDER + '/waypoints/' + inputs['scenario/id'].numpy()[0].decode("utf-8") + '.pkl','wb') as f: 
            pickle.dump(true_waypoints_dict, f)
    
    print("Done Converting to numpy pkls...")
        
if __name__ == '__main__':
    main()