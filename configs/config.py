# Data location. Please edit.
# DATASET_FOLDER = '/media/hdd/benchmarks/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed'
DATASET_FOLDER = '/home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed'

# TFRecord dataset.
TRAIN_FILES  = f'{DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
VAL_FILES    = f'{DATASET_FOLDER}/tf_example/validation/validation_tfexample.tfrecord*'
TEST_FILES   = f'{DATASET_FOLDER}/tf_example/testing/testing_tfexample.tfrecord*'
SAMPLE_FILES = f'{DATASET_FOLDER}/tf_example/sample/training_tfexample.tfrecord*'

DATASET_PKL_FOLDER =  f'{DATASET_FOLDER}/pkl_example/training/'

# Text files containing validation and test scenario IDs for this challenge.
VAL_SCENARIO_IDS_FILE  = f'{DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
TEST_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

NUM_PRED_CHANNELS = 4

# num_past_steps: 10
# num_future_steps: 80
NUM_WAYPOINTS = 8
# cumulative_waypoints: true
# normalize_sdc_yaw: true
# grid_height_cells: 256
# grid_width_cells: 256
# sdc_y_in_grid: 192
# sdc_x_in_grid: 128
# pixels_per_meter: 3.2
# agent_points_per_side_length: 48
# agent_points_per_side_width: 16

# Train configs
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE   = 1

INPUT_SIZE       = 23
NUM_CLASSES      = 32

WEIGHT_DECAY     = 0.007
EPOCHS           = 4
LR               = 0.001
MOMENTUM         = 0.8
