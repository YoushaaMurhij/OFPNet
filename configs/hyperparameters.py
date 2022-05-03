# coding=utf-8
"""Hyperparameters of the structured occupancy prediction models."""

class ConfigDict(dict):
  """A dictionary whose keys can be accessed as attributes."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

  def get(self, key, default=None):
    """Allows to specify defaults when accessing the config."""
    if key not in self:
      return default
    return self[key]

def get_config():

    """Default values for all hyperparameters."""
    cfg = ConfigDict()

    # Directories:
    cfg.DATASET_FOLDER = '/home/workspace/Occ_Flow_Pred/data/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed'
    # TFRecord dataset.
    cfg.TRAIN_FILES    = f'{cfg.DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
    cfg.VAL_FILES      = f'{cfg.DATASET_FOLDER}/tf_example/validation/validation_tfexample.tfrecord*'
    cfg.TEST_FILES     = f'{cfg.DATASET_FOLDER}/tf_example/testing/testing_tfexample.tfrecord*'
    cfg.SAMPLE_FILES   = f'{cfg.DATASET_FOLDER}/tf_example/sample/training_tfexample.tfrecord*'

    cfg.DATASET_PKL_FOLDER = f'{cfg.DATASET_FOLDER}/pkl_example/training/'
    cfg.VALSET_PKL_FOLDER  = f'{cfg.DATASET_FOLDER}/pkl_example/validation/'
    # Text files containing validation and test scenario IDs for this challenge.
    cfg.VAL_SCENARIO_IDS_FILE  = f'{cfg.DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
    cfg.TEST_SCENARIO_IDS_FILE = f'{cfg.DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

    # Architecture:
    cfg.INPUT_SIZE                   = 23
    cfg.NUM_CLASSES                  = 32
            
    # Optimization:            
    cfg.OPTIMIZER                    = 'adamw'
    cfg.SCHEDULER                    = 'CosineAnnealingLR' # 'CosineAnnealingLR' , GetInitLR, ReduceLROnPlateau
    cfg.TRAIN_BATCH_SIZE             = 12
    cfg.VAL_BATCH_SIZE               = 1
    cfg.WEIGHT_DECAY                 = 0.007
    cfg.EPOCHS                       = 3
    cfg.LR                           = 0.0001
    cfg.MOMENTUM                     = 0.8

    # Grid sequence parameters:
    cfg.NUM_PRED_CHANNELS            = 4
    cfg.num_past_steps               = 10
    cfg.num_future_steps             = 80
    cfg.NUM_WAYPOINTS                = 8
    cfg.cumulative_waypoints         = False
    cfg.normalize_sdc_yaw            = True
    cfg.grid_height_cells            = 256
    cfg.grid_width_cells             = 256
    cfg.sdc_y_in_grid                = 192
    cfg.sdc_x_in_grid                = 128
    cfg.pixels_per_meter             = 3.2
    cfg.agent_points_per_side_length = 48
    cfg.agent_points_per_side_width  = 16

    # Train configs
    cfg.WANDB_MODE                   = "online"  # {'run', 'online', 'offline', 'dryrun', 'disabled'}

    return cfg







