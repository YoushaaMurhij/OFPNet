import os
import pickle
import argparse
from tqdm import tqdm
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from configs import hyperparameters
cfg = hyperparameters.get_config()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Dataset Preparing')
    parser.add_argument('--split', help='Choose train or val')
    args = parser.parse_args()
    return args

def main(args):

    split = args.split
    assert split == 'train' or split == 'val' , "Invalid Split Name"
    if args.split == 'train':
        filenames = tf.io.gfile.glob(cfg.TRAIN_FILES)
        SAVE_PATH = cfg.DATASET_PKL_FOLDER
    else:
        filenames = tf.io.gfile.glob(cfg.VAL_FILES)
        SAVE_PATH = cfg.VALSET_PKL_FOLDER

    os.makedirs(SAVE_PATH, exist_ok=True)
    config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
    config_text = """
    num_past_steps: 10
    num_future_steps: 80
    num_waypoints: 8
    cumulative_waypoints: false
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
    for filename in tqdm(filenames):
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        dataset = dataset.batch(1)
        # print('processing frames from scene #' + str(i))
        for inputs in dataset:
            ID = inputs['scenario/id'].numpy()[0].decode("utf-8")

            with open(SAVE_PATH + filename.split('-')[-3] + '_' +  ID + '.pkl','wb') as f: 
                pickle.dump(inputs, f)
    
    print("Done Converting to numpy pkls...")
        
if __name__ == '__main__':
    args = parse_args()
    main(args)