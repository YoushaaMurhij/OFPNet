import os
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from configs import config as CONFIG
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main():
    filenames = tf.io.gfile.glob(CONFIG.TRAIN_FILES)
    
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
    for i, filename in enumerate(filenames):
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        dataset = dataset.batch(1)
        print('processing frames from scene #' + str(i))
        for j, inputs in enumerate(dataset):
            ID = inputs['scenario/id'].numpy()[0].decode("utf-8")

            with open(CONFIG.DATASET_PKL_FOLDER + filename.split('-')[-3] + '_' +  ID + '.pkl','wb') as f: 
                pickle.dump(inputs, f)
    
    print("Done Converting to numpy pkls...")
        
if __name__ == '__main__':
    main()