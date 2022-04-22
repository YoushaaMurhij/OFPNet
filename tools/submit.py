import argparse
import tensorflow as tf
from core.utils.submission import *
from configs import hyperparameters
cfg = hyperparameters.get_config()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction submission')
    parser.add_argument('--method', help='Unique method name', required=True)
    parser.add_argument("--description", help="Brief description of the method", required=True)

    args = parser.parse_args()
    return args

def main(args):

    test_shard_paths = sorted(tf.io.gfile.glob(cfg.VAL_FILES))

    with tf.io.gfile.GFile(cfg.VAL_SCENARIO_IDS_FILE) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]

    print("------------------------------------------")
    print('Got', len(test_scenario_ids), 'Val scenario ids.')
    print("------------------------------------------")

    for i, test_shard_path in enumerate(test_shard_paths):

        print(f'Creating submission for test shard {test_shard_path}...')

        test_dataset = make_test_dataset(test_shard_path=test_shard_path)
        submission = make_submission_proto(args.method, args.description)
        generate_predictions_for_one_test_shard(
            submission=submission,
            test_dataset=test_dataset,
            test_scenario_ids=test_scenario_ids,
            shard_message=f'{i + 1} of {len(test_shard_paths)}')

        save_submission_to_file(
            submission=submission, test_shard_path=test_shard_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)