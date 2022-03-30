import tensorflow as tf
from core.utils.submission import *
from configs import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

def main():

    test_shard_paths = tf.io.gfile.glob(config.VAL_FILES)

    with tf.io.gfile.GFile(config.VAL_SCENARIO_IDS_FILE) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]

    print("------------------------------------------")
    print('Got', len(test_scenario_ids), 'Val scenario ids.')
    print("------------------------------------------")

    for i, test_shard_path in enumerate(test_shard_paths):

        print(f'Creating submission for test shard {test_shard_path}...')

        test_dataset = make_test_dataset(test_shard_path=test_shard_path)
        submission = make_submission_proto()
        generate_predictions_for_one_test_shard(
            submission=submission,
            test_dataset=test_dataset,
            test_scenario_ids=test_scenario_ids,
            shard_message=f'{i + 1} of {len(test_shard_paths)}')

        save_submission_to_file(
            submission=submission, test_shard_path=test_shard_path)

        # if i == 0:
        #     print('Sample scenario prediction:\n')
        #     print(submission.scenario_predictions[-1])

if __name__ == "__main__":
    main()