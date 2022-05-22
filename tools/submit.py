import argparse
from tqdm import tqdm
import tensorflow as tf
from core.utils.submission import *
from core.models.unet_lstm import UNet_LSTM
from core.models.mfnet_3d import MFNET_3D

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
    parser.add_argument("--pretrained" , default="pretrained/20220521_145830_MFNET_3D/Epoch_1_Iter_5073.pth", help="Use pre-trained models")
    parser.add_argument('--split', help='either validation or testing split', default='val')
    args = parser.parse_args()
    return args

def main(args):
    # model = UNet_LSTM(n_channels=23, n_classes=32, with_head=False, sequence=False).to(DEVICE)
    # model = UNet_LSTM(n_channels=23, n_classes=32, with_head=True, sequence=False).to(DEVICE)
    model = MFNET_3D(pretrained=False,batch_size=cfg.VAL_BATCH_SIZE).to(DEVICE)

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.split == 'validation':
        FILES = cfg.VAL_FILES
        SCENARIO_IDS = cfg.VAL_SCENARIO_IDS_FILE
    elif args.split == 'testing':
        FILES = cfg.TEST_FILES
        SCENARIO_IDS = cfg.TEST_SCENARIO_IDS_FILE
    else:
        print('No split named {}. Exiting submission process'.format(args.split))
        exit(1)

    test_shard_paths = sorted(tf.io.gfile.glob(FILES))
    with tf.io.gfile.GFile(SCENARIO_IDS) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]

    print("------------------------------------------")
    print('Got', len(test_scenario_ids), args.split, 'scenario ids.')
    print("------------------------------------------")

    for i, test_shard_path in enumerate(tqdm(test_shard_paths)):
        # print(f'Creating submission for test shard {test_shard_path}...')
        test_dataset = make_test_dataset(test_shard_path=test_shard_path)
        submission = make_submission_proto(args.method, args.description)
        generate_predictions_for_one_test_shard(
            submission=submission,
            test_dataset=test_dataset,
            test_scenario_ids=test_scenario_ids,
            shard_message=f'{i + 1} of {len(test_shard_paths)}',
            model=model)
        save_submission_to_file(submission=submission, test_shard_path=test_shard_path, folder=args.method, split=args.split)

if __name__ == "__main__":
    args = parse_args()
    main(args)