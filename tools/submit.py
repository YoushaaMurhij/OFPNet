import argparse
from tqdm import tqdm
import tensorflow as tf
from core.utils.submission import *
from core.models.unet_nest import R2AttU_Net
from core.models.unet_head import R2AttU_sepHead
from core.models.unet_seq import R2AttU_seq
from core.models.xception import Xception
from core.models.wnet import WNet

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
    parser.add_argument("--pretrained" , default="pretrained/20220429_221529_R2AttU_T2/Epoch_4.pth", help="Use pre-trained models")
    parser.add_argument('--split', help='either validation or testing split', default='val')
    args = parser.parse_args()
    return args

def main(args):

    # model = R2AttU_sepHead(img_ch=cfg.INPUT_SIZE, output_ch=cfg.NUM_CLASSES, t=2, sliced_head=True).to(DEVICE)
    model = R2AttU_Net(in_ch=cfg.INPUT_SIZE, out_ch=cfg.NUM_CLASSES, t=2).to(DEVICE)
    # model = WNet(img_ch=cfg.INPUT_SIZE, output_ch=cfg.NUM_CLASSES, t=1).to(DEVICE)
    # model = R2AttU_seq(img_ch=23, output_ch=32, t=1).to(DEVICE)
    # model = Xception('xception71', in_channels=23, time_limit=8, n_traj=64, with_head=True).to(DEVICE)

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