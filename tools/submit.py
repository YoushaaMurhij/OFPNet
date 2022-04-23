import argparse
from ast import arg
from tqdm import tqdm
import tensorflow as tf
from core.utils.submission import *
from core.models.unet_nest import R2AttU_Net
from core.models.models_mae import  mae_vit_large_patch16_dec512d8b
from core.models.efficientdet.backbone import EfficientDetBackbone

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
    parser.add_argument("--pretrained" , default="logs/train_data/20220422_205330_R2AttU_Net_New/Epoch_1_Iter_7609.pth", help="Use pre-trained models")

    args = parser.parse_args()
    return args

def main(args):

    # model = EfficientDetBackbone(compound_coef=1).to(DEVICE)
    # model = mae_vit_large_patch16_dec512d8b().to(DEVICE)
    model = R2AttU_Net(in_ch=cfg.INPUT_SIZE, out_ch=cfg.NUM_CLASSES, t=6).to(DEVICE)
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_shard_paths = sorted(tf.io.gfile.glob(cfg.VAL_FILES))

    with tf.io.gfile.GFile(cfg.VAL_SCENARIO_IDS_FILE) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]

    print("------------------------------------------")
    print('Got', len(test_scenario_ids), 'Val scenario ids.')
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

        save_submission_to_file(
            submission=submission, test_shard_path=test_shard_path, folder=args.method)

if __name__ == "__main__":
    args = parse_args()
    main(args)