import  argparse
import torch
from core.datasets.dataset import WaymoOccupancyFlowDataset
from configs import hyperparameters
cfg = hyperparameters.get_config()


def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('-g', '--gpu' , default=0, type=int, help='gpu id')

    args = parser.parse_args()
    return args
def main(args):

    dataset = WaymoOccupancyFlowDataset(data_dir=cfg.DATASET_PKL_FOLDER, gpu=args.gpu) 
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=False,     
        num_workers=0,
        pin_memory=False)
    
    for j, data in enumerate(train_loader):
        print(data.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args)