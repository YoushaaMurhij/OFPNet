"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2022.03.26
# email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Training script for Occupancy and Flow Prediction
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import WaymoOccupancyFlowDataset
from model import UNetWithResnet50Encoder
from losses import GDiceLossV2
from configs import config
from tqdm import tqdm

import numpy as np
from time import sleep

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--resume', default='', help='resume from checkpoint', action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", default="seg_head.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='/logs/train_data/', help='path where to save output models and logs')
    args = parser.parse_args()
    return args

def main(args):

    now = datetime.now()
    tag = "test 1"
    save_str = '.' + args.save_dir + now.strftime("%d-%m-%Y-%H:%M:%S") + tag
    print("------------------------------------------")
    print("Use : tensorboard --logdir logs/train_data")
    print("------------------------------------------")

    writer = SummaryWriter(save_str)

    device = torch.device(args.device)
    print(f'cuda device is: {device}')

    dataset = WaymoOccupancyFlowDataset(grids_dir=config.GRIDS_DIR, waypoints_dir=config.WAYPOINTS_DIR, device=device) 
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.VAL_SPLIT * dataset_size))

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=valid_sampler)

    model = UNetWithResnet50Encoder().to(device)


    if args.resume:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint)
    writer.add_graph(model, torch.randn(1, 384, 128, 128, requires_grad=False).to(device))

    if args.focal_loss:
        criterion = FocalLoss_(gamma=2, alpha=CLASS_WEIGHTS)
    else:
        loss_weights = torch.FloatTensor(CLASS_WEIGHTS).to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

    optimizer = optim.SGD(model.parameters(), weight_decay = config.WEIGHT_DECAY, lr=config.LR, momentum=config.MOMENTUM)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(train_loader) * config.EPOCHS)) ** 0.8)
    model.train()
    for epoch in range(config.EPOCHS):
            
        with tqdm(train_loader, unit = "batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                features = data['grids']
                labels = data['waypoints']
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                tepoch.set_postfix(loss=loss.item())
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch * len(train_loader) + i) #optimizer.param_groups[0]['lr']
                sleep(0.01)

        # writer.add_scalar(f'accuracy', confmat.acc_global, epoch)
        # writer.add_scalar(f'mean_IoU', confmat.mean_IoU, epoch)

        PATH = save_str +'/Epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), PATH)
    print('Finished Training. Model Saved!')
    writer.close()

if __name__=="__main__":
    args = parse_args()
    main(args)

