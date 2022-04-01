"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2022.03.26
# email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Training script for Occupancy and Flow Prediction
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from time import sleep
from datetime import datetime
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.datasets.dataset import WaymoOccupancyFlowDataset
from core.models.unet import UNet
from core.losses.occupancy_flow_loss import Occupancy_Flow_Loss
from core.utils.io import get_pred_waypoint_logits
from configs import config
import wandb

os.environ["WANDB_API_KEY"] = 'cccdc2dfb027090440d22b2ea4b94d57b9724115'
os.environ["WANDB_MODE"] = "online"
wandb.init(project="occupancy-flow", entity="youshaamurhij")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--resume', default='', help='resume from checkpoint', action="store_true")
    parser.add_argument("--pretrained", default="./logs/Epoch_4.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./logs/train_data/', help='path where to save output models and logs')
    args = parser.parse_args()
    return args

def main(args):
    wandb.config.update(args)
    now = datetime.now()
    tag = "train_unet_15ep"
    save_str = args.save_dir + now.strftime("%d-%m-%Y-%H-%M-%S-") + tag

    device = torch.device(args.device)
    print(f'cuda device is: {device}')

    model = UNet(config.INPUT_SIZE, config.NUM_CLASSES).to(device)
    wandb.watch(model)

    if args.resume:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f'Weights are loaded from: {args.pretrained}.')

    optimizer = optim.SGD(model.parameters(), weight_decay = config.WEIGHT_DECAY, lr=config.LR, momentum=config.MOMENTUM)
    
    model.train()
    for epoch in range(config.EPOCHS):
        dataset = WaymoOccupancyFlowDataset(FILES=config.TRAIN_FILES) 
        train_loader = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.EPOCHS * len(train_loader), eta_min=0)
        with tqdm(train_loader, unit = "batch") as tepoch:
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                grids = data['grids'].to(device)

                true_waypoints = data['waypoints']
                for key in true_waypoints["vehicles"].keys():
                    true_waypoints["vehicles"][key] = [wp.to(device) for wp in true_waypoints["vehicles"][key]]
            
                optimizer.zero_grad()
                model_outputs = model(grids)
                pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)

                loss_dict = Occupancy_Flow_Loss(true_waypoints, pred_waypoint_logits) 
                wandb.log({"observed_xe loss": loss_dict['observed_xe']})
                wandb.log({"occluded_xe loss": loss_dict['occluded_xe']})
                wandb.log({"flow loss": loss_dict['flow']})
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()
                scheduler.step()

                tepoch.set_postfix(loss=loss.item())
                wandb.log({"loss": loss.item()})
                wandb.log({"learning rate": optimizer.param_groups[0]['lr']})

        PATH = save_str +'/Epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), PATH)
    print('Finished Training. Model Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

