"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2022.03.26
# Email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Training script for Occupancy and Flow Prediction
"""
import os
import time
import wandb
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from core.datasets.dataset import WaymoOccupancyFlowDataset
from core.models.efficientdet.backbone import EfficientDetBackbone
# from core.models.unet_nest import R2AttU_Net
from core.losses.occupancy_flow_loss import Occupancy_Flow_Loss
from core.utils.io import get_pred_waypoint_logits
from configs import config

os.environ["WANDB_API_KEY"] = 'cccdc2dfb027090440d22b2ea4b94d57b9724115'
os.environ["WANDB_MODE"] = "disabled"  # {'run', 'online', 'offline', 'dryrun', 'disabled'}
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--resume', help='resume from checkpoint', action="store_true")
    parser.add_argument("--pretrained", default="/logs/Epoch_4.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir'  , default='/home/workspace/Occ_Flow_Pred/logs/train_data/', help='path where to save output models and logs')

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus' , default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr'  , default=0, type=int, help='ranking within the nodes')

    parser.add_argument('--master_port', help='specify a port', required=True)
    parser.add_argument('--title'      , help='choose a title for your wandb/log process', required=True)

    args = parser.parse_args()
    return args

def train(gpu, args):
    wandb.init(project="occupancy-flow", entity="youshaamurhij", name=args.title)

    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )    

    wandb.config.update(args)

    save_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.title
    PATH = os.path.join(args.save_dir, save_str)
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)

    torch.cuda.set_device(gpu)
    # model = R2AttU_Net(in_ch=config.INPUT_SIZE, out_ch=config.NUM_CLASSES, t=6).cuda(gpu)
    model = EfficientDetBackbone(compound_coef=1).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    print("Model structure: ")
    print(model)

    if config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    else:
        optimizer = optim.SGD(model.parameters(), weight_decay = config.WEIGHT_DECAY, lr=config.LR, momentum=config.MOMENTUM)

    epoch = 0
    if args.resume:
        checkpoint = torch.load(args.pretrained, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'Weights are loaded from: {args.pretrained}.')

    dataset = WaymoOccupancyFlowDataset(data_dir=config.DATASET_PKL_FOLDER, gpu=gpu) 

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size, 
        rank=rank)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=False,     
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler)


    if config.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.EPOCHS * len(train_loader), eta_min=0)
    
    model.train()
    while epoch <= config.EPOCHS:
        epoch += 1
        with tqdm(train_loader, unit = "batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            for j, data in enumerate(tepoch):
                grids = data['grids'] 

                true_waypoints = data['waypoints']
                # for key in true_waypoints["vehicles"].keys():
                #     true_waypoints["vehicles"][key] = [wp.to(device) for wp in true_waypoints["vehicles"][key]]
            
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

                # if j % 50 == 0:
                #     print("Epoch:", epoch + 1,"of", N_EPOCHS, "| Iter:", '{:0>6}'.format(j),"of", N_DATALOADER,
                #      "| Loss:", '{:.5f}'.format(loss.item()), "| Lr:", '{:.6f}'.format(optimizer.param_groups[0]['lr']),
                #      "| Forward Time:", '{:.2f}s'.format(t_end - t_start), "| Estimated Time:", '{:.1f}h'.format((t_end - t_start) * (N_DATALOADER - j) // 3600 + 1))

                tepoch.set_postfix(loss=loss.item())
                wandb.log({"loss": loss.item()})
                wandb.log({"learning rate": optimizer.param_groups[0]['lr']})

        print("Saving checkpoint for epoch:", epoch)
        CKPT_DIR = PATH +'/Epoch_'+str(epoch)+'.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, CKPT_DIR)

    print('Finished Training. Model Saved!')

if __name__=="__main__":
    args = parse_args()

    args.world_size = args.gpus * args.nodes               
    os.environ['MASTER_ADDR'] = 'localhost'            
    os.environ['MASTER_PORT'] = args.master_port  # '8882'                    
    mp.spawn(train, nprocs=args.gpus, args=(args,))

    # main(args)

