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
from core.models.wnet import WNet
from core.models.xception import Xception
from core.models.unet_head import R2AttU_sepHead
from core.models.unet_seq import R2AttU_seq
from core.models.unext import UNext

from core.models.unet_nest import R2AttU_Net
from core.losses.occupancy_flow_loss import Occupancy_Flow_Loss
from core.utils.io import get_pred_waypoint_logits
from configs import hyperparameters
cfg = hyperparameters.get_config()

os.environ["WANDB_API_KEY"] = 'cccdc2dfb027090440d22b2ea4b94d57b9724115'
os.environ["WANDB_MODE"]    = cfg.WANDB_MODE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument("--pretrained" , default="./pretrained/Epoch_5.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir'   , default='/home/workspace/Occ_Flow_Pred/logs/train_data/', help='path where to save output models and logs')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus' , default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr'  , default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--resume'     , help='resume from checkpoint', action="store_true")
    parser.add_argument('--master_port', help='specify a port', default="8898")
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
    wandb.config.update(cfg)

    save_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.title
    PATH = os.path.join(args.save_dir, save_str)
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)

    torch.cuda.set_device(gpu)
    # model = Xception('xception71', in_channels=23, time_limit=8, n_traj=64, with_head=True).cuda(gpu)
    # model = R2AttU_seq(img_ch=23, output_ch=32, t=1).cuda(gpu)
    # model = R2AttU_Net(in_ch=cfg.INPUT_SIZE, out_ch=cfg.NUM_CLASSES, t=2).cuda(gpu)
    # model = WNet(img_ch=cfg.INPUT_SIZE, output_ch=cfg.NUM_CLASSES, t=1).cuda(gpu)
    model = UNext(num_classes=32).cuda(gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)
    print("Model structure: ")
    print(model)

    if cfg.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    else:
        optimizer = optim.SGD(model.parameters(), weight_decay = cfg.WEIGHT_DECAY, lr=cfg.LR, momentum=cfg.MOMENTUM)
    epoch = 0
    if args.resume:
        checkpoint = torch.load(args.pretrained, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f'Weights are loaded from: {args.pretrained}.')

    dataset = WaymoOccupancyFlowDataset(data_dir=cfg.DATASET_PKL_FOLDER, gpu=gpu) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size, 
        rank=rank)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=False,     
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler)
    if cfg.SCHEDULER == 'GetInitLR':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.3333333333333333, total_iters=5, last_epoch=- 1, verbose=False)
    elif cfg.SCHEDULER == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)
    elif cfg.SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=0)
    elif cfg.SCHEDULER == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader),
        T_mult=1,
        eta_min=max(1e-2 * cfg.LR, 1e-6),
        last_epoch=-1)
    else:
        raise AttributeError("This scheduler is not implemented ")
    # torch.autograd.set_detect_anomaly(True)
    model.train()
    while epoch <= cfg.EPOCHS:
        epoch += 1
        with tqdm(train_loader, unit = "batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            TMP_CKPT_SPLT = len(tepoch) // 4
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

                if j % TMP_CKPT_SPLT == 0 and j:
                    print("Saving temporary checkpoint for epoch:", epoch, ", iteration:", j)
                    TMP_CKPT_DIR = PATH +'/Epoch_'+ str(epoch) +'_Iter_'+ str(j) + '.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, TMP_CKPT_DIR)

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

