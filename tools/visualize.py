import os
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from matplotlib.animation import PillowWriter
import torch
from torch.utils.data import DataLoader
from core.datasets.dataset import WaymoOccupancyFlowDataset
from core.models.unet_nest import R2AttU_Net
from core.utils.visual import *
from core.utils.submission import apply_sigmoid_to_occupancy_logits
from core.utils.io import get_pred_waypoint_logits
from configs import hyperparameters
cfg = hyperparameters.get_config()

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--title', help='choose a title for your wandb/log process', required=True)
    parser.add_argument('--gpu', default='0', help='device')
    parser.add_argument("--ckpt", default="pretrained/20220423_160110_R2AttU_Net/Epoch_2_Iter_30436.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./logs/vis_data/', help='path where to save output models and logs')
    args = parser.parse_args()
    return args

def main(args):
    DEVICE = 'cuda:' + args.gpu
    save_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.title
    PATH = os.path.join(args.save_dir, save_str)
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)
    # model = EfficientDetBackbone(compound_coef=1).cuda(args.gpu)
    # model = mae_vit_large_patch16_dec512d8b().cuda(args.gpu)
    model = R2AttU_Net(in_ch=cfg.INPUT_SIZE, out_ch=cfg.NUM_CLASSES, t=6).to(DEVICE)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = WaymoOccupancyFlowDataset(data_dir=cfg.VALSET_PKL_FOLDER, gpu=int(args.gpu))
    validation_loader = DataLoader(dataset=dataset, batch_size=cfg.VAL_BATCH_SIZE)
    counter = 0

    PAD = torch.zeros((1, 256, 3))
    PAD_H = torch.zeros((513, 1, 3))
    for data in tqdm(validation_loader):
        grids = data['grids'] 
        true_waypoints = data['waypoints']

        model_outputs = model(grids)
        pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)
        pred_waypoints = apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)
        
        roadgraph = grids[:, 0 ,: , :]
        roadgraph = roadgraph[None, :]
        roadgraph = torch.permute(roadgraph, (0, 2, 3, 1))

        pred_observed_occupancy_images = []
        true_observed_occupancy_images = []
        comb_observed_occupancy_images = []
        pred_occluded_occupancy_images = []
        true_occluded_occupancy_images = []
        comb_occluded_occupancy_images = []
        pred_flow_images = []
        true_flow_images = []
        comb_flow_images = []

        for k in range(cfg.NUM_WAYPOINTS):
            observed_occupancy_grids = get_observed_occupancy_at_waypoint(pred_waypoints, k)
            observed_occupancy_rgb = occupancy_rgb_image(
                agent_grids=observed_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            pred_observed_occupancy_images.append(observed_occupancy_rgb[0].detach().cpu())

        for k in range(cfg.NUM_WAYPOINTS):
            true_waypoints['pedestrians'] = None
            true_waypoints['cyclists'] = None
            observed_occupancy_grids = get_observed_occupancy_at_waypoint(true_waypoints, k)
            observed_occupancy_rgb = occupancy_rgb_image(
                agent_grids=observed_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            true_observed_occupancy_images.append(observed_occupancy_rgb[0].detach().cpu())

        for im1, im2 in zip(pred_observed_occupancy_images, true_observed_occupancy_images):
            comb_observed_occupancy_images.append(torch.concat([im1, PAD, im2], axis=0))


        for k in range(cfg.NUM_WAYPOINTS):
            occluded_occupancy_grids = get_occluded_occupancy_at_waypoint(pred_waypoints, k)
            occluded_occupancy_rgb = occupancy_rgb_image(
                agent_grids=occluded_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            pred_occluded_occupancy_images.append(occluded_occupancy_rgb[0].detach().cpu())
 
        for k in range(cfg.NUM_WAYPOINTS):
            true_waypoints['pedestrians'] = None
            true_waypoints['cyclists'] = None
            occluded_occupancy_grids = get_occluded_occupancy_at_waypoint(true_waypoints, k)
            occluded_occupancy_rgb = occupancy_rgb_image(
                agent_grids=occluded_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            true_occluded_occupancy_images.append(occluded_occupancy_rgb[0].detach().cpu())
        
        for im1, im2 in zip(pred_occluded_occupancy_images, true_occluded_occupancy_images):
            comb_occluded_occupancy_images.append(torch.concat([im1, PAD, im2], axis=0))

        # for k in range(cfg.NUM_WAYPOINTS):
        #     flow_grids = get_flow_at_waypoint(pred_waypoints, k)
        #     flow_rgb = occupancy_rgb_image(
        #         agent_grids=flow_grids,
        #         roadgraph_image=roadgraph,
        #     )
        #     pred_flow_images.append(flow_rgb[0].detach().cpu())
 
        # for k in range(cfg.NUM_WAYPOINTS):
        #     true_waypoints['pedestrians'] = None
        #     true_waypoints['cyclists'] = None
        #     flow_grids = get_flow_at_waypoint(true_waypoints, k)
        #     flow_rgb = occupancy_rgb_image(
        #         agent_grids=flow_grids,
        #         roadgraph_image=roadgraph,
        #     )
        #     true_flow_images.append(flow_rgb[0].detach().cpu())

        # for im1, im2 in zip(pred_flow_images, true_flow_images):
        #     comb_flow_images.append(torch.concat([im1, PAD, im2], axis=0))

        
        all_images = []
        for im1, im2 in zip(comb_observed_occupancy_images, comb_occluded_occupancy_images):
            all_images.append(torch.concat([im1, PAD_H, im2], axis=1))


        anim = create_animation(all_images, interval=200)
        anim.save(os.path.join(PATH, 'all_' + str(counter) + '.gif'), writer=PillowWriter(fps=5))

        counter += 1
     
    print('Finished validation. Metrics Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

