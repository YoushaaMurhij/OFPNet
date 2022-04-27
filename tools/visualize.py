import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from core.datasets.dataset import WaymoOccupancyFlowDataset
from core.models.unet_nest import R2AttU_Net
from core.utils.visual import get_observed_occupancy_at_waypoint, occupancy_rgb_image, create_animation
from core.utils.io import get_pred_waypoint_logits
from core.utils.submission import apply_sigmoid_to_occupancy_logits
from configs import hyperparameters
cfg = hyperparameters.get_config()

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--title', help='choose a title for your wandb/log process', required=True)
    parser.add_argument('--gpu', default='0', help='device')
    parser.add_argument("--ckpt", default="pretrained/20220423_160110_R2AttU_Net/Epoch_1_Iter_30436.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./logs/vis_data/', help='path where to save output models and logs')
    args = parser.parse_args()
    return args

def main(args):

    save_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.title
    PATH = os.path.join(args.save_dir, save_str)
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)
    # model = EfficientDetBackbone(compound_coef=1).cuda(args.gpu)
    # model = mae_vit_large_patch16_dec512d8b().cuda(args.gpu)
    model = R2AttU_Net(in_ch=cfg.INPUT_SIZE, out_ch=cfg.NUM_CLASSES, t=6).cuda(args.gpu)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = WaymoOccupancyFlowDataset(data_dir=cfg.VALSET_PKL_FOLDER, gpu=int(args.gpu))
    validation_loader = DataLoader(dataset=dataset, batch_size=cfg.VAL_BATCH_SIZE)



    for data in tqdm(validation_loader):
        grids = data['grids'] 
        true_waypoints = data['waypoints']

        model_outputs = model(grids)
        pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)

        pred_waypoints = apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)

        images = []
        roadgraph = grids[:, : ,: , 1]
        for k in range(cfg.num_waypoints):
            observed_occupancy_grids = get_observed_occupancy_at_waypoint(pred_waypoints, k)
            observed_occupancy_rgb = occupancy_rgb_image(
                agent_grids=observed_occupancy_grids,
                roadgraph_image=roadgraph,
            )
            images.append(observed_occupancy_rgb[0])

        anim = create_animation(images, interval=200)
        HTML(anim.to_html5_video())

     
    print('Finished validation. Metrics Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

