import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import torch
from torch.utils.data import DataLoader
from core.datasets.dataset import WaymoOccupancyFlowDataset
from core.models.unet_nest import R2AttU_Net
from core.utils.occupancy_flow_metrics import compute_occupancy_flow_metrics
from core.utils.submission import *
from configs import hyperparameters
cfg = hyperparameters.get_config()

def parse_args():
    parser = argparse.ArgumentParser(description='Occupancy and Flow Prediction Model Training')
    parser.add_argument('--title', help='choose a title for your wandb/log process', required=True)
    parser.add_argument('--gpu', default='0', help='device')
    parser.add_argument("--ckpt", default="logs/train_data/20220422_205330_R2AttU_Net_New/Epoch_1_Iter_7609.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./logs/val_data/', help='path where to save output models and logs')
    args = parser.parse_args()
    return args

def main(args):

    save_str = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.title
    PATH = os.path.join(args.save_dir, save_str)
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)
    # model = EfficientDetBackbone(compound_coef=1).to(DEVICE)
    # model = mae_vit_large_patch16_dec512d8b().to(DEVICE)
    model = R2AttU_Net(in_ch=cfg.INPUT_SIZE, out_ch=cfg.NUM_CLASSES, t=6).to(DEVICE)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = WaymoOccupancyFlowDataset(data_dir=cfg.VALSET_PKL_FOLDER, gpu=int(args.gpu))
    validation_loader = DataLoader(dataset=dataset, batch_size=cfg.VAL_BATCH_SIZE)

    metrics_dict = {
            'vehicles_observed_auc': [],
            'vehicles_occluded_auc': [],
            'vehicles_observed_iou': [],
            'vehicles_occluded_iou': [],
            'vehicles_flow_epe': [],
            'vehicles_flow_warped_occupancy_auc': [],
            'vehicles_flow_warped_occupancy_iou': [],
        }

    for data in tqdm(validation_loader):
        grids = data['grids'] 
        true_waypoints = data['waypoints']

        model_outputs = model(grids)
        pred_waypoint_logits = get_pred_waypoint_logits(model_outputs)

        pred_waypoints = apply_sigmoid_to_occupancy_logits(pred_waypoint_logits)
        metrics = compute_occupancy_flow_metrics(config=cfg, true_waypoints=true_waypoints, pred_waypoints=pred_waypoints)

        # wandb.log({"observed_xe loss": loss_dict['observed_xe']})
        # wandb.log({"occluded_xe loss": loss_dict['occluded_xe']})
        # wandb.log({"flow loss": loss_dict['flow']})
        metrics_dict['vehicles_observed_iou'].append(metrics['vehicles_observed_iou'])
        metrics_dict['vehicles_occluded_iou'].append(metrics['vehicles_occluded_iou'])
        metrics_dict['vehicles_flow_epe'].append(metrics['vehicles_flow_epe'])

    Final_Metrics = {}
    Final_Metrics['vehicles_observed_iou'] =  sum(metrics_dict['vehicles_observed_iou']) / len(validation_loader)
    Final_Metrics['vehicles_occluded_iou'] =  sum(metrics_dict['vehicles_occluded_iou']) / len(validation_loader)
    Final_Metrics['vehicles_flow_epe']     =  sum(metrics_dict['vehicles_flow_epe'])     / len(validation_loader)

    print(Final_Metrics)
    print(json.dumps(Final_Metrics, sort_keys=True, indent=4))
    with open(os.path.join(PATH, args.title + '.json'), 'w') as handle:
            json.dump(Final_Metrics , handle, sort_keys=True, indent=4)
     
    print('Finished validation. Metrics Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

