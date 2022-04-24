
"""Occupancy and flow metrics."""
import torch
import numpy as np

# import tensorflow as tf
# import tensorflow_graphics.image.transformer as tfg_transformer

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_grids
# from core.utils.auc import AUC

def compute_occupancy_flow_metrics(config, true_waypoints, pred_waypoints):
    """Computes occupancy (observed, occluded) and flow metrics.

    Args:
        config: OccupancyFlowTaskConfig proto message.
        true_waypoints: Set of num_waypoints ground truth labels.
        pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

    Returns:
        OccupancyFlowMetrics proto message containing mean metric values averaged
        over all waypoints.
    """
    # Accumulate metric values for each waypoint and then compute the mean.
    metrics_dict = {
        'vehicles_observed_auc': [],
        'vehicles_occluded_auc': [],
        'vehicles_observed_iou': [],
        'vehicles_occluded_iou': [],
        'vehicles_flow_epe': [],
        'vehicles_flow_warped_occupancy_auc': [],
        'vehicles_flow_warped_occupancy_iou': [],
    }

    # Warp flow-origin occupancies according to predicted flow fields.
    # warped_flow_origins = _flow_warp(config=config, true_waypoints=true_waypoints, pred_waypoints=pred_waypoints)

    # Iterate over waypoints.
    for k in range(config.NUM_WAYPOINTS):
        true_observed_occupancy = true_waypoints['vehicles']['observed_occupancy'][k]
        pred_observed_occupancy = pred_waypoints['vehicles']['observed_occupancy'][k]
        true_occluded_occupancy = true_waypoints['vehicles']['occluded_occupancy'][k]
        pred_occluded_occupancy = pred_waypoints['vehicles']['occluded_occupancy'][k]
        true_flow = true_waypoints['vehicles']['flow'][k]
        pred_flow = pred_waypoints['vehicles']['flow'][k]

        # Compute occupancy metrics.
        # metrics_dict['vehicles_observed_auc'].append(_compute_occupancy_auc(true_observed_occupancy, pred_observed_occupancy))
        # metrics_dict['vehicles_occluded_auc'].append(_compute_occupancy_auc(true_occluded_occupancy, pred_occluded_occupancy))
        metrics_dict['vehicles_observed_iou'].append(_compute_occupancy_soft_iou(true_observed_occupancy, pred_observed_occupancy))
        metrics_dict['vehicles_occluded_iou'].append(_compute_occupancy_soft_iou(true_occluded_occupancy, pred_occluded_occupancy))

        # Compute flow metrics.
        metrics_dict['vehicles_flow_epe'].append(_compute_flow_epe(true_flow, pred_flow))

        # Compute flow-warped occupancy metrics.
        # First, construct ground-truth occupancy of all observed and occluded
        # vehicles.
        # true_all_occupancy = torch.clamp(true_observed_occupancy + true_occluded_occupancy, 0, 1)
        # # Construct predicted version of same value.
        # pred_all_occupancy = torch.clamp(pred_observed_occupancy + pred_occluded_occupancy, 0, 1)
        # # We expect to see the same results by warping the flow-origin occupancies.
        # flow_warped_origin_occupancy = warped_flow_origins[k]
        # # Construct quantity that requires both prediction paths to be correct.
        # flow_grounded_pred_all_occupancy = (pred_all_occupancy * flow_warped_origin_occupancy)
        # # Now compute occupancy metrics between this quantity and ground-truth.
        # metrics_dict['vehicles_flow_warped_occupancy_auc'].append(_compute_occupancy_auc(flow_grounded_pred_all_occupancy, true_all_occupancy))
        # metrics_dict['vehicles_flow_warped_occupancy_iou'].append(_compute_occupancy_soft_iou(flow_grounded_pred_all_occupancy, true_all_occupancy))

    # Compute means and return as proto message.
    metrics = {} #occupancy_flow_metrics_pb2.OccupancyFlowMetrics()
    # metrics['vehicles_observed_auc'] = _mean(metrics_dict['vehicles_observed_auc'])
    # metrics['vehicles_occluded_auc'] = _mean(metrics_dict['vehicles_occluded_auc'])
    metrics['vehicles_observed_iou'] = _mean(metrics_dict['vehicles_observed_iou'])
    metrics['vehicles_occluded_iou'] = _mean(metrics_dict['vehicles_occluded_iou'])
    metrics['vehicles_flow_epe'] = _mean(metrics_dict['vehicles_flow_epe'])
    # metrics['vehicles_flow_warped_occupancy_auc'] = _mean(metrics_dict['vehicles_flow_warped_occupancy_auc'])
    # metrics['vehicles_flow_warped_occupancy_iou'] = _mean(metrics_dict['vehicles_flow_warped_occupancy_iou'])
    return metrics


def _mean(tensor_list):
    """
    Compute mean value from a list of scalar tensors.
    """
    num_tensors = len(tensor_list)
    sum_tensors = sum(tensor_list).detach().cpu().numpy()
    return sum_tensors / num_tensors


# def _compute_occupancy_auc(true_occupancy, pred_occupancy):

#     """Computes the AUC between the predicted and true occupancy grids.

#     Args:
#         true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
#         pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

#     Returns:
#         AUC: float32 scalar.
#     """
#     auc = AUC()
#     auc.update(
#         target=true_occupancy,
#         preds=pred_occupancy,
#     )
#     return auc.compute()


def _compute_occupancy_soft_iou(true_occupancy, pred_occupancy):
    """Computes the soft IoU between the predicted and true occupancy grids.

    Args:
        true_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].
        pred_occupancy: float32 [batch_size, height, width, 1] tensor in [0, 1].

    Returns:
        Soft IoU score: float32 scalar.
    """
    true_occupancy = torch.reshape(true_occupancy, [-1])
    pred_occupancy = torch.reshape(pred_occupancy, [-1])

    intersection = torch.mean(torch.multiply(pred_occupancy, true_occupancy))
    true_sum = torch.mean(true_occupancy)
    pred_sum = torch.mean(pred_occupancy)
    # Scenes with empty ground-truth will have a score of 0.
    score = torch.div(intersection, pred_sum + true_sum - intersection)
    return score


def _compute_flow_epe(true_flow, pred_flow):
    """Computes average end-point-error between predicted and true flow fields.

    Flow end-point-error measures the Euclidean distance between the predicted and
    ground-truth flow vector endpoints.

    Args:
        true_flow: float32 Tensor shaped [batch_size, height, width, 2].
        pred_flow: float32 Tensor shaped [batch_size, height, width, 2].

    Returns:
        EPE averaged over all grid cells: float32 scalar.
    """
    # [batch_size, height, width, 2]
    diff = true_flow - pred_flow
    # [batch_size, height, width, 1], [batch_size, height, width, 1]
    true_flow_dx, true_flow_dy = torch.split(true_flow, true_flow.size(-1) // 2, dim=-1)
    # [batch_size, height, width, 1]
    flow_exists = torch.logical_or(torch.not_equal(true_flow_dx, 0.0), torch.not_equal(true_flow_dy, 0.0))
    flow_exists = flow_exists.float() # tf.cast(flow_exists, tf.float32)

    # Check shapes.
    # assert true_flow_dx.shape == true_flow_dy.shape
    # tf.debugging.assert_shapes([
    #     (true_flow_dx, ['batch_size', 'height', 'width', 1]),
    #     (true_flow_dy, ['batch_size', 'height', 'width', 1]),
    #     (diff, ['batch_size', 'height', 'width', 2]),
    # ])

    diff = diff * flow_exists
    # [batch_size, height, width, 1]
    epe = torch.linalg.norm(diff, ord=2, axis=-1, keepdims=True)
    # Scalar.
    sum_epe = torch.sum(epe)
    # Scalar.
    sum_flow_exists = torch.sum(flow_exists)
    # Scalar.
    mean_epe = torch.div(sum_epe, sum_flow_exists)

    # tf.debugging.assert_shapes([
    #     (epe, ['batch_size', 'height', 'width', 1]),
    #     (sum_epe, []),
    #     (mean_epe, []),
    # ])

    return mean_epe


# def _flow_warp(config, true_waypoints, pred_waypoints):
#   """Warps ground-truth flow-origin occupancies according to predicted flows.

#   Performs bilinear interpolation and samples from 4 pixels for each flow
#   vector.

#   Args:
#     config: OccupancyFlowTaskConfig proto message.
#     true_waypoints: Set of num_waypoints ground truth labels.
#     pred_waypoints: Predicted set of num_waypoints occupancy and flow topdowns.

#   Returns:
#     List of `num_waypoints` occupancy grids for vehicles as float32
#       [batch_size, height, width, 1] tensors.
#   """
#   h = torch.range(config.grid_height_cells, dtype=torch.float32)
#   w = torch.range(config.grid_width_cells, dtype=torch.float32)
#   h_idx, w_idx = torch.meshgrid(h, w)
#   # These indices map each (x, y) location to (x, y).
#   # [height, width, 2] but storing x, y coordinates.
#   identity_indices = torch.stack(
#       (
#           torch.transpose(w_idx),
#           torch.transpose(h_idx),
#       ),
#       axis=-1,
#   )

#   warped_flow_origins = []
#   for k in range(config.num_waypoints):
#     # [batch_size, height, width, 1]
#     flow_origin_occupancy = true_waypoints['vehicles']['flow_origin_occupancy'][k]
#     # [batch_size, height, width, 2]
#     pred_flow = pred_waypoints['vehicles']['flow'][k]
#     # Shifting the identity grid indices according to predicted flow tells us
#     # the source (origin) grid cell for each flow vector.  We simply sample
#     # occupancy values from these locations.
#     # [batch_size, height, width, 2]
#     warped_indices = identity_indices + pred_flow
#     # Pad flow_origin with a blank (zeros) boundary so that flow vectors
#     # reaching outside the grid bring in zero values instead of producing edge
#     # artifacts.
#     flow_origin_occupancy = torch.pad(flow_origin_occupancy, [[0, 0], [1, 1], [1, 1], [0, 0]])
#     # Shift warped indices as well to map to the padded origin.
#     warped_indices = warped_indices + 1
#     # NOTE: tensorflow graphics expects warp to contain (x, y) as well.
#     # [batch_size, height, width, 2]
#     warped_origin = tfg_transformer.sample(
#         image=flow_origin_occupancy,
#         warp=warped_indices,
#         pixel_type=tfg_transformer.PixelType.INTEGER,
#     )
#     warped_flow_origins.append(warped_origin)

#   return warped_flow_origins