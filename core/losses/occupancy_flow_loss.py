from functools import reduce
from multiprocessing import reduction
from tkinter.tix import Tree
import torch
import torch.nn.functional as F
from configs import config
def Occupancy_Flow_Loss(true_waypoints, pred_waypoint_logits):
    """Loss function.

    Args:
      config: OccupancyFlowTaskConfig proto message.
      true_waypoints: Ground truth labels.
      pred_waypoint_logits: Predicted occupancy logits and flows.

    Returns:
      A dict containing different loss tensors:
        observed_xe: Observed occupancy cross-entropy loss.
        occluded_xe: Occluded occupancy cross-entropy loss.
        flow: Flow loss.
    """
    loss_dict = {}
    # Store loss tensors for each waypoint and average at the end.
    loss_dict['observed_xe'] = []
    loss_dict['occluded_xe'] = []
    loss_dict['flow'] = []

    # Iterate over waypoints.
    for k in range(config.NUM_WAYPOINTS):
        # Occupancy cross-entropy loss.
        pred_observed_occupancy_logit = (pred_waypoint_logits['vehicles']['observed_occupancy'][k])
        pred_occluded_occupancy_logit = (pred_waypoint_logits['vehicles']['occluded_occupancy'][k])
        true_observed_occupancy = true_waypoints['vehicles']['observed_occupancy'][k]
        true_occluded_occupancy = true_waypoints['vehicles']['occluded_occupancy'][k]

        # Accumulate over waypoints.
        loss_dict['observed_xe'].append(_sigmoid_xe_loss(true_occupancy=true_observed_occupancy, pred_occupancy=pred_observed_occupancy_logit))
        loss_dict['occluded_xe'].append(_sigmoid_xe_loss(true_occupancy=true_occluded_occupancy, pred_occupancy=pred_occluded_occupancy_logit))

        # Flow loss.
        pred_flow = pred_waypoint_logits['vehicles']['flow'][k]
        true_flow = true_waypoints['vehicles']['flow'][k]
        loss_dict['flow'].append(_flow_loss(pred_flow, true_flow))

    # Mean over waypoints.
    loss_dict['observed_xe'] = (sum(loss_dict['observed_xe']) / config.NUM_WAYPOINTS)
    loss_dict['occluded_xe'] = (sum(loss_dict['occluded_xe']) / config.NUM_WAYPOINTS)
    loss_dict['flow']        = sum(loss_dict['flow']) / config.NUM_WAYPOINTS

    return loss_dict


def _sigmoid_xe_loss(true_occupancy, pred_occupancy, loss_weight: float = 1000):
    """Computes sigmoid cross-entropy loss over all grid cells."""
    # Since the mean over per-pixel cross-entropy values can get very small,
    # we compute the sum and multiply it by the loss weight before computing
    # the mean.
    xe_sum = F.binary_cross_entropy_with_logits(input=torch.flatten(pred_occupancy), target=torch.flatten(true_occupancy), reduce=True, reduction='sum')
    # Return mean.
    return loss_weight * xe_sum / list(torch.flatten(pred_occupancy).size())[0] # torch.shape(pred_occupancy, out_type=torch.float32)   


def _flow_loss(true_flow, pred_flow, loss_weight: float = 1):
    """Computes L1 flow loss."""
    diff = true_flow - pred_flow
    # Ignore predictions in areas where ground-truth flow is zero.
    # [batch_size, height, width, 1], [batch_size, height, width, 1]
    (true_flow_dx, true_flow_dy) = torch.split(true_flow, true_flow.size(-1) // 2, dim=-1)
    # [batch_size, height, width, 1]
    flow_exists = torch.logical_or(torch.not_equal(true_flow_dx, 0.0), torch.not_equal(true_flow_dy, 0.0))
    flow_exists = flow_exists.type(torch.float32)
    diff = diff * flow_exists
    diff_norm = torch.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
    mean_diff = torch.div(torch.sum(diff_norm), torch.sum(flow_exists) / 2)  # / 2 since (dx, dy) is counted twice.
    return loss_weight * mean_diff


# def _batch_flatten(input_tensor: tf.Tensor) -> tf.Tensor:
#   """Flatten tensor to a shape [batch_size, -1]."""
#   image_shape = tf.shape(input_tensor)
#   return tf.reshape(input_tensor, tf.concat([image_shape[0:1], [-1]], 0))