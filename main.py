import pathlib
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_graphics.image.transformer as tfg_transformer
from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_metrics
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis

# Data location. Please edit.
# A tfrecord containing tf.Example protos as downloaded from the Waymo Open
# Dataset (motion) webpage.

# Replace this path with your own tfrecords.
DATASET_FOLDER = '/media/hdd/benchmarks/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/uncompressed'
DATASET_PKL_FOLDER = '/media/hdd/benchmarks/Waymo_Motion/waymo_open_dataset_motion_v_1_1_0/pkls/training'
# TFRecord dataset.
TRAIN_FILES = f'{DATASET_FOLDER}/tf_example/training/training_tfexample.tfrecord*'
VAL_FILES = f'{DATASET_FOLDER}/tf_example/validation/validation_tfexample.tfrecord*'
TEST_FILES = f'{DATASET_FOLDER}/tf_example/testing/testing_tfexample.tfrecord*'
SAMPLE_FILES = f'{DATASET_FOLDER}/tf_example/sample/training_tfexample.tfrecord*'

# Text files containing validation and test scenario IDs for this challenge.
VAL_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/validation_scenario_ids.txt'
TEST_SCENARIO_IDS_FILE = f'{DATASET_FOLDER}/occupancy_flow_challenge/testing_scenario_ids.txt'

NUM_PRED_CHANNELS = 4

def create_figure_and_axes(size_pixels):
  """Initializes a unique figure and axes for plotting."""
  fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

  # Sets output image to pixel resolution.
  dpi = 100
  size_inches = size_pixels / dpi
  fig.set_size_inches([size_inches, size_inches])
  fig.set_dpi(dpi)
  fig.set_facecolor('white')
  ax.set_facecolor('white')
  ax.xaxis.label.set_color('black')
  ax.tick_params(axis='x', colors='black')
  ax.yaxis.label.set_color('black')
  ax.tick_params(axis='y', colors='black')
  fig.set_tight_layout(True)
  ax.grid(False)
  return fig, ax

def fig_canvas_image(fig):
  """Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb()."""
  # Just enough margin in the figure to display xticks and yticks.
  fig.subplots_adjust(
      left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
  fig.canvas.draw()
  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def get_colormap(num_agents):
  """Compute a color map array of shape [num_agents, 4]."""
  colors = plt.cm.get_cmap('jet', num_agents)
  colors = colors(range(num_agents))
  np.random.shuffle(colors)
  return colors

def get_viewport(all_states, all_states_mask):
  """Gets the region containing the data.

  Args:
    all_states: states of agents as an array of shape [num_agents, num_steps,
      2].
    all_states_mask: binary mask of shape [num_agents, num_steps] for
      `all_states`.

  Returns:
    center_y: float. y coordinate for center of data.
    center_x: float. x coordinate for center of data.
    width: float. Width of data.
  """
  valid_states = all_states[all_states_mask]
  all_y = valid_states[..., 1]
  all_x = valid_states[..., 0]

  center_y = (np.max(all_y) + np.min(all_y)) / 2
  center_x = (np.max(all_x) + np.min(all_x)) / 2

  range_y = np.ptp(all_y)
  range_x = np.ptp(all_x)

  width = max(range_y, range_x)

  return center_y, center_x, width

def visualize_one_step(
    states,
    mask,
    roadgraph,
    title,
    center_y,
    center_x,
    width,
    color_map,
    size_pixels=1000,
):
  """Generate visualization for a single step."""

  # Create figure and axes.
  fig, ax = create_figure_and_axes(size_pixels=size_pixels)

  # Plot roadgraph.
  rg_pts = roadgraph[:, :2].T
  ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

  masked_x = states[:, 0][mask]
  masked_y = states[:, 1][mask]
  colors = color_map[mask]

  # Plot agent current position.
  ax.scatter(
      masked_x,
      masked_y,
      marker='o',
      linewidths=3,
      color=colors,
  )

  # Title.
  ax.set_title(title)

  # Set axes.  Should be at least 10m on a side.
  size = max(10, width * 1.0)
  ax.axis([
      -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
      size / 2 + center_y
  ])
  ax.set_aspect('equal')

  image = fig_canvas_image(fig)
  plt.close(fig)
  return image

def visualize_all_agents_smooth(
    decoded_example,
    size_pixels=1000,
):
  """Visualizes all agent predicted trajectories in a serie of images.

  Args:
    decoded_example: Dictionary containing agent info about all modeled agents.
    size_pixels: The size in pixels of the output image.

  Returns:
    T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
  """
  # [num_agents, num_past_steps, 2] float32.
  past_states = tf.stack(
      [decoded_example['state/past/x'], decoded_example['state/past/y']],
      -1).numpy()
  past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

  # [num_agents, 1, 2] float32.
  current_states = tf.stack(
      [decoded_example['state/current/x'], decoded_example['state/current/y']],
      -1).numpy()
  current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

  # [num_agents, num_future_steps, 2] float32.
  future_states = tf.stack(
      [decoded_example['state/future/x'], decoded_example['state/future/y']],
      -1).numpy()
  future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

  # [num_points, 3] float32.
  roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

  num_agents, num_past_steps, _ = past_states.shape
  num_future_steps = future_states.shape[1]

  color_map = get_colormap(num_agents)

  # [num_agents, num_past_steps + 1 + num_future_steps, depth] float32.
  all_states = np.concatenate([past_states, current_states, future_states], 1)

  # [num_agents, num_past_steps + 1 + num_future_steps] float32.
  all_states_mask = np.concatenate(
      [past_states_mask, current_states_mask, future_states_mask], 1)

  center_y, center_x, width = get_viewport(all_states, all_states_mask)

  images = []

  # Generate images from past time steps.
  for i, (s, m) in enumerate(
      zip(
          np.split(past_states, num_past_steps, 1),
          np.split(past_states_mask, num_past_steps, 1))):
    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                            'past: %d' % (num_past_steps - i), center_y,
                            center_x, width, color_map, size_pixels)
    images.append(im)

  # Generate one image for the current time step.
  s = current_states
  m = current_states_mask

  im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                          center_x, width, color_map, size_pixels)
  images.append(im)

  # Generate images from future time steps.
  for i, (s, m) in enumerate(
      zip(
          np.split(future_states, num_future_steps, 1),
          np.split(future_states_mask, num_future_steps, 1))):
    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                            'future: %d' % (i + 1), center_y, center_x, width,
                            color_map, size_pixels)
    images.append(im)

  return images

def create_animation(images, interval=100):
  """ Creates a Matplotlib animation of the given images.

  Args:
    images: A list of numpy arrays representing the images.
    interval: Delay between frames in milliseconds.

  Returns:
    A matplotlib.animation.Animation.

  Usage:
    anim = create_animation(images)
    anim.save('/tmp/animation.avi')
    HTML(anim.to_html5_video())
  """

  plt.ioff()
  fig, ax = plt.subplots()
  dpi = 100
  size_inches = 1000 / dpi
  fig.set_size_inches([size_inches, size_inches])
  plt.ion()

  def animate_func(i):
    ax.imshow(images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid('off')

  anim = animation.FuncAnimation(
      fig, animate_func, frames=len(images), interval=interval)
  plt.close(fig)
  return anim

def _make_model_inputs(
    timestep_grids: occupancy_flow_grids.TimestepGrids,
    vis_grids: occupancy_flow_grids.VisGrids,
) -> tf.Tensor:
  """Concatenates all occupancy grids over past, current to a single tensor."""
  model_inputs = tf.concat(
      [
          vis_grids.roadgraph,
          timestep_grids.vehicles.past_occupancy,
          timestep_grids.vehicles.current_occupancy,
          tf.clip_by_value(
              timestep_grids.pedestrians.past_occupancy +
              timestep_grids.cyclists.past_occupancy, 0, 1),
          tf.clip_by_value(
              timestep_grids.pedestrians.current_occupancy +
              timestep_grids.cyclists.current_occupancy, 0, 1),
      ],
      axis=-1,
  )
  return model_inputs

def _get_pred_waypoint_logits(
    model_outputs: tf.Tensor) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * NUM_PRED_CHANNELS
    waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy)
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy)
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits

def main():

    filenames = tf.io.matching_files(TRAIN_FILES)
    dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    dataset = dataset.repeat()
    dataset = dataset.map(occupancy_flow_data.parse_tf_example)
    dataset = dataset.batch(1)
    it = iter(dataset)

    inputs = next(it)

    inputs_no_batch = {k: v[0] for k, v in inputs.items()}
    images = visualize_all_agents_smooth(inputs_no_batch)


    config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
    config_text = """
    num_past_steps: 10
    num_future_steps: 80
    num_waypoints: 8
    cumulative_waypoints: true
    normalize_sdc_yaw: true
    grid_height_cells: 256
    grid_width_cells: 256
    sdc_y_in_grid: 192
    sdc_x_in_grid: 128
    pixels_per_meter: 3.2
    agent_points_per_side_length: 48
    agent_points_per_side_width: 16
    """
    text_format.Parse(config_text, config)

    inputs = occupancy_flow_data.add_sdc_fields(inputs)

    timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(
        inputs=inputs, config=config)
    print(timestep_grids.vehicles.future_observed_occupancy.shape)

    true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(
        timestep_grids=timestep_grids, config=config)
    print(true_waypoints.vehicles.observed_occupancy[0].shape)
    print(true_waypoints.vehicles.occluded_occupancy[0].shape)
    print(true_waypoints.vehicles.flow[0].shape)

    vis_grids = occupancy_flow_grids.create_ground_truth_vis_grids(
        inputs=inputs, timestep_grids=timestep_grids, config=config)
    print(vis_grids.roadgraph.shape)
    print(vis_grids.agent_trails.shape)

    # Visualize waypoint 4 out of 8.
    k = 7
    observed_occupancy_grids = true_waypoints.get_observed_occupancy_at_waypoint(k)
    observed_occupancy_rgb = occupancy_flow_vis.occupancy_rgb_image(
        agent_grids=observed_occupancy_grids,
        roadgraph_image=vis_grids.roadgraph,
    )
    plt.imshow(observed_occupancy_rgb[0])

    occluded_occupancy_grids = true_waypoints.get_occluded_occupancy_at_waypoint(k)
    occluded_occupancy_rgb = occupancy_flow_vis.occupancy_rgb_image(
        agent_grids=occluded_occupancy_grids,
        roadgraph_image=vis_grids.roadgraph,
    )

    plt.imshow(occluded_occupancy_rgb[0])

    flow_rgb = occupancy_flow_vis.flow_rgb_image(
        flow=true_waypoints.vehicles.flow[k],
        roadgraph_image=vis_grids.roadgraph,
        agent_trails=vis_grids.agent_trails,
    )

    plt.imshow(flow_rgb[0])

    model_inputs = _make_model_inputs(timestep_grids, vis_grids)



    




if __name__ == '__main__':
    main()