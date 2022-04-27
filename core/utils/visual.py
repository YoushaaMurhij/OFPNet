import uuid
import torch
import numpy as np
import tensorflow as tf

from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt



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
    """
    Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb().
    """
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """
    Compute a color map array of shape [num_agents, 4].
    """
    colors = plt.cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """
    Gets the region containing the data.

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
    size_pixels=1000):
    """
    Generate visualization for a single step.
    """

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


def visualize_all_agents_smooth(decoded_example, size_pixels=1000):
    """
    Visualizes all agent predicted trajectories in a serie of images.

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
    """ 
    Creates a Matplotlib animation of the given images.

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

    
def occupancy_rgb_image(agent_grids, roadgraph_image, gamma: float = 1.6):
    """
    Visualize predictions or ground-truth occupancy.
    Args:
        agent_grids: AgentGrids object containing optional
        vehicles/pedestrians/cyclists.
        roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
        gamma: Amplify predicted probabilities so that they are easier to see.
    Returns:
        [batch_size, height, width, 3] float32 RGB image.
    """
    zeros = torch.zeros_like(roadgraph_image)
    ones  = torch.ones_like(zeros)

    agents = agent_grids
    veh = zeros if agents['vehicles'] is None else torch.squeeze(agents['vehicles'], -1)
    ped = zeros if agents['pedestrians'] is None else agents['pedestrians']
    cyc = zeros if agents['cyclists'] is None else agents['cyclists']

    veh = torch.pow(veh, 1 / gamma)
    ped = torch.pow(ped, 1 / gamma)
    cyc = torch.pow(cyc, 1 / gamma)

    # Convert layers to RGB.
    rg_rgb  = torch.concat([zeros, zeros, zeros], axis=-1)
    veh_rgb = torch.concat([veh, zeros, zeros], axis=-1)  # Red.
    ped_rgb = torch.concat([zeros, ped * 0.67, zeros], axis=-1)  # Green.
    cyc_rgb = torch.concat([cyc * 0.33, zeros, zeros * 0.33], axis=-1)  # Purple.
    bg_rgb  = torch.concat([ones, ones, ones], axis=-1)  # White background.
    # Set alpha layers over all RGB channels.
    rg_a  = torch.concat([roadgraph_image, roadgraph_image, roadgraph_image], axis=-1)
    veh_a = torch.concat([veh, veh, veh], axis=-1)
    ped_a = torch.concat([ped, ped, ped], axis=-1)
    cyc_a = torch.concat([cyc, cyc, cyc], axis=-1)
    # Stack layers one by one.
    img, img_a = _alpha_blend(fg=rg_rgb, bg=bg_rgb, fg_a=rg_a)
    img, img_a = _alpha_blend(fg=veh_rgb, bg=img, fg_a=veh_a, bg_a=img_a)
    img, img_a = _alpha_blend(fg=ped_rgb, bg=img, fg_a=ped_a, bg_a=img_a)
    img, img_a = _alpha_blend(fg=cyc_rgb, bg=img, fg_a=cyc_a, bg_a=img_a)
    return img


def _alpha_blend(fg, bg, fg_a = None, bg_a = None):
  """
  Overlays foreground and background image with custom alpha values.
  Implements alpha compositing using Porter/Duff equations.
  https://en.wikipedia.org/wiki/Alpha_compositing
  Works with 1-channel or 3-channel images.
  If alpha values are not specified, they are set to the intensity of RGB
  values.
  Args:
    fg: Foreground: float32 tensor shaped [batch, grid_height, grid_width, d].
    bg: Background: float32 tensor shaped [batch, grid_height, grid_width, d].
    fg_a: Foreground alpha: float32 tensor broadcastable to fg.
    bg_a: Background alpha: float32 tensor broadcastable to bg.
  Returns:
    Output image: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
    Output alpha: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
  """
  if fg_a is None:
    fg_a = fg
  if bg_a is None:
    bg_a = bg
  eps = 1e-10
  out_a = fg_a + bg_a * (1 - fg_a)
  out_rgb = (fg * fg_a + bg * bg_a * (1 - fg_a)) / (out_a + eps)
  return out_rgb, out_a

def get_observed_occupancy_at_waypoint(waypoints, k: int):
    """
    Returns occupancies of currently-observed agents at waypoint k.
    """
    agent_grids = {'vehicles': None, 'pedestrians': None, 'cyclists' : None}
    if waypoints['vehicles']['observed_occupancy']:
      agent_grids['vehicles'] = waypoints['vehicles']['observed_occupancy'][k]
    if waypoints['pedestrians'] and waypoints['pedestrians']['observed_occupancy']:
      agent_grids['pedestrians'] = waypoints['pedestrians']['observed_occupancy'][k]
    if waypoints['cyclists'] and waypoints['cyclists']['observed_occupancy']:
      agent_grids['cyclists'] = waypoints['cyclists']['observed_occupancy'][k]
    return agent_grids

def get_occluded_occupancy_at_waypoint(waypoints, k: int):
    """
    Returns occupancies of currently-occluded agents at waypoint k.
    """
    agent_grids = {'vehicles': None, 'pedestrians': None, 'cyclists' : None}
    if waypoints['vehicles']['occluded_occupancy']:
      agent_grids['vehicles'] = waypoints['vehicles']['occluded_occupancy'][k]
    if waypoints['pedestrians'] and waypoints['pedestrians']['occluded_occupancy']:
      agent_grids['pedestrians'] = waypoints['pedestrians']['occluded_occupancy'][k]
    if waypoints['cyclists'] and waypoints['cyclists']['occluded_occupancy']:
      agent_grids['cyclists'] = waypoints['cyclists']['occluded_occupancy'][k]
    return agent_grids

def get_flow_at_waypoint(waypoints, k: int):
    """
    Returns flow fields of all agents at waypoint k.
    """
    agent_grids = {'vehicles': None, 'pedestrians': None, 'cyclists' : None}
    if waypoints['vehicles']['flow']:
      agent_grids['vehicles'] = waypoints['vehicles']['flow'][k]
    if waypoints['pedestrians'] and waypoints['pedestrians']['flow']:
      agent_grids['pedestrians'] = waypoints['pedestrians']['flow'][k]
    if waypoints['cyclists'] and waypoints['cyclists']['flow']:
      agent_grids['cyclists'] = waypoints['cyclists']['flow'][k]
    return agent_grids