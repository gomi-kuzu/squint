# Visualize all SO-101 ManiSkill3 simulation tasks

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.visualization.misc import tile_images

import utils

# Add tasks
import envs
import mani_skill.envs


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Tasks to visualize
    'tasks': [
        'SO101ReachCube-v1', 'SO101ReachCan-v1',
        'SO101LiftCube-v1', 'SO101LiftCan-v1',
        'SO101PlaceCube-v1', 'SO101PlaceCan-v1',
        'SO101StackCube-v1', 'SO101StackCan-v1',
    ],

    # Environment settings
    'num_envs': 16,
    'seed': 1,
    'obs_mode': 'rgb+segmentation', # For Wrist Camera View
    'render_mode': 'rgb_array',
    'image_size': 128,
    'color_jitter': False,
    'downsample_size': 128,
    'control_mode': None,
    'domain_randomization': True,

    # Visualization settings
    'window_size': 512,
    'steps_per_task': 90,
    'reset_interval': 10,

    # Video output settings
    'save_video': False,
    'output_dir': 'outputs/visualization_videos',
    'video_fps': 30,
    'show_display': True,  # Set to False for headless environments
}


# =============================================================================
# Environment Factory
# =============================================================================

def make_env(task: str, config: dict = CONFIG):
    """Create a ManiSkill environment with the given configuration."""

    sensor_size = {'width': config['image_size'], 'height': config['image_size']}

    env_kwargs = dict(
        obs_mode=config['obs_mode'],
        render_mode=config['render_mode'],
        sensor_configs=sensor_size,
        human_render_camera_configs=sensor_size,
        num_envs=config['num_envs'],
        domain_randomization=config['domain_randomization'],
        reconfiguration_freq=None,
    )

    if config['control_mode'] is not None:
        env_kwargs['control_mode'] = config['control_mode']

    env = gym.make(task, **env_kwargs)

    if "rgb" in config['obs_mode']:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if config['downsample_size'] is not None:
            env = utils.DownsampleObsWrapper(env, target_size=config['downsample_size'])
        if config['color_jitter']:
            env = utils.ColorJitterWrapper(env)

    env.reset(seed=config['seed'])
    return env


# =============================================================================
# Visualization
# =============================================================================

def visualize_tasks(config: dict = CONFIG):
    """Visualize all configured tasks with random actions."""

    tasks = config['tasks']
    window_size = config['window_size']
    steps_per_task = config['steps_per_task']
    reset_interval = config['reset_interval']
    save_video = config.get('save_video', False)
    output_dir = config.get('output_dir', 'outputs/visualization_videos')
    video_fps = config.get('video_fps', 30)
    show_display = config.get('show_display', False)

    # Create output directory if saving videos
    if save_video:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Videos will be saved to: {output_dir}")

    for task in tasks:
        print(f"Instantiating: {task}")
        env = make_env(task, config)

        obs, info = env.reset()
        action_shape = env.action_space.shape
        num_envs = config['num_envs']
        video_nrows = int(np.sqrt(num_envs))

        # Initialize video writer if saving
        video_writer = None
        if save_video:
            video_path = os.path.join(output_dir, f"{task}.mp4")
            print(f"Saving video to: {video_path}")

        print(f"Running: {task}")

        for step in range(steps_per_task):
            # Generate action: open gripper for first 20 steps, close after
            action = np.zeros(action_shape)
            if step < 20:
                action[..., -1] = 1
            else:
                action[..., -1] = -1

            obs, reward, terminated, truncated, info = env.step(action)
            done = (terminated | truncated).any()

            # Get third-person render view (N, H, W, 3)
            render_rgb = env.render()

            # Get observation RGB (wrist camera view)
            if isinstance(obs, dict) and 'rgb' in obs:
                obs_rgb = obs['rgb']  # (N, H, W, C) where C may be 3 or 3*num_views

                # Handle multiple camera views - just take first view for simplicity
                if obs_rgb.shape[-1] != 3 and obs_rgb.shape[-1] % 3 == 0:
                    obs_rgb = obs_rgb[..., :3]  # Take first camera view

                # Resize obs to match render size (obs may be downsampled)
                render_h, render_w = render_rgb.shape[1], render_rgb.shape[2]
                if obs_rgb.shape[1] != render_h or obs_rgb.shape[2] != render_w:
                    obs_rgb = torch.nn.functional.interpolate(
                        obs_rgb.permute(0, 3, 1, 2).float(),  # (N, 3, H, W)
                        size=(render_h, render_w),
                        mode='nearest',
                    ).permute(0, 2, 3, 1).to(torch.uint8)  # (N, H, W, 3)

                # Interleave: concatenate obs and render for each env, then tile
                paired = torch.cat([obs_rgb, render_rgb], dim=2)
                rgb = tile_images(paired, nrows=video_nrows).cpu().numpy().astype(np.uint8)
                rgb = cv2.resize(rgb, dsize=(window_size * 2, window_size))
            else:
                # State mode: only show render view
                rgb = tile_images(render_rgb, nrows=video_nrows).cpu().numpy().astype(np.uint8)
                rgb = cv2.resize(rgb, dsize=(window_size, window_size))

            # Display
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Initialize video writer on first frame
            if save_video and video_writer is None:
                video_height, video_width = rgb_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    os.path.join(output_dir, f"{task}.mp4"),
                    fourcc,
                    video_fps,
                    (video_width, video_height)
                )

            # Write frame to video
            if save_video and video_writer is not None:
                video_writer.write(rgb_bgr)

            print(f"Step: {step}/{steps_per_task}, done={done}", end="\r")
            
            # Display only if requested (requires display server)
            if show_display:
                cv2.imshow("Interleaved: Obs | Render per env", rgb_bgr)
                cv2.waitKey(30)

            # Reset on interval or done
            if (step % reset_interval == 0) or done:
                env.reset()

        # Release video writer
        if video_writer is not None:
            video_writer.release()
            print(f"\nVideo saved: {os.path.join(output_dir, f'{task}.mp4')}")

        env.close()
        if show_display:
            cv2.destroyAllWindows()
        print(f"Finished: {task}                    ")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    visualize_tasks()