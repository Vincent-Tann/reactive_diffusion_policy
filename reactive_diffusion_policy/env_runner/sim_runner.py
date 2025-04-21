import threading
import time
import os.path as osp
import numpy as np
import torch
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional
import transforms3d as t3d
import cv2
from robomimic.utils.file_utils import EnvUtils
import json
import h5py
from omegaconf import DictConfig, ListConfig
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions
)

import os
import psutil
from copy import deepcopy

# set looger level
import sys
logger.remove()
logger.add(sys.stdout, level="DEBUG")

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 10
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

class SimRunner:
    def __init__(self,
                 output_dir: str,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 latent_tcp_ensemble_buffer_params: DictConfig = None,
                 latent_gripper_ensemble_buffer_params: DictConfig = None,
                 use_latent_action_with_rnn_decoder: bool = False,
                 use_relative_action: bool = False,
                 use_relative_tcp_obs_for_relative_action: bool = True,
                 action_interpolation_ratio: int = 1,
                 eval_episodes=10,
                 max_ep_length: int = 200,
                 tcp_action_update_interval: int = 6,
                 gripper_action_update_interval: int = 10,
                 tcp_pos_clip_range: ListConfig = ListConfig([[0.6, -0.4, 0.03], [1.0, 0.45, 0.4]]),
                 tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, 0., np.pi], [-np.pi, 0., np.pi]]),
                 tqdm_interval_sec = 5.0,
                 control_fps: float = 12,
                 inference_fps: float = 6,
                 latency_step: int = 0,
                 gripper_latency_step: Optional[int] = None,
                 n_obs_steps: int = 2,
                 obs_temporal_downsample_ratio: int = 2,
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 enable_video_recording: bool = False,
                 save_video_path: Optional[str] = None,
                 task_name=None,
                 ):
        self.task_name = task_name
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes
        self.max_ep_length = max_ep_length

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys

        # Initialize the environment

        # Load environment from dataset file
        f_path = env_params.dataset_hdf5_path
        f_in = h5py.File(f_path, 'r')
        env_meta = json.loads(f_in["data"].attrs["env_args"])
        f_in.close()
        
        # import pdb; pdb.set_trace()
        
        self.env = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta,
            camera_names=env_params.camera_names,
            camera_height=env_params.camera_height,
            camera_width=env_params.camera_width,
            reward_shaping=True,
        )

        self.max_ep_length = max_ep_length
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        if self.use_latent_action_with_rnn_decoder:
            assert latent_tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            assert latent_gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            self.tcp_ensemble_buffer = EnsembleBuffer(**latent_tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**latent_gripper_ensemble_buffer_params)
        else:
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)
        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio

        self.enable_video_recording = enable_video_recording
        self.save_video_path = save_video_path
        if enable_video_recording:
            self.video_dir = osp.join(output_dir, 'videos') if save_video_path is None else save_video_path
            os.makedirs(self.video_dir, exist_ok=True)

        self.stop_event = threading.Event()
        self.obs_history = []
        self.camera_names = env_params.camera_names
        
        self.episode_stats_lock = threading.Lock()  # Add a lock for thread safety

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)

        for key in self.lowdim_keys:
            if "wrt" not in key:
                # Check if the observation is already 2D (has time dimension)
                if len(obs_dict[key].shape) == 2:
                    obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]
                else:
                    # If 1D, reshape to add time dimension (assuming single time step)
                    obs_dict[key] = obs_dict[key][:self.shape_meta['obs'][key]['shape'][0]][np.newaxis, :]

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        return obs_dict, absolute_obs_dict

    def pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        extended_obs_dict = deepcopy(extended_obs_dict)

        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            # Check if the observation is already 2D (has time dimension)
            if len(extended_obs_dict[key].shape) == 2:
                extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            else:
                # If 1D, reshape to add time dimension (assuming single time step)
                extended_obs_dict[key] = extended_obs_dict[key][:self.shape_meta['extended_obs'][key]['shape'][0]][np.newaxis, :]
            
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = extended_obs_dict[key][-1].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)

        return extended_obs_dict, absolute_extended_obs_dict

    def post_process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Post-process the action before sending to the robot
        """
        assert len(action.shape) == 2  # (action_steps, d_a)
        
        # For robomimic, we need to convert the action to the format expected by the environment
        # This will depend on the specific environment and action space
        
        # For now, we'll assume the action is already in the correct format
        # and just return it as is, with some basic clipping
        
        # # Clip position if needed
        # if action.shape[-1] >= 3:  # If action includes position
        #     action[:, :3] = np.clip(action[:, :3], 
        #                             np.array(self.tcp_pos_clip_range[0][:3]), 
        #                             np.array(self.tcp_pos_clip_range[1][:3]))
        
        # # If action includes gripper, ensure it's within [-1, 1] range
        # if action.shape[-1] == 7:  # pos(3) + rot(3) + gripper(1)
        #     action[:, -1] = np.clip(action[:, -1], -1, 1)
        
        # return action
        
        if action.shape[-1] == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
            # convert to 6D pose
            left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  # (action_steps, 3, 3)
            left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
            left_trans_batch = action[:, :3]  # (action_steps, 3)
            left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
            
            # clip
            left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[0]), np.array(self.tcp_pos_clip_range[1]))
            left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[0]), np.array(self.tcp_rot_clip_range[1]))
            
            # concatenate gripper action
            left_action = np.concatenate([left_action_6d, action[:, 9:10]], axis=1)
            
            return left_action
        
        else:
            raise NotImplementedError   

    def get_obs_dict(self, env_obs, obs_steps=1):
        """
        Convert environment observation to the format expected by the policy
        """
        # Initialize observation dictionary
        obs_dict = {}
        
        # Process image observations
        robomimic_key_to_rdp_key = {
            "robot0_eye_in_hand_image": "left_wrist_img",
            "agentview_image": "external_img",
        }
        for i, camera_name in enumerate(self.camera_names):
            robomimic_img_key = f"{camera_name}_image"
            rdp_img_key = robomimic_key_to_rdp_key.get(robomimic_img_key, None)
            if rdp_img_key in self.rgb_keys:
                # Get image from environment observation
                img = env_obs[robomimic_img_key]
                
                # Repeat the image for the required number of observation steps
                img_stack = np.repeat(img[np.newaxis, ...], obs_steps, axis=0)
                
                # Transpose from (T, H, W, C) to (T, C, H, W) for PyTorch
                img_stack = img_stack.transpose(0, 3, 1, 2)
                
                obs_dict[rdp_img_key] = img_stack
                
                # import pdb; pdb.set_trace()
        
        # Process low-dimensional observations
        # this is tring to query all 'key's from self.lowdim_keys (defined in config file) in env_obs. Note that the actual keys used should be matched witht he robomimic env.
        for key in self.lowdim_keys:
            if key == 'left_robot_tcp_pose' or key == 'right_robot_tcp_pose':
                # Get robot state from environment observation
                pos = env_obs["robot0_eef_pos"]
                quat = env_obs["robot0_eef_quat"]
                
                # convert quat to 6d rotation (the first 2 columns of the rotation matrix)
                rot = t3d.quaternions.quat2mat(quat)[:, :2].T.flatten()
                
                # Convert to the format expected by the policy
                # concatenate position and rotation
                tcp_pose = np.concatenate([pos, rot])
                
                # Repeat for the required number of observation steps
                tcp_pose_stack = np.repeat(tcp_pose[np.newaxis, ...], obs_steps, axis=0)
                
                obs_dict[key] = tcp_pose_stack
            
            elif key == 'left_robot_gripper_width' or key == 'gripper_width':
                gripper_qpose = env_obs['robot0_gripper_qpos']
                gripper_width = np.array([gripper_qpose[0] - gripper_qpose[1]])  # Make sure it's an array
                
                # Repeat for the required number of observation steps
                gripper_width_stack = np.repeat(gripper_width[np.newaxis, ...], obs_steps, axis=0)
                
                obs_dict[key] = gripper_width_stack
                
            elif 'wrench' in key:
                # wrench = env_obs['ft']
                # # Repeat for the required number of observation steps
                # wrench_stack = np.repeat(wrench[np.newaxis, ...], obs_steps, axis=0)
                
                # obs_dict[key] = wrench_stack
            
                # Quiry wrench seperately
                force = self.env.env.robots[0].ee_force
                torque = self.env.env.robots[0].ee_torque
                wrench = np.concatenate([force, torque])
                # Repeat for the required number of observation steps
                wrench_stack = np.repeat(wrench[np.newaxis, ...], obs_steps, axis=0)
                
                obs_dict[key] = wrench_stack
                
        return obs_dict

    def action_command_thread(self, policy: Union[DiffusionUnetImagePolicy], stop_event):
        while not stop_event.is_set():
            start_time = time.time()
            # get step action from ensemble buffer
            tcp_step_action = self.tcp_ensemble_buffer.get_action()
            gripper_step_action = self.gripper_ensemble_buffer.get_action()
            if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                             f"cur_time-start_time: {cur_time - start_time}")
                self.action_step_count += 1
                continue

            if self.use_latent_action_with_rnn_decoder:
                tcp_extended_obs_step = int(tcp_step_action[-1])
                gripper_extended_obs_step = int(gripper_step_action[-1])
                tcp_step_action = tcp_step_action[:-1]
                gripper_step_action = gripper_step_action[:-1]

                longer_extended_obs_step = max(tcp_extended_obs_step, gripper_extended_obs_step)
                
                # Get extended observation from history
                if longer_extended_obs_step < len(self.obs_history):
                    extended_obs = self.obs_history[-longer_extended_obs_step]
                else:
                    # If we don't have enough history, use the latest observation
                    extended_obs = self.obs_history[-1] if self.obs_history else self.current_obs
                
                extended_obs_dict = self.get_obs_dict(extended_obs, obs_steps=1)

                if self.use_relative_action:
                    action_dim = self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
                    if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                        action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
                    # the last action_dim is the base absolute action; the rest is the latent action.
                    tcp_base_absolute_action = tcp_step_action[-action_dim:]
                    gripper_base_absolute_action = gripper_step_action[-action_dim:]
                    # get rid of the base absolute action from tcp_step_action
                    tcp_step_action = tcp_step_action[:-action_dim]
                    gripper_step_action = gripper_step_action[:-action_dim]
                    # import pdb; pdb.set_trace()

                np_extended_obs_dict = dict(extended_obs_dict)
                np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)
                extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

                tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
                gripper_step_latent_action = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

                # import pdb; pdb.set_trace()
                dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio
                # tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                # gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                
                tcp_result = policy.predict_from_latent_action(
                    tcp_step_latent_action, 
                    extended_obs_dict, 
                    tcp_extended_obs_step, 
                    dataset_obs_temporal_downsample_ratio
                )
                
                gripper_result = policy.predict_from_latent_action(
                    gripper_step_latent_action, 
                    extended_obs_dict, 
                    gripper_extended_obs_step, 
                    dataset_obs_temporal_downsample_ratio
                )
                
                # Check if the action is empty and handle it
                if tcp_result['action'].numel() == 0:
                    # Use the first action from action_pred instead
                    logger.warning("TCP action is empty, using first action from action_pred")
                    tcp_step_action = tcp_result['action_pred'][0].detach().cpu().numpy()
                else:
                    tcp_step_action = tcp_result['action'][0].detach().cpu().numpy()
                
                if gripper_result['action'].numel() == 0:
                    # Use the first action from action_pred instead
                    logger.warning("Gripper action is empty, using first action from action_pred")
                    gripper_step_action = gripper_result['action_pred'][0].detach().cpu().numpy()
                else:
                    gripper_step_action = gripper_result['action'][0].detach().cpu().numpy()
                    
                if self.use_relative_action:
                    # import pdb; pdb.set_trace()
                    tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action)
                    gripper_step_action = relative_actions_to_absolute_actions(gripper_step_action, gripper_base_absolute_action)

                if tcp_step_action.shape[-1] == 4: # (x, y, z, gripper_width)
                    tcp_len = 3
                elif tcp_step_action.shape[-1] == 8: # (x_l, y_l, z_l, x_r, y_r, z_r, gripper_width_l, gripper_width_r)
                    tcp_len = 6
                elif tcp_step_action.shape[-1] == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
                    tcp_len = 9
                elif tcp_step_action.shape[-1] == 20: # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r, gripper_width_l, gripper_width_r)
                    tcp_len = 18
                else:
                    raise NotImplementedError
                tcp_step_action = tcp_step_action[-1]
                gripper_step_action = gripper_step_action[-1]

                tcp_step_action = tcp_step_action[:tcp_len]
                gripper_step_action = gripper_step_action[tcp_len:]

            # import pdb; pdb.set_trace()
            # Combine TCP and gripper actions
            combined_action = np.concatenate([tcp_step_action, gripper_step_action])
            
            # Post-process action for the environment
            step_action = self.post_process_action(combined_action[np.newaxis, :]).squeeze(0)
            
            # Execute action in environment
            obs, reward, done, info = self.env.step(step_action)
            
            # Update observation history
            self.current_obs = obs
            self.obs_history.append(obs)
            if len(self.obs_history) > 100:  # Limit history size
                self.obs_history.pop(0)
            
            # Update episode stats with thread lock to ensure visibility
            with self.episode_stats_lock:
                self.episode_reward += reward
                self.episode_length += 1
                
                # Check if episode is done
                if done:
                    self.episode_done = True
                    logger.debug(f"Episode done set to True in action thread, length: {self.episode_length}")
            
            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
            self.action_step_count += 1

    def save_video(self, frames, episode_idx):
        """
        Save frames as a video
        """
        if not frames:
            return
            
        video_path = osp.join(self.video_dir, f'episode_{episode_idx}.mp4')
        height, width, _ = frames[0].shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        logger.info(f"Video saved to {video_path}")

    def run(self, policy: Union[DiffusionUnetImagePolicy]):
        if self.use_latent_action_with_rnn_decoder:
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."

        device = policy.device
        
        all_returns = []
        all_ep_lengths = []

        try:
            for episode_idx in tqdm.tqdm(range(0, self.eval_episodes),
                                        desc=f"Eval for {self.task_name}",
                                        leave=False, mininterval=self.tqdm_interval_sec):
                logger.info(f"Start evaluation episode {episode_idx}")
                
                # Reset environment
                obs = self.env.reset()
                self.current_obs = obs
                self.obs_history = [obs]  # Initialize observation history
                
                # Reset policy and buffers
                policy.reset()
                self.tcp_ensemble_buffer.clear()
                self.gripper_ensemble_buffer.clear()
                logger.debug("Reset environment and policy.")
                
                # Initialize episode variables
                
                with self.episode_stats_lock:
                    self.episode_done = False
                    self.episode_reward = 0
                    self.episode_length = 0
                    logger.debug("Reset episode stats")
                frames = [] if self.enable_video_recording else None
                
                # Start action command thread
                self.action_step_count = 0
                self.stop_event.clear()
                action_thread = threading.Thread(target=self.action_command_thread, args=(policy, self.stop_event,),
                                                daemon=True)
                action_thread.start()
                
                # Give the action thread a moment to start
                time.sleep(0.1)
                
                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps)
                start_timestamp = time.time()
                
                try:
                    # Use a timeout to prevent infinite loops
                    # max_episode_time = self.max_ep_length * (1.0 / self.control_fps) * 2  # Double the expected time
                    while True:
                        # Check episode status with lock
                        with self.episode_stats_lock:
                            is_done = self.episode_done
                            current_length = self.episode_length
                        
                        if is_done or current_length >= self.max_ep_length:
                            logger.debug(f"Breaking loop: done={is_done}, length={current_length}, max={self.max_ep_length}")
                            break
                        
                        # # Check for timeout
                        # if time.time() - start_timestamp > max_episode_time:
                        #     logger.warning(f"Episode timed out after {max_episode_time} seconds")
                        #     break
                        
                        logger.debug(f"Step: {step_count}, Episode length: {current_length}, Episode done: {is_done}")
                        start_time = time.time()
                        
                        # Get observation
                        # import pdb; pdb.set_trace()
                        obs_dict = self.get_obs_dict(self.current_obs, obs_steps=self.n_obs_steps)
                        
                        # Pre-process observation
                        np_obs_dict, np_absolute_obs_dict = self.pre_process_obs(obs_dict)
                        
                        # Device transfer
                        obs_dict = dict_apply(np_obs_dict,
                                            lambda x: torch.from_numpy(x).unsqueeze(0).to(
                                                device=device))

                        policy_time = time.time()
                        # Run policy
                        with torch.no_grad():
                            if self.use_latent_action_with_rnn_decoder:
                                action_dict = policy.predict_action(obs_dict,
                                                                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                                    return_latent_action=True)
                            else:
                                action_dict = policy.predict_action(obs_dict)
                        logger.debug(f"Policy inference time: {time.time() - policy_time:.3f}s")
                        # Device transfer
                        np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                        
                        action_all = np_action_dict['action'].squeeze(0)
                        if self.use_latent_action_with_rnn_decoder:
                            # Add first absolute action to get absolute action
                            if self.use_relative_action:
                                base_absolute_action = np.concatenate([
                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                ], axis=-1)
                                action_all = np.concatenate([
                                    action_all,
                                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                                ], axis=-1)
                            # Add action step to get corresponding observation
                            action_all = np.concatenate([
                                action_all,
                                np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, 
                                        action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]
                            ], axis=-1)
                        else:
                            if self.use_relative_action:
                                base_absolute_action = np.concatenate([
                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                ], axis=-1)
                                action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)
                        
                        if self.action_interpolation_ratio > 1:
                            if self.use_latent_action_with_rnn_decoder:
                                action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                            else:
                                action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)
                        
                        # Add actions to ensemble buffers
                        if step_count % self.tcp_action_update_interval == 0:
                            if self.use_latent_action_with_rnn_decoder:
                                tcp_action = action_all[self.latency_step:, ...]
                            else:
                                if action_all.shape[-1] == 4:
                                    tcp_action = action_all[self.latency_step:, :3]
                                elif action_all.shape[-1] == 8:
                                    tcp_action = action_all[self.latency_step:, :6]
                                elif action_all.shape[-1] == 10:
                                    tcp_action = action_all[self.latency_step:, :9]
                                elif action_all.shape[-1] == 20:
                                    tcp_action = action_all[self.latency_step:, :18]
                                else:
                                    raise NotImplementedError
                            # Add to ensemble buffer
                            logger.debug(f"step_count: {step_count}, Add TCP action to ensemble buffer, shape: {tcp_action.shape}, self.action_step_count: {self.action_step_count}")
                            # self.tcp_ensemble_buffer.add_action(tcp_action, step_count)
                            # logger.debug(f"self.action_step_count: {self.action_step_count}")
                            self.tcp_ensemble_buffer.add_action(tcp_action, self.action_step_count)
                            
                        
                        if step_count % self.gripper_action_update_interval == 0:
                            if self.use_latent_action_with_rnn_decoder:
                                gripper_action = action_all[self.gripper_latency_step:, ...]
                            else:
                                # Extract gripper part of the action
                                if action_all.shape[-1] == 4:
                                    gripper_action = action_all[self.gripper_latency_step:, 3:]
                                elif action_all.shape[-1] == 8:
                                    gripper_action = action_all[self.gripper_latency_step:, 6:]
                                elif action_all.shape[-1] == 10:
                                    gripper_action = action_all[self.gripper_latency_step:, 9:]
                                elif action_all.shape[-1] == 20:
                                    gripper_action = action_all[self.gripper_latency_step:, 18:]
                                else:
                                    raise NotImplementedError
                            
                            # Add to ensemble buffer
                            logger.debug(f"step_count: {step_count}, Add gripper action to ensemble buffer, shape: {gripper_action.shape}, self.action_step_count: {self.action_step_count}")
                            # self.gripper_ensemble_buffer.add_action(gripper_action, step_count)
                            self.gripper_ensemble_buffer.add_action(gripper_action, self.action_step_count)
                        
                        # Record frame if video recording is enabled
                        if self.enable_video_recording:
                            frame = self.env.render(mode="rgb_array", height=240, width=320)
                            frames.append(frame)
                        
                        # Sleep to maintain inference rate
                        cur_time = time.time()
                        precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        step_count += steps_per_inference
                
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                finally:
                    # Stop action thread
                    self.stop_event.set()
                    action_thread.join()
                    
                    # Get final episode stats with lock
                    with self.episode_stats_lock:
                        final_reward = self.episode_reward
                        final_length = self.episode_length
                    
                    # Save video if enabled
                    if self.enable_video_recording and frames:
                        self.save_video(frames, episode_idx)
                    
                    # Log episode results
                    logger.info(f"Episode {episode_idx} finished with return {final_reward} and length {final_length}")
                    all_returns.append(final_reward)
                    all_ep_lengths.append(final_length)
            
            # Return average performance
            avg_return = np.mean(all_returns)
            avg_length = np.mean(all_ep_lengths)
            logger.info(f"Evaluation complete. Average return: {avg_return}, Average episode length: {avg_length}")
            return avg_return, avg_length
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise 