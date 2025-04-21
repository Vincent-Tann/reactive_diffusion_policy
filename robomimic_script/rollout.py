import numpy as np
import robosuite as suite
import robomimic
import h5py
import json
from robomimic.utils.file_utils import EnvUtils
from datetime import datetime
import cv2
import os
import yaml
import torch
f_path = "/home/txs/Code/tactile/reactive_diffusion_policy/robosuite_dataset/square.hdf5"
f_in = h5py.File(f_path, 'r')
env_meta = json.loads(f_in["data"].attrs["env_args"])
f_in.close()
env_name = env_meta["env_name"]
env_kwargs = env_meta["env_kwargs"]

import pdb; pdb.set_trace()

env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_height=240,
        camera_width=320,
        reward_shaping=True,
    )

def save_video_helper(frames, path):
    # frames is a list of numpy arrays
    os.makedirs(path, exist_ok=True)       
        
    for i, episode_frames in enumerate(frames):
        video_path = os.path.join(path, f"video_ep{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        height, width, channels = episode_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for frame in episode_frames:
            frame = cv2.flip(frame, 0)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
            out.write(frame_bgr) 
        out.release()
    print(f"Video saved to {video_path}") 

def eval_tactil(num_episodes=100, save_video=False, save_path=None):
    from algo.CompNet.FlowMatching import PlannerTrainer,DataSampler
    from models.CompNet.planner import Planner
    from data.sim_dataset import SimDataset
    import torch
    from utils.data_utils import DataTransformer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_config = {
    "obs": {
        "camera_0": (3, 224, 224), 
        "camera_1": (3, 224, 224), 
        "wrench": (6,)
        },
    "actions": {
        "pose": (20,6), 
        "gripper": (20,1)
        },
    "robot_states": {
        "robot0_eef_pos": (3,), 
        "robot0_eef_quat": (4,)
        },
    }
    model = Planner(
        input_config=input_config,
    )
    model.load_state_dict(torch.load('./checkpoints/TactIL/square/model_200.pth',weights_only=True))
    model.to(device)
    model.eval()

    from algo.CompNet.FlowMatching import PlannerODE,PlannerInference

    ode = PlannerODE(model, device, guidance_scale=3.0)
    data_transformer = DataTransformer.from_json("square_TactIL_stats.json",
                                                device=device)
    planner_inference = PlannerInference(input_config,ode, 
                                        data_transformer, 
                                        device,
                                        num_steps=20,
                                        action_buffer_size=10)
    returns = []
    ep_lengths = []
    all_episode_frames = []
    max_ep_length = 200
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        episode_frames = []
        while not done and ep_length < max_ep_length:
            force = env.env.robots[0].ee_force
            torque = env.env.robots[0].ee_torque
            ft = np.concatenate([force, torque])
            
            input_dict = {
                "obs": {
                    "camera_0": obs["agentview_image"][None,:,:,:].transpose(0,3,1,2).copy(),
                    "camera_1": obs["robot0_eye_in_hand_image"][None,:,:,:].transpose(0,3,1,2).copy(),
                    "wrench": ft[None,:].copy(),
                },
                "robot_states": {
                    "robot0_eef_pos": obs["robot0_eef_pos"][None,:].copy(),
                    "robot0_eef_quat": obs["robot0_eef_quat"][None,:].copy(),
                },
            }
            
            action = planner_inference.act(input_dict).reshape(-1)
            action[-1] = -1 if action[-1] < 0 else 1
            obs, reward, done, info = env.step(action)
            
            if save_video:
                # episode_frames.append(obs['agentview_image'])
                episode_frames.append(env.render(mode="rgb_array",
                                                 height=224, width=224))
                
            ep_return += reward
            ep_length += 1
            
        returns.append(ep_return)
        ep_lengths.append(ep_length)
        
        if save_video:
            all_episode_frames.append(episode_frames)
    if save_video and save_path:
        save_video_helper(all_episode_frames, save_path)
    return np.mean(returns), np.mean(ep_lengths)

def eval_comp_act( num_episodes=100, save_video=False, save_path=None):
    ## initialize model
    from CompACT.libs.act.act.utils import get_normalizers
    from CompACT.libs.act.act.policy import ACTPolicy
    config_filepath = '/share/data/ripl/jjt/projects/tactile/TactIL/checkpoints/CompACT/sim_square_pos/config.yaml'
    checkpoint_path = '/share/data/ripl/jjt/projects/tactile/TactIL/checkpoints/CompACT/sim_square_pos/policy_epoch_6000_seed_1.ckpt'

    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
        task_config = config['task_parameters']
        policy_config = config['policy_parameters']

    dataset_config = {
        'action_space': policy_config['action_space'],
        'include_ft': policy_config['include_ft'],
        'episode_len': task_config['episode_len'],
        'camera_names': task_config['camera_names'],
        'real_robot': False,
    }
    stats_path = os.path.join(os.path.dirname(checkpoint_path), 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = np.load(f,allow_pickle=True)
    for key in stats.keys():
        if not isinstance(stats[key], np.ndarray):
            stats[key] = np.array(stats[key])
        
    action_dim = policy_config["action_dim"]
    policy_config = policy_config
    max_timesteps = task_config["episode_len"]
    temporal_agg = policy_config["temporal_agg"]
    
    # load policy and stats
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(checkpoint_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    
    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1  # higher -> slower
        num_queries = policy_config["num_queries"]
        
    returns = []
    ep_lengths = []
    all_episode_frames = []
    max_ep_length = 200
    for i in range(num_episodes):
        obs = env.reset()
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, action_dim]
            ).cuda()
        done = False
        ep_return = 0
        ep_length = 0
        episode_frames = []
        t = 0
        while not done and ep_length < max_ep_length:
            force = env.env.robots[0].ee_force
            torque = env.env.robots[0].ee_torque
            
            
            # Prepare Input data
            ## Prepare image data 
            all_camera_images = [
                obs["agentview_image"][None,:,:,:].transpose(0,3,1,2),
                obs["robot0_eye_in_hand_image"][None,:,:,:].transpose(0,3,1,2),
            ]
            all_camera_images = np.stack(all_camera_images, axis=1)
            all_camera_images = all_camera_images / 255.0
            
            image_data = torch.from_numpy(all_camera_images)
            image_data = image_data.cuda().float()
            
            ## Prepare state data
            state = np.concatenate([
                obs["robot0_eef_pos"][None,:],
                obs["robot0_eef_quat"][None,:],
            ], axis=-1)
            state = (state - stats["obs_mean"]) / stats["obs_std"]
            state = torch.from_numpy(state).cuda().float()
            
            
            ## Prepare ft data
            ft = np.concatenate([force, torque])
            ft = (ft - stats["ft_mean"]) / stats["ft_std"]
            ft = torch.from_numpy(ft[None,:]).cuda().float()
            
            if t% query_frequency == 0:
                with torch.inference_mode():
                    all_actions = policy(image=image_data, state=state, ft=ft)
            
            if temporal_agg:
                all_time_actions[[t], t: t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
            else:
                raw_action = all_actions[:, t % query_frequency]
            
            action = raw_action[0].cpu().numpy()
            action = stats['action_mean'] + stats['action_std'] * action
            action[-1] = -1 if action[-1] < 0 else 1
            
            obs, reward, done, info = env.step(action)
            
            if save_video:
                episode_frames.append(env.render(mode="rgb_array",
                                                 height=224, width=224))
            
            
            t += 1
            
        
            
            
            if save_video:
                # episode_frames.append(obs['agentview_image'])
                episode_frames.append(env.render(mode="rgb_array",
                                                 height=224, width=224))
                
            ep_return += reward
            ep_length += 1
            
        returns.append(ep_return)
        ep_lengths.append(ep_length)
        
        if save_video:
            all_episode_frames.append(episode_frames)
    
    # Save videos and/or frames
    if save_video and save_path:
        os.makedirs(save_path, exist_ok=True)       
        
        for i, episode_frames in enumerate(all_episode_frames):
            video_path = os.path.join(save_path, f"video_ep{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            
            height, width, channels = episode_frames[0].shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in episode_frames:
                frame = cv2.flip(frame, 0)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                out.write(frame_bgr)
                
            out.release()
            print(f"Video saved to {video_path}")       
            
    return np.mean(returns), np.mean(ep_lengths)



def eval_rdp( num_episodes=100, save_video=False, save_path=None):
    ## initialize model
    # config_filepath = '/share/data/ripl/jjt/projects/tactile/TactIL/checkpoints/CompACT/sim_square_pos/config.yaml'
    # checkpoint_path = '/share/data/ripl/jjt/projects/tactile/TactIL/checkpoints/CompACT/sim_square_pos/policy_epoch_6000_seed_1.ckpt'

    # with open(config_filepath, 'r') as f:
    #     config = yaml.safe_load(f)
    #     task_config = config['task_parameters']
    #     policy_config = config['policy_parameters']

    # dataset_config = {
    #     'action_space': policy_config['action_space'],
    #     'include_ft': policy_config['include_ft'],
    #     'episode_len': task_config['episode_len'],
    #     'camera_names': task_config['camera_names'],
    #     'real_robot': False,
    # }
    # stats_path = os.path.join(os.path.dirname(checkpoint_path), 'dataset_stats.pkl')
    # with open(stats_path, 'rb') as f:
    #     stats = np.load(f,allow_pickle=True)
    # for key in stats.keys():
    #     if not isinstance(stats[key], np.ndarray):
    #         stats[key] = np.array(stats[key])
        
    # action_dim = policy_config["action_dim"]
    # policy_config = policy_config
    # max_timesteps = task_config["episode_len"]
    # temporal_agg = policy_config["temporal_agg"]
    
    # # load policy and stats
    # policy = ACTPolicy(policy_config)
    # loading_status = policy.load_state_dict(torch.load(checkpoint_path))
    # print(loading_status)
    # policy.cuda()
    # policy.eval()
    
    # query_frequency = policy_config["num_queries"]
    # if temporal_agg:
    #     query_frequency = 1  # higher -> slower
    #     num_queries = policy_config["num_queries"]
        
    returns = []
    ep_lengths = []
    all_episode_frames = []
    max_ep_length = 200
    for i in range(num_episodes):
        obs = env.reset()
        # if temporal_agg:
        #     all_time_actions = torch.zeros(
        #         [max_timesteps, max_timesteps + num_queries, action_dim]
        #     ).cuda()
        done = False
        ep_return = 0
        ep_length = 0
        episode_frames = []
        t = 0
        while not done and ep_length < max_ep_length:
            force = env.env.robots[0].ee_force
            torque = env.env.robots[0].ee_torque
            
            
            # Prepare Input data
            ## Prepare image data 
            all_camera_images = [
                obs["agentview_image"][None,:,:,:].transpose(0,3,1,2),
                obs["robot0_eye_in_hand_image"][None,:,:,:].transpose(0,3,1,2),
            ]
            all_camera_images = np.stack(all_camera_images, axis=1)
            all_camera_images = all_camera_images / 255.0
            
            image_data = torch.from_numpy(all_camera_images)
            image_data = image_data.cuda().float()
            
            ## Prepare state data
            state = np.concatenate([
                obs["robot0_eef_pos"][None,:],
                obs["robot0_eef_quat"][None,:],
            ], axis=-1)
            # state = (state - stats["obs_mean"]) / stats["obs_std"]
            # state = torch.from_numpy(state).cuda().float()
            
            
            # ## Prepare ft data
            # ft = np.concatenate([force, torque])
            # ft = (ft - stats["ft_mean"]) / stats["ft_std"]
            # ft = torch.from_numpy(ft[None,:]).cuda().float()
            
            # if t% query_frequency == 0:
            #     with torch.inference_mode():
            #         all_actions = policy(image=image_data, state=state, ft=ft)
            
            # if temporal_agg:
            #     all_time_actions[[t], t: t + num_queries] = all_actions
            #     actions_for_curr_step = all_time_actions[:, t]
            #     actions_populated = torch.all(
            #                 actions_for_curr_step != 0, axis=1
            #             )
            #     actions_for_curr_step = actions_for_curr_step[actions_populated]
            #     k = 0.01
            #     exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            #     exp_weights = exp_weights / exp_weights.sum()
            #     exp_weights = (
            #                 torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            #             )
            #     raw_action = (actions_for_curr_step * exp_weights).sum(
            #                 dim=0, keepdim=True
            #             )
            # else:
            #     raw_action = all_actions[:, t % query_frequency]
            
            # action = raw_action[0].cpu().numpy()
            # action = stats['action_mean'] + stats['action_std'] * action
            # action[-1] = -1 if action[-1] < 0 else 1
            
            # use random action
            action = np.random.randn(7)
            
            obs, reward, done, info = env.step(action)
            
            if save_video:
                episode_frames.append(env.render(mode="rgb_array",
                                                 height=224, width=224))
            t += 1
            
            if save_video:
                # episode_frames.append(obs['agentview_image'])
                episode_frames.append(env.render(mode="rgb_array",
                                                 height=224, width=224))
                
            ep_return += reward
            ep_length += 1
            
        returns.append(ep_return)
        ep_lengths.append(ep_length)
        
        if save_video:
            all_episode_frames.append(episode_frames)
    
    # Save videos and/or frames
    if save_video and save_path:
        os.makedirs(save_path, exist_ok=True)       
        
        for i, episode_frames in enumerate(all_episode_frames):
            video_path = os.path.join(save_path, f"video_ep{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            
            height, width, channels = episode_frames[0].shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in episode_frames:
                frame = cv2.flip(frame, 0)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                out.write(frame_bgr)
                
            out.release()
            print(f"Video saved to {video_path}")       
            
    return np.mean(returns), np.mean(ep_lengths)

if __name__ == "__main__":
    # returns, ep_lengths = eval_tactil(
    #     num_episodes=10,
    #     save_video=True,
    #     save_path="output/videos/tactil"
    # )
    # print(f"Average return: {returns}, Average episode length: {ep_lengths}")  
    
    # returns, ep_lengths = eval_comp_act(
    #     num_episodes=5,
    #     save_video=True,
    #     save_path="output/videos/comp_act"
    # )
    
    returns, ep_lengths = eval_rdp(
        num_episodes=1,
        save_video=True,
        save_path="output/videos/rdp"
    )
    print(f"Average return: {returns}, Average episode length: {ep_lengths}")  
        
        
        
        