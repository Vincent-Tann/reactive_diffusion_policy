import h5py
import zarr
import numpy as np
import os
import cv2
import argparse
from loguru import logger
import py_cli_interaction
from omegaconf import DictConfig

def resize_image_by_size(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize an image to the specified size."""
    return cv2.resize(image, size)

def convert_hdf5_to_zarr(
    hdf5_path, 
    zarr_path, 
    image_resize_shape=(320, 240),
    temporal_downsample_ratio=1,
    use_6d_rotation=True,
    debug=False
):
    """
    Convert HDF5 robosuite dataset to zarr format compatible with reactive diffusion policy.
    
    Args:
        hdf5_path: Path to the HDF5 file
        zarr_path: Path to save the zarr dataset
        image_resize_shape: Tuple of (width, height) for resizing images
        temporal_downsample_ratio: Ratio for temporal downsampling
        use_6d_rotation: Whether to use 6D rotation representation
        debug: Whether to print debug information
    """
    # Check if zarr file already exists
    if os.path.exists(zarr_path):
        logger.info(f'Data already exists at {zarr_path}')
        if py_cli_interaction.parse_cli_bool('Do you want to overwrite the data?', default_value=True):
            logger.warning(f'Overwriting {zarr_path}')
            os.system(f'rm -rf {zarr_path}')
        else:
            logger.info('Exiting without overwriting')
            return

    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        # Get all demo keys
        demo_keys = [key for key in hdf5_file['data'].keys() if key.startswith('demo_')]
        logger.info(f'Found {len(demo_keys)} demos in the HDF5 file')
        
        # Initialize arrays to store data
        timestamp_arrays = []
        external_img_arrays = []  # agentview_image
        left_wrist_img_arrays = []  # robot0_eye_in_hand_image
        
        # Robot state arrays
        left_robot_tcp_pose_arrays = []  # robot0_eef_pos + robot0_eef_quat
        left_robot_tcp_vel_arrays = []  # robot0_eef_vel_lin + robot0_eef_vel_ang
        left_robot_tcp_wrench_arrays = []  # ft
        left_robot_gripper_width_arrays = []  # from actions[:, 6]
        left_robot_gripper_force_arrays = []  # placeholder
        
        # For compatibility with the original format
        right_robot_tcp_pose_arrays = []
        right_robot_tcp_vel_arrays = []
        right_robot_tcp_wrench_arrays = []
        right_robot_gripper_width_arrays = []
        right_robot_gripper_force_arrays = []
        
        episode_ends_arrays = []
        total_count = 0
        
        # Process each demo
        for demo_idx, demo_key in enumerate(demo_keys):
            logger.info(f'Processing demo {demo_idx+1}/{len(demo_keys)}: {demo_key}')
            demo_group = hdf5_file['data'][demo_key]
            
            # Get the length of the demo
            demo_length = demo_group['actions'].shape[0]
            
            # Process each timestep
            for step_idx in range(demo_length):
                if debug and step_idx % 10 == 0:
                    logger.debug(f'Processing step {step_idx}/{demo_length} in demo {demo_idx}')
                
                # Get observations
                obs = demo_group['obs']
                
                # Add timestamp (use step_idx as a placeholder)
                timestamp_arrays.append(float(step_idx))
                
                # Process images
                agentview_img = obs['agentview_image'][step_idx]
                eye_in_hand_img = obs['robot0_eye_in_hand_image'][step_idx]
                
                # Resize images
                agentview_img_resized = resize_image_by_size(agentview_img, image_resize_shape)
                eye_in_hand_img_resized = resize_image_by_size(eye_in_hand_img, image_resize_shape)
                
                external_img_arrays.append(agentview_img_resized)
                left_wrist_img_arrays.append(eye_in_hand_img_resized)
                
                # Process robot state
                eef_pos = obs['robot0_eef_pos'][step_idx]
                eef_quat = obs['robot0_eef_quat'][step_idx]

                # Convert to 9D pose if using 6D rotation
                if use_6d_rotation:
                    # Extract rotation matrix from quaternion
                    w, x, y, z = eef_quat
                    # First column of rotation matrix
                    r00 = 1 - 2*y*y - 2*z*z
                    r10 = 2*x*y + 2*w*z
                    r20 = 2*x*z - 2*w*y
                    # Second column of rotation matrix
                    r01 = 2*x*y - 2*w*z
                    r11 = 1 - 2*x*x - 2*z*z
                    r21 = 2*y*z + 2*w*x

                    # 9D pose: position (3) + first two columns of rotation matrix (6)
                    pose_9d = np.array([
                        eef_pos[0], eef_pos[1], eef_pos[2],
                        r00, r10, r20, r01, r11, r21
                    ])
                    left_robot_tcp_pose_arrays.append(pose_9d)
                else:
                    # 7D pose: position (3) + quaternion (4)
                    pose_7d = np.concatenate([eef_pos, eef_quat])
                    left_robot_tcp_pose_arrays.append(pose_7d)
                
                # Process velocities
                vel_lin = obs['robot0_eef_vel_lin'][step_idx]
                vel_ang = obs['robot0_eef_vel_ang'][step_idx]
                left_robot_tcp_vel_arrays.append(np.concatenate([vel_lin, vel_ang]))
                
                # Process force-torque
                ft = obs['ft'][step_idx]
                left_robot_tcp_wrench_arrays.append(ft)
                
                # Process gripper
                if step_idx < demo_length:
                    # gripper_width = demo_group['actions'][step_idx, 6:7]  # Last dimension of action
                    gripper_width = obs['robot0_gripper_qpos'][step_idx, 0:1] - obs['robot0_gripper_qpos'][step_idx, 1:2]
                else:
                    gripper_width = np.array([0.0])  # Default value for last step
                
                left_robot_gripper_width_arrays.append(gripper_width)
                left_robot_gripper_force_arrays.append(np.array([0.0]))  # Placeholder
                
                # Add placeholder data for right robot (for compatibility)
                right_robot_tcp_pose_arrays.append(np.zeros(9) if use_6d_rotation else np.zeros(7))
                right_robot_tcp_vel_arrays.append(np.zeros(6))
                right_robot_tcp_wrench_arrays.append(np.zeros(6))
                right_robot_gripper_width_arrays.append(np.array([0.0]))
                right_robot_gripper_force_arrays.append(np.array([0.0]))
                
                total_count += 1
            
            # Record episode end
            episode_ends_arrays.append(total_count)
            logger.info(f'Finished processing demo {demo_idx+1}, total steps: {total_count}')
        
        # Convert lists to arrays
        timestamp_arrays = np.array(timestamp_arrays)
        external_img_arrays = np.stack(external_img_arrays, axis=0)
        left_wrist_img_arrays = np.stack(left_wrist_img_arrays, axis=0)
        
        left_robot_tcp_pose_arrays = np.stack(left_robot_tcp_pose_arrays, axis=0)
        left_robot_tcp_vel_arrays = np.stack(left_robot_tcp_vel_arrays, axis=0)
        left_robot_tcp_wrench_arrays = np.stack(left_robot_tcp_wrench_arrays, axis=0)
        left_robot_gripper_width_arrays = np.stack(left_robot_gripper_width_arrays, axis=0)
        left_robot_gripper_force_arrays = np.stack(left_robot_gripper_force_arrays, axis=0)
        
        right_robot_tcp_pose_arrays = np.stack(right_robot_tcp_pose_arrays, axis=0)
        right_robot_tcp_vel_arrays = np.stack(right_robot_tcp_vel_arrays, axis=0)
        right_robot_tcp_wrench_arrays = np.stack(right_robot_tcp_wrench_arrays, axis=0)
        right_robot_gripper_width_arrays = np.stack(right_robot_gripper_width_arrays, axis=0)
        right_robot_gripper_force_arrays = np.stack(right_robot_gripper_force_arrays, axis=0)
        

        
        episode_ends_arrays = np.array(episode_ends_arrays)
        
        # Create state and action arrays (similar to the original script)
        # For single arm: (left_tcp_x, left_tcp_y, left_tcp_z, left_gripper_width)
        state_arrays = np.concatenate([
            left_robot_tcp_pose_arrays[:, :9],  # Position + Rotation (3+6)
            left_robot_gripper_width_arrays
        ], axis=-1)
        
        # Action is the next state (absolute action)
        new_action_arrays = state_arrays[1:, ...].copy()
        action_arrays = np.concatenate([new_action_arrays, new_action_arrays[-1][np.newaxis, :]], axis=0)
        
        # Fix the last action of each episode
        for i in range(0, len(episode_ends_arrays)):
            start_idx = 0 if i == 0 else episode_ends_arrays[i-1]
            end_idx = episode_ends_arrays[i] - 1
            if end_idx > start_idx:  # Make sure there are at least 2 steps in the episode
                action_arrays[end_idx] = action_arrays[end_idx - 1]
        
        # Apply temporal downsampling if needed
        if temporal_downsample_ratio > 1:
            logger.info(f'Applying temporal downsampling with ratio {temporal_downsample_ratio}')
            
            # Calculate indices to keep after downsampling
            keep_indices = []
            current_episode_start = 0
            
            # Process each episode separately
            for episode_end in episode_ends_arrays:
                # Get indices for current episode
                episode_indices = np.arange(current_episode_start, episode_end)
                
                # Calculate downsampled indices for this episode
                # Keep first and last frame of each episode, downsample middle frames
                if len(episode_indices) > 2:
                    middle_indices = episode_indices[1:-1]
                    downsampled_middle_indices = middle_indices[::temporal_downsample_ratio]
                    episode_keep_indices = np.concatenate([
                        [episode_indices[0]],
                        downsampled_middle_indices,
                        [episode_indices[-1]]
                    ])
                else:
                    # If episode is too short, keep all frames
                    episode_keep_indices = episode_indices
                
                keep_indices.extend(episode_keep_indices)
                current_episode_start = episode_end
            
            keep_indices = np.array(keep_indices)
            
            # Downsample all arrays
            timestamp_arrays = timestamp_arrays[keep_indices]
            external_img_arrays = external_img_arrays[keep_indices]
            left_wrist_img_arrays = left_wrist_img_arrays[keep_indices]
            
            left_robot_tcp_pose_arrays = left_robot_tcp_pose_arrays[keep_indices]
            left_robot_tcp_vel_arrays = left_robot_tcp_vel_arrays[keep_indices]
            left_robot_tcp_wrench_arrays = left_robot_tcp_wrench_arrays[keep_indices]
            left_robot_gripper_width_arrays = left_robot_gripper_width_arrays[keep_indices]
            left_robot_gripper_force_arrays = left_robot_gripper_force_arrays[keep_indices]
            
            right_robot_tcp_pose_arrays = right_robot_tcp_pose_arrays[keep_indices]
            right_robot_tcp_vel_arrays = right_robot_tcp_vel_arrays[keep_indices]
            right_robot_tcp_wrench_arrays = right_robot_tcp_wrench_arrays[keep_indices]
            right_robot_gripper_width_arrays = right_robot_gripper_width_arrays[keep_indices]
            right_robot_gripper_force_arrays = right_robot_gripper_force_arrays[keep_indices]
            
            action_arrays = action_arrays[keep_indices]
            state_arrays = state_arrays[keep_indices]
            
            # Recalculate episode_ends
            new_episode_ends = []
            count = 0
            current_episode_start = 0
            for episode_end in episode_ends_arrays:
                episode_indices = np.arange(current_episode_start, episode_end)
                if len(episode_indices) > 2:
                    middle_indices = episode_indices[1:-1]
                    downsampled_middle_indices = middle_indices[::temporal_downsample_ratio]
                    count += len(downsampled_middle_indices) + 2  # +2 for first and last frame
                else:
                    count += len(episode_indices)
                new_episode_ends.append(count)
                current_episode_start = episode_end
            
            episode_ends_arrays = np.array(new_episode_ends)
        
        # Create zarr file
        logger.info(f'Creating zarr file at {zarr_path}')
        zarr_root = zarr.group(zarr_path)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        
        # Compute chunk sizes
        external_img_chunk_size = (100, external_img_arrays.shape[1], external_img_arrays.shape[2], external_img_arrays.shape[3])
        wrist_img_chunk_size = (100, left_wrist_img_arrays.shape[1], left_wrist_img_arrays.shape[2], left_wrist_img_arrays.shape[3])
        action_chunk_size = (10000, action_arrays.shape[1])
        
        # Use compression
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        
        # Store data in zarr format
        zarr_data.create_dataset('timestamp', data=timestamp_arrays, chunks=(10000,), dtype='float32', 
                                 compressor=compressor)
        
        zarr_data.create_dataset('left_robot_tcp_pose', data=left_robot_tcp_pose_arrays, 
                                 chunks=(10000, left_robot_tcp_pose_arrays.shape[1]), dtype='float32', 
                                 compressor=compressor)
        zarr_data.create_dataset('left_robot_tcp_vel', data=left_robot_tcp_vel_arrays, 
                                 chunks=(10000, 6), dtype='float32', compressor=compressor)
        zarr_data.create_dataset('left_robot_tcp_wrench', data=left_robot_tcp_wrench_arrays, 
                                 chunks=(10000, 6), dtype='float32', compressor=compressor)
        zarr_data.create_dataset('left_robot_gripper_width', data=left_robot_gripper_width_arrays, 
                                 chunks=(10000, 1), dtype='float32', compressor=compressor)
        zarr_data.create_dataset('left_robot_gripper_force', data=left_robot_gripper_force_arrays, 
                                 chunks=(10000, 1), dtype='float32', compressor=compressor)
        
        zarr_data.create_dataset('right_robot_tcp_pose', data=right_robot_tcp_pose_arrays, 
                                 chunks=(10000, right_robot_tcp_pose_arrays.shape[1]), dtype='float32', 
                                 compressor=compressor)
        zarr_data.create_dataset('right_robot_tcp_vel', data=right_robot_tcp_vel_arrays, 
                                 chunks=(10000, 6), dtype='float32', compressor=compressor)
        zarr_data.create_dataset('right_robot_tcp_wrench', data=right_robot_tcp_wrench_arrays, 
                                 chunks=(10000, 6), dtype='float32', compressor=compressor)
        zarr_data.create_dataset('right_robot_gripper_width', data=right_robot_gripper_width_arrays, 
                                 chunks=(10000, 1), dtype='float32', compressor=compressor)
        zarr_data.create_dataset('right_robot_gripper_force', data=right_robot_gripper_force_arrays, 
                                 chunks=(10000, 1), dtype='float32', compressor=compressor)
        
        zarr_data.create_dataset('target', data=state_arrays, chunks=action_chunk_size, 
                                 dtype='float32', compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, 
                                 dtype='float32', compressor=compressor)
        
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(10000,), 
                                 dtype='int64', compressor=compressor)
        
        zarr_data.create_dataset('external_img', data=external_img_arrays, chunks=external_img_chunk_size, 
                                 dtype='uint8', compressor=compressor)
        zarr_data.create_dataset('left_wrist_img', data=left_wrist_img_arrays, chunks=wrist_img_chunk_size, 
                                 dtype='uint8', compressor=compressor)
        
        # Print zarr data structure
        logger.info('Zarr data structure:')
        logger.info(zarr_data.tree())
        logger.info(f'Total count: {action_arrays.shape[0]}')
        logger.info(f'Data saved at {zarr_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HDF5 robosuite dataset to zarr format')
    parser.add_argument('--hdf5_path', type=str, default='robosuite_dataset/square.hdf5',
                        help='Path to the HDF5 file')
    parser.add_argument('--zarr_path', type=str, default='robosuite_dataset/square_zarr/replay_buffer.zarr',
                        help='Path to save the zarr dataset')
    parser.add_argument('--image_width', type=int, default=320,
                        help='Width for resizing images')
    parser.add_argument('--image_height', type=int, default=240,
                        help='Height for resizing images')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Temporal downsampling ratio')
    parser.add_argument('--use_6d_rotation', action='store_true', default=True,
                        help='Use 6D rotation representation')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.zarr_path), exist_ok=True)
    
    # Convert HDF5 to zarr
    convert_hdf5_to_zarr(
        args.hdf5_path,
        args.zarr_path,
        image_resize_shape=(args.image_width, args.image_height),
        temporal_downsample_ratio=args.downsample,
        use_6d_rotation=args.use_6d_rotation,
        debug=args.debug
    ) 