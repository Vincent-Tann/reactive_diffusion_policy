import pickle

data_path = "/home/txs/Code/tactile/reactive_diffusion_policy/reactive_diffusion_policy_dataset/dataset_mini/wipe_v1.5/seq_01.pkl"

timestamp_arrays = []
# Load the data
with open(data_path, 'rb') as f:
    data = pickle.load(f)
for step_idx, sensor_msg in enumerate(data.sensorMessages):
    # if DEBUG and step_idx <= 60:
    #     continue
    # total_count += 1
    # logger.info(f'Processing {step_idx}th sensor message in sequence {seq_idx}')
    # # TODO: add timestamp
    # obs_dict = data_processing_manager.convert_sensor_msg_to_obs_dict(sensor_msg)
    timestamp_arrays.append(sensor_msg.timestamp)
    

import matplotlib.pyplot as plt

plt.plot(timestamp_arrays)
plt.show()