import numpy as np
import os
import collections
import matplotlib.pyplot as plt
import imageio
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.env_utils as EnvUtils
from collections import namedtuple


State = namedtuple("State", "observation reward")

class RobomimicWrapper: # For ACT
    def __init__(self, env):
        self.env = env
        self.max_timesteps = 300
        self.current_step = 0
        self.max_reward = 1

    def reset(self):
        state = self.env.reset()
        self.current_step = 0
        # qpos = np.concatenate((state['robot0_joint_pos'], state['robot1_joint_pos']))
        # obs = {
        #     'images': {
        #         'camera' : state['agentview_image']
        #     },
        #     'qpos':qpos
        # }
        # output = State(obs, 0)

        return state['agentview_image']

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        self.current_step += 1
        if self.current_step == self.max_timesteps:
            done = True
        # qpos = np.concatenate((state['robot0_joint_pos'], state['robot1_joint_pos']))
        # obs = {
        #     'images': {
        #         'camera' : state['agentview_image']
        #     },
        #     'qpos':qpos
        # }
        # output = State(obs, reward)
        # obs, reward, done, info info["all_completions_ids"]
        return state['agentview_image'], reward, done, {'image':state['agentview_image'], "all_completions_ids":reward}

def get_robomimic_env(dataset_dir):
    dataset_dir = f'{dataset_dir}/image_ds.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_dir)
    env_meta['env_kwargs']['camera_heights'] = 240
    env_meta['env_kwargs']['camera_widths'] = 360

    print(env_meta)
    
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=['agentview'], 
        camera_height=240, 
        camera_width=360,
        reward_shaping=False
    )
    
    return RobomimicWrapper(env)

if __name__ == "__main__": 
    env = get_robomimic_env('transport')
    # create a video writer

    print(env.step(np.random.randn(14,)))
