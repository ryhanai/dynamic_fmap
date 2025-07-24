import gymnasium as gym
from dynamic_fmap.force_label.fmap_utils import FmapUtils

# visualize smoothed force
# load button-pushing scenario

env = gym.make(
    "PushCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="pointcloud", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

fmap_utils = FmapUtils(env)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    fmap_utils.update(obs=obs)
    env.render()  # a display is required to render
# env.close()
