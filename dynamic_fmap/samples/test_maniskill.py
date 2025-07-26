import gymnasium as gym
# import mani_skill.envs
import dynamic_fmap.benchmarks.maniskill


env = gym.make(
    "PickCube-v1f", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgb+depth", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    obs2 = env.get_obs(unflattened=True)
    done = terminated or truncated
    env.render()  # a display is required to render
env.close()