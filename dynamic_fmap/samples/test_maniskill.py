# from requests import get
import gymnasium as gym
# import mani_skill.envs
import dynamic_fmap.benchmarks.maniskill
from force_estimation import force_distribution_viewer
from pathlib import Path
import numpy as np
import copy

class ManiskillRVizViewer:
    def __init__(self, env, force_smoothing=True):
        self._env = env
        self._viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
        self._force_smoothing = force_smoothing

    def load_robot(self):
        a = self._env.base_env.agent
        urdf_file_name = Path(a.urdf_path).name
        self._viewer.load_urdf(urdf_file_name)
        self._joint_names = copy.copy(a.arm_joint_names)
        if hasattr(a, "gripper_joint_names"):
            self._joint_names += a.gripper_joint_names
        self._viewer.set_static_transform(translation=a.robot.pose.get_p()[0].numpy().astype(float).tolist(),
                                          child_frame='panda_link0')

    def update(self):
        scene = self._env.base_env.scene
        robot = self._env.base_env.agent.robot
        qpos = robot.qpos # q_pos = torch.Tensor[9], 7 DoF for arm, 2 DoF for gripper
        self._viewer.set_joint_positions(self._joint_names, qpos[0].numpy())

        self._viewer.clear()
        for actor in scene.actors.values():
            self._publish_actor_state(actor)

        obs = self._env.base_env.get_obs()
        pfs = obs['sensor_data']['force_camera']['point_forces']
        poss = pfs[:, :3]
        forces = pfs[:, 3:6]
        normals = pfs[:, 6:]

        self._viewer.draw_vector_field(poss, np.log(forces+1.), scale=0.2, frame_id="map")

        if self._force_smoothing:
            pass

        self._viewer.show()

    def _publish_actor_state(self, actor):
        # URDF file names cannot be obtained from Sapien, but Maniskill hold the information
        # env.base_env.agent.urdf_path
        # /home/ryo/miniconda3/envs/maniskill_ros2/lib/python3.10/site-packages/mani_skill/assets/robots/panda/panda_v2.urdf

        try:
            if actor.has_collision_shapes:
                meshes = actor.get_collision_meshes()
                for m in meshes:
                    # colors = m.visual.vertex_colors / 255.
                    colors = np.tile([[0.1, 0.1, 0.3, 1.]],(len(m.vertices), 1))  # Due to the restriction of Rviz2, alpha < 1 is not drawn.
                    self._viewer.draw_mesh(m.vertices, m.faces, colors, frame_id="map")
        except Exception as e:
            print(e)
            # ground object has has_collision_shapes property but no method causes error


# env = gym.make(
#     "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
#     # "PegInsertionSide-v1",
#     num_envs=1,
#     obs_mode="rgb+depth", # there is also "state_dict", "rgbd", ...
#     control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
#     render_mode="human"
# )
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)

# obs, _ = env.reset(seed=0) # reset with a seed for determinism
# viewer = ManiskillRVizViewer(env)


def main():    
    viewer.load_robot()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.update()
        obs2 = env.get_obs(unflattened=True)
        done = terminated or truncated
        env.render()  # a display is required to render

    # env.close()


# sed -i 's/franka_description/package:\/\/force_estimation\/robots\/panda\/franka_description/g' panda_v2.urdf