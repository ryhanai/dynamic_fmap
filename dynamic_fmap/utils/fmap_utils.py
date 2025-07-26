from functools import reduce
import transforms3d as tf
import re
import numpy as np
import sapien
from dynamic_fmap.utils.viewer import Viewer


class FmapUtils:
    def __init__(self, env):
        self._env = env    
        self._index = 0
        self._viewer = Viewer()

    def get_contact_force(self):
        dt = 1 / self._env.base_env.sim_config.sim_freq

        def get_point_forces(c):
            return [(p.position, p.impulse / dt, p.normal) for p in c.points]

        cs = self._env.base_env.scene.get_contacts()
        return reduce(lambda x, y: x + y, [get_point_forces(c) for c in cs])

    def add_force_vector(self, pose=([0, 0.4, 0.3], [1, 0, 0, 0]), half_length=0.3, index=0, color=[0, 0, 1]):
        builder = self._env.base_env.scene.create_actor_builder()
        p = sapien.Pose(*pose)
        builder.set_initial_pose(p)
        builder.add_cylinder_visual(radius=0.005, half_length=half_length, material=color)
        name = f'force_vector{index}'
        c = builder.build_static(name=name)
        # c.set_pose(p)

    def draw_point_forces(self, point_forces, scale=0.02):
        # remove existing force vectors
        for actor in self._env.base_env.scene.get_all_actors():
            if re.match('.*force_vector.*', actor.name) != None:
                actor.remove_from_scene()

        for position, force, normal in point_forces:
            force_mag = np.linalg.norm(force)
            if force_mag < 1e-5:
                continue

            # print(force)
            force_direction = force / force_mag
            xaxis = np.array([1, 0, 0])
            rotvec = np.cross(xaxis, force_direction)
            axis = rotvec / np.linalg.norm(rotvec)
            angle = np.arccos(np.dot(xaxis, force_direction))
            quat = tf.quaternions.mat2quat(tf.axangles.axangle2mat(axis, angle))
            self.add_force_vector(pose=[position, quat], half_length=scale*force_mag, index=self._index)
            self._index += 1

    def update(self, obs):
        print(obs.keys())
        print(obs['sensor_data']['base_camera'].keys())
        point_forces = self.get_contact_force()
        self.draw_point_forces(point_forces)  # draw point forces in the simulator
        self._viewer.update(obs, point_forces)  # update 3D viewer
