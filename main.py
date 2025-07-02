from isaacgym import gymapi
from isaacgym import gymutil

import os
import time
import math
import pygame
import numpy as np

from ikpy.chain import Chain


class z1_Simulator:

    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_env()
        self.create_viewer()
        self.load_assets()
        self.build_ground()
        self.initialize_arm()
        self.initialize_events()

        self.moving_to_target = False
        self.target_reached = False
        self.steps_count = 0

    def build_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        
    def create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
        look_at = gymapi.Vec3(0.0, 0.0, 0.0)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, look_at)

        # Set viewer window always on top
        window_id = os.popen("wmctrl -l | grep 'Isaac Gym' | awk '{print $1}'").read().strip()
        if window_id:
            os.system(f"wmctrl -i -r {window_id} -b add,above")

    def load_assets(self):
        asset_root = "../z1_simulator/z1/urdf"
        asset_file = "z1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.disable_gravity = False
        asset_options.armature = 0.01

        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def create_env(self):
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)

        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

    def initialize_arm(self):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0)
        pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

        self.actor = self.gym.create_actor(self.env, self.asset, pose, "z1", 0, -1)

        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        print(f"Number of DOFs: {self.num_dofs}")
        dof_props = self.gym.get_actor_dof_properties(self.env, self.actor)
        self.lower_limits = dof_props['lower']
        self.upper_limits = dof_props['upper']

        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        self.dof_targets = dof_states['pos'].copy()

        active_mask = [False, True, True, True, True, True, True, False, False]
        self.robot_chain = Chain.from_urdf_file("../z1_simulator/z1/urdf/z1.urdf", base_elements=['link00'], active_links_mask=active_mask)
        
    def initialize_events(self):

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "input_coords")
        
        for i in range(1, 8):
            self.gym.subscribe_viewer_keyboard_event(self.viewer, getattr(gymapi, f"KEY_{i}"), f"joint_{i}")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_SHIFT, "shift")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT_SHIFT, "shift")

        self.key_states = {
            "joint_1": False, "joint_2": False, "joint_3": False,
            "joint_4": False, "joint_5": False, "joint_6": False,
            "joint_7": False, "shift": False
        }

        transform = self.gym.get_rigid_transform(self.env, 8)
        self.target_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
    
    def step(self):

        if self.gym.query_viewer_has_closed(self.viewer):
            print("Viewer closed, exiting...")
            self.end()
            exit(0)
        
        events = self.gym.query_viewer_action_events(self.viewer)
        for event in events:
            if event.action == "input_coords" and event.value > 0:
                try:

                    user_input = input("Please input the target coordinate (x y z), split by spaces: ")
                    coords = [float(x) for x in user_input.split()]
                    
                    if len(coords) == 3:
                        self.target_position = np.array(coords,dtype=np.float32)
                        print(f"New target position: {self.target_position}")
                        self.moving_to_target = True
                        self.target_reached = False
                        self.steps_count = 0

                    else:
                        print("Error: Please input exactly three coordinates (x, y, z)")

                except ValueError:
                    print("Error: Invalid input. Please enter three numeric values separated by spaces.")

        if self.moving_to_target and not self.target_reached:

            self.steps_count += 1

            transform = self.gym.get_rigid_transform(self.env, 8)
            current_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
            print(f"Step {self.steps_count}  Current position: {current_position}, Target position: {self.target_position}")
            distance = np.linalg.norm(current_position - self.target_position)

            if distance < 0.05 or self.steps_count > 100:

                self.target_reached = True
                self.moving_to_target = False
                print("Target position reached." if distance < 0.05 else "Steps limit exceeded, target cannot be reached.")

            else:

                self.gym.fetch_results(self.sim, True)
                ik_solution = self.robot_chain.inverse_kinematics(self.target_position)
                self.dof_targets[:6] = np.array(ik_solution[1:7], dtype=np.float32)
                self.dof_targets = np.clip(self.dof_targets, self.lower_limits, self.upper_limits)
                self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)

        else:

            for event in events:
                if event.action in self.key_states:
                    self.key_states[event.action] = event.value > 0

            for i in range(7):
                action_key = f"joint_{i+1}"
                if self.key_states[action_key]:
                    direction = -1 if self.key_states["shift"] else 1
                    self.dof_targets[i] += direction * 0.05

        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    simulator = z1_Simulator()
    try:
        while True:
            simulator.step()
    except KeyboardInterrupt:
        simulator.end()
        exit(0)


