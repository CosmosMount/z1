from isaacgym import gymapi
from isaacgym import gymutil
import os
import time
import math
import pygame
import numpy as np

class z1_Simulator:

    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_env()
        self.create_viewer()
        self.load_assets()
        self.build_ground()
        self.initialize_arm()
        self.initialize_keyboard()

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
        sim_params.physx.contact_offset = 0.002
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

        self.actor = self.gym.create_actor(self.env, self.asset, pose, "z1", 0, 1)

        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        print(f"Number of DOFs: {self.num_dofs}")
        dof_props = self.gym.get_actor_dof_properties(self.env, self.actor)
        self.lower_limits = dof_props['lower']
        self.upper_limits = dof_props['upper']

        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        self.dof_targets = dof_states['pos'].copy()

    def initialize_keyboard(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Z1 Arm Keyboard Control")

        self.delta_angle = 0.05
        self.clock = pygame.time.Clock()

        print("Use keys 1-6 to increase joint 1-6 angle, SHIFT+1-6 to decrease. ESC to exit.")
        
    def step_keyboard(self):
        self.gym.fetch_results(self.sim, True)

        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                print("Exiting...")
                self.end()
                exit(0)

        for i in range(self.num_dofs):
            if keys[pygame.K_1 + i]:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.dof_targets[i] -= self.delta_angle
                else:
                    self.dof_targets[i] += self.delta_angle

            self.dof_targets[i] = np.clip(self.dof_targets[i], self.lower_limits[i], self.upper_limits[i])

        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)

        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.clock.tick(60)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    simulator = z1_Simulator()

    try:
        while True:
            simulator.step_keyboard()
    except KeyboardInterrupt:
        simulator.end()
        exit(0)


