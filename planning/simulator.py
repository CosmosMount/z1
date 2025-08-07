from isaacgym import gymapi
from isaacgym import gymutil

import os
import time
import math
import json
import socket
import threading
import numpy as np

import pinocchio as pin
from numpy.linalg import norm,solve
from scipy.spatial.transform import Rotation as R


class z1_simulator:

    def __init__(self, host='127.0.0.1', port=9999):
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_env()
        self.create_viewer()

        self.build_ground()
        # self.build_objects()

        self.initialize_arm()
        self.initialize_events()

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((host, port))
        self.running = True
        self.listener_thread = threading.Thread(target=self.listen_cmd, daemon=True)
        self.listener_thread.start()

        self.moving_to_target = False
        self.target_reached = False
        self.steps_count = 0
    
    def send_sim_state(self, state):
        try:
            msg = json.dumps({"type": "state", "data": state}) + '\n'
            self.conn.sendall(msg.encode('utf-8'))
            # print("[IsaacSim] Sent message:", msg)
        except Exception as e:
            print("[IsaacSim] Failed to send message:", e)

    def listen_cmd(self):
        buffer = ""
        while self.running:
            recv = self.conn.recv(1024).decode('utf-8')
            buffer += recv
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                msg = json.loads(line)
                if msg["type"] == "cmd":
                    print("[IsaacSim] Received command:", msg["data"])

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

    def create_env(self):
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)

        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

    def build_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
    
    # def build_objects(self):
    #     # Load table assets
    #     table_dims = gymapi.Vec3(0.4, 0.4, 0.3)  # 长、宽、高
    #     asset_options = gymapi.AssetOptions()
    #     asset_options.fix_base_link = True
    #     table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    #     # Create the first table
    #     table1_pose = gymapi.Transform()
    #     table1_pose.p = gymapi.Vec3(0.0, 0.5, table_dims.z/2)
    #     self.gym.create_actor(self.env, table_asset, table1_pose, "table1", 0, 0)

    #     # Create the second table
    #     table2_pose = gymapi.Transform()
    #     table2_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z/2)
    #     self.gym.create_actor(self.env, table_asset, table2_pose, "table2", 0, 0)

    #     # Load block assets
    #     block_dims = gymapi.Vec3(0.05, 0.02, 0.1)  # 立方体
    #     asset_options = gymapi.AssetOptions()
    #     block_asset = self.gym.create_box(self.sim, block_dims.x, block_dims.y, block_dims.z, asset_options)

    #     # Place the block on the table
    #     block_pose = gymapi.Transform()
    #     block_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + block_dims.z/2)
    #     block_actor = self.gym.create_actor(self.env, block_asset, block_pose, "block", 0, 0)

    #     # Set block physical properties
    #     props = self.gym.get_actor_rigid_body_properties(self.env, block_actor)
    #     props[0].mass = 0.1  # Mass
    #     self.gym.set_actor_rigid_body_properties(self.env, block_actor, props)

    #     props = self.gym.get_actor_rigid_shape_properties(self.env, block_actor)
    #     props[0].friction = 10.0
    #     self.gym.set_actor_rigid_shape_properties(self.env, block_actor, props)

    #     block_transform = self.gym.get_rigid_transform(self.env, block_actor)
    #     print(f"Block position: {block_transform.p.x - block_dims.x/2}, {block_transform.p.y - block_dims.y/2}, {block_transform.p.z - block_dims.z/2}")


    #     ball_density = 1.0
    #     ball_radius = 0.02

    #     # Create a sphere asset for the ball
    #     asset_options = gymapi.AssetOptions()
    #     asset_options.density = 0.5
    #     asset_options.fix_base_link = False  # Movable
    #     ball_asset = self.gym.create_sphere(self.sim, ball_radius, asset_options)

    #     ball_pose = gymapi.Transform()
    #     ball_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + ball_radius)

    #     ball_actor = self.gym.create_actor(self.env, ball_asset, ball_pose, "ball", 0, 0)

    #     props = self.gym.get_actor_rigid_body_properties(self.env, ball_actor)
    #     props[0].mass = (4/3) * 3.14159 * ball_radius**3 * ball_density
    #     self.gym.set_actor_rigid_body_properties(self.env, ball_actor, props)
    
    def initialize_arm(self):
        asset_root = "../assets/z1/urdf"
        asset_file = "z1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.disable_gravity = False
        asset_options.armature = 0.01
        asset_options.use_mesh_materials = True  
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
        asset_options.override_com = True 
        asset_options.override_inertia = True 
        asset_options.vhacd_enabled = True 
        asset_options.vhacd_params = gymapi.VhacdParams() 
        asset_options.vhacd_params.resolution = 300000 

        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

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

        urdf_path = os.path.join(asset_root, asset_file)
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data  = self.pin_model.createData()

        # Initialize Pinocchio model and pose
        self.q_pin = pin.neutral(self.pin_model)
        self.q_home = pin.neutral(self.pin_model)

        # transform = self.gym.get_rigid_transform(self.env, 7)
        # T_base_to_link06 = np.eye(4)
        # T_base_to_link06[:3, :3] = R.from_quat([transform.r.x, transform.r.y, transform.r.z, transform.r.w]).as_matrix()
        # T_base_to_link06[:3, 3] = np.array([transform.p.x, transform.p.y, transform.p.z])
        # print(f"Base to Link06 transform: {T_base_to_link06}")
        # print(f"Initial position: {transform.p.x}, {transform.p.y}, {transform.p.z} Initial rotation: {transform.r.x}, {transform.r.y}, {transform.r.z}, {transform.r.w}")
        # self.target_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)

        
    def initialize_events(self):

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "input_coords")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "show_coords")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_all")
        
        for i in range(1, 8):
            self.gym.subscribe_viewer_keyboard_event(self.viewer, getattr(gymapi, f"KEY_{i}"), f"joint_{i}")

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT_SHIFT, "shift")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT_SHIFT, "shift")

        self.key_states = {
            "joint_1": False, "joint_2": False, "joint_3": False,
            "joint_4": False, "joint_5": False, "joint_6": False,
            "joint_7": False, "shift": False
        }
    
    def step(self):

        if self.gym.query_viewer_has_closed(self.viewer):
            print("Viewer closed, exiting...")
            self.end()
            exit(0)
        
        
        events = self.gym.query_viewer_action_events(self.viewer)
        for event in events:
            
            if event.action == "show_coords" and event.value > 0:
                # transform = self.gym.get_rigid_transform(self.env, 7)
                # current_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
                # print(f"Current position: {current_position}")
                # print(transform)
                

                # Forward kinematics
                
                print(f"Current pose:{self.pin_data.oMf[7]}")
                # print(f"Current DOF states: {dof_state}")

            if event.action == "reset_all":
                self.dof_targets[:6] = self.q_home[:6]

        for event in events:
            if event.action in self.key_states:
                self.key_states[event.action] = event.value > 0

        for i in range(7):
            action_key = f"joint_{i+1}"
            if self.key_states[action_key]:
                direction = -1 if self.key_states["shift"] else 1
                if i == 0:
                    self.dof_targets[i] += direction * 0.05
                else:
                    self.dof_targets[i] += direction * 0.005
            


        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)
        transform = self.gym.get_rigid_transform(self.env, 8)
        state = {
            "position": [transform.p.x, transform.p.y, transform.p.z],
            "rotation": [transform.r.x, transform.r.y, transform.r.z, transform.r.w]
        }
        
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        self.send_sim_state(state)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    

if __name__ == '__main__':
    simulator = z1_simulator()
    try:
        while True:
            simulator.step()
    except KeyboardInterrupt:
        simulator.end()
        exit(0)


