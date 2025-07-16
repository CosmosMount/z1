from isaacgym import gymapi
from isaacgym import gymutil

import os
import time
import math
import numpy as np

import pinocchio as pin
from numpy.linalg import norm,solve



class z1_Simulator:

    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.create_env()
        self.create_viewer()

        self.build_ground()
        # self.build_objects()

        self.initialize_arm()
        self.initialize_events()

        self.moving_to_target = False
        self.target_reached = False
        self.steps_count = 0

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
    
    def build_objects(self):
        # Load table assets
        table_dims = gymapi.Vec3(0.4, 0.4, 0.3)  # 长、宽、高
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # Create the first table
        table1_pose = gymapi.Transform()
        table1_pose.p = gymapi.Vec3(0.0, 0.5, table_dims.z/2)
        self.gym.create_actor(self.env, table_asset, table1_pose, "table1", 0, 0)

        # Create the second table
        table2_pose = gymapi.Transform()
        table2_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z/2)
        self.gym.create_actor(self.env, table_asset, table2_pose, "table2", 0, 0)

        # Load block assets
        block_dims = gymapi.Vec3(0.05, 0.02, 0.1)  # 立方体
        asset_options = gymapi.AssetOptions()
        block_asset = self.gym.create_box(self.sim, block_dims.x, block_dims.y, block_dims.z, asset_options)

        # Place the block on the table
        block_pose = gymapi.Transform()
        block_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + block_dims.z/2)
        block_actor = self.gym.create_actor(self.env, block_asset, block_pose, "block", 0, 0)

        # Set block physical properties
        props = self.gym.get_actor_rigid_body_properties(self.env, block_actor)
        props[0].mass = 0.1  # Mass
        self.gym.set_actor_rigid_body_properties(self.env, block_actor, props)

        props = self.gym.get_actor_rigid_shape_properties(self.env, block_actor)
        props[0].friction = 10.0
        self.gym.set_actor_rigid_shape_properties(self.env, block_actor, props)

        block_transform = self.gym.get_rigid_transform(self.env, block_actor)
        print(f"Block position: {block_transform.p.x - block_dims.x/2}, {block_transform.p.y - block_dims.y/2}, {block_transform.p.z - block_dims.z/2}")


        ball_density = 1.0
        ball_radius = 0.02

        # Create a sphere asset for the ball
        asset_options = gymapi.AssetOptions()
        asset_options.density = 0.5
        asset_options.fix_base_link = False  # Movable
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, asset_options)

        ball_pose = gymapi.Transform()
        ball_pose.p = gymapi.Vec3(0.5, 0.0, table_dims.z + ball_radius)

        ball_actor = self.gym.create_actor(self.env, ball_asset, ball_pose, "ball", 0, 0)

        props = self.gym.get_actor_rigid_body_properties(self.env, ball_actor)
        props[0].mass = (4/3) * 3.14159 * ball_radius**3 * ball_density
        self.gym.set_actor_rigid_body_properties(self.env, ball_actor, props)
    
    def initialize_arm(self):
        asset_root = "../z1_simulator/z1/urdf"
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

        transform = self.gym.get_rigid_transform(self.env, 7)
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
            
            if event.action == "show_coords" and event.value > 0:
                transform = self.gym.get_rigid_transform(self.env, 7)
                current_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
                print(f"Current position: {current_position}")

            if event.action == "reset_all":
                self.dof_targets[:6] = self.q_home[:6]
                self.q_pin[:] = self.q_home

        if self.moving_to_target and not self.target_reached:

            self.steps_count += 1

            transform = self.gym.get_rigid_transform(self.env, 7)
            current_position = np.array([transform.p.x, transform.p.y, transform.p.z], dtype=np.float32)
            print(f"Step {self.steps_count}  Current position: {current_position}, Target position: {self.target_position}")
            distance = np.linalg.norm(current_position - self.target_position)

            if distance < 0.05 or self.steps_count > 500:

                self.target_reached = True
                self.moving_to_target = False
                print("Target position reached." if distance < 0.05 else "Steps limit exceeded, target cannot be reached.")
                if self.steps_count > 500:
                    self.dof_targets[:6] = self.q_home[:6]
                    self.q_pin[:] = self.q_home


            else:

                self.gym.fetch_results(self.sim, True)
                q = self.inverse_kinematics(self.target_position)
                self.dof_targets[:5] = q[:5].astype(np.float32)  # Only use the first 5 joints for IK

        else:

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

        self.dof_targets = np.clip(self.dof_targets, self.lower_limits, self.upper_limits)
        self.gym.set_actor_dof_position_targets(self.env, self.actor, self.dof_targets)
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
    
    def inverse_kinematics(self, target_pos):

        JOINT_ID = 6
        oMdes = pin.SE3(np.eye(3), target_pos)

        q = self.q_pin.copy()
        eps = 1e-4
        IT_MAX = 1000
        DT = 1e-4
        damp = 1e-12

        for _ in range(IT_MAX):
            # Forward kinematics to compute the current wrist pose
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            # Calculate the current wrist pose in the world frame
            iMd = self.pin_data.oMi[JOINT_ID].actInv(oMdes)
            # Calculate the error vector in the Lie algebra space
            err = pin.log(iMd).vector

            # Check if the error is within the acceptable range
            if norm(err) < eps:
                success = True
                break

            # Calculate the Jacobian matrix for the wrist joint
            J = pin.computeJointJacobian(self.pin_model, self.pin_data, q, JOINT_ID)
            # Calculate the Jacobian matrix in the Lie algebra space
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            # Use the damped least squares method to compute the joint velocity
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            # Update the joint positions using the computed velocity
            q = pin.integrate(self.pin_model, q, v * DT)

        self.q_pin = q
        return q

if __name__ == '__main__':
    simulator = z1_Simulator()
    try:
        while True:
            simulator.step()
    except KeyboardInterrupt:
        simulator.end()
        exit(0)


