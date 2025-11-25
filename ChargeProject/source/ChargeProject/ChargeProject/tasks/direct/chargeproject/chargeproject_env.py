from __future__ import annotations
import math
import colorsys
from time import time
from enum import IntEnum

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCfg, patterns
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .map_manager import MapManager
from .spider_robot import SPIDER_JOINT_INFO
from .natural_terrain import MultiBiomeTerrainCfg

from .chargeproject_env_cfg import ChargeprojectEnvCfg

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

import inspect
import os


"""
states: 
    Patrol: Goes slowly between set patrol points
        When in patrol mode the env has an area where the robot can patrol
        There is a "safe" area where the AI should be
            when within x distance of the boundary it will be given a target position of the closest non-warning point
            when outside the safe area it will be penalized
        Transitions:
            Investigate: If hear loud noise go to investigate with location of noise
            Look: If it sees the player
    Investigate: Moves to a location to gather more information
        Transitions:
            Patrol: Time out on patrol
            Look: If it sees the player
    Look: If it sees the player for more than ~0.1 seconds, stop and look at them for a moment
        A counter that adds up when player is in sight and subtracts when not
            see goes up scaled by distance to player (closer = faster)
        Note: not given player position so it encourages not moving
        Transitions:
            Patrol: If counter goes to -X
            Attack: If counter goes to +X
    Attack: Engages the target
        Transitions:
            Investigate: If loses sight of target for long enough
            Hide: After hitting target
    Hide: Seeks cover from the target
        Transitions:
            Attack: After hiding for a set time
obs: 
    robot/leg/height_data
    One hot state encoding in obs
    Flag for if player is in LOS
    Target position (Player or investigation point) | i/a/h

actions:
    Joint position targets
rewards:
    General:
        Leg efficiency (minimize torque, accel, vel, action rate)
        z_vel_error
        ang_vel_error
        undesired contacts
        flat orientation
        hip joint deviation from neutral
        feet under body penalty
    Patrol:
        Velocity matching target patrol speed
        Penalize being outside safe area
        Reward for exploring new areas
    Investigate:
        Moving towards investigation point
        Penalize being outside safe area
        Reward for reaching investigation point
    Look:
        Reward for in LOS
        Penalize feet movement
    Attack:
        Reward for in LOS
        Reward for approaching player
        Reward for force of contact with player
        Flat Reward for impact
    Hide:
        Reward for staying out of sight
        Penalize movement while hiding
        Penalize contact force between feet and ground (to encourage quiet movement)
"""

# Enum for robot states
class RobotState(IntEnum):
    PATROL = 0
    INVESTIGATE = 1
    LOOK = 2
    ATTACK = 3
    HIDE = 4


class ChargeprojectEnv(DirectRLEnv):
    cfg: ChargeprojectEnvCfg

    def __init__(
        self, cfg: ChargeprojectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Player
        self._player_movement_angle = torch.zeros(self.num_envs, device=self.device)

        # num_envs, 6, 4
        self._actions = torch.zeros(
            self.num_envs, self.action_space.shape[1],# self.action_space.shape[2],
            device=self.device,
        )
        self._previous_actions = torch.zeros_like(self._actions)

        self._last_targets_reached = torch.zeros(self.num_envs, device=self.device)

        self.died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        # self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.base_contact_ids, _ = self._contact_sensor.find_bodies(self.cfg.base_name)
        self.base_body_ids, _ = self._robot.find_bodies(self.cfg.base_name)
        self.undesired_contact_ids, _ = self._contact_sensor.find_bodies(
            self.cfg.undesired_contact_body_names
        )
        self.lower_leg_body_ids, _ = self._robot.find_bodies(self.cfg.lower_leg_names)
        self.hip_joint_ids, _ = self._robot.find_joints(self.cfg.hip_joint_names)
        self.feet_body_ids, _ = self._robot.find_bodies(self.cfg.foot_names)
        self.hip_body_ids = []
        self.leg_joint_ids = []
        self.feet_contact_ids = []
        self.dof_idx = []
        for i in range(6):
            vals = []
            vals.append(self._robot.find_bodies(f".*hip_{i}")[0][0])
            vals.append(self._robot.find_bodies(f".*upper_{i}")[0][0])
            vals.append(self._robot.find_bodies(f".*middle_{i}")[0][0])
            vals.append(self._robot.find_bodies(f".*lower_{i}")[0][0])
            self.leg_joint_ids.append(vals)
            self.hip_body_ids.append(vals[0])
            self.feet_contact_ids.append(self._contact_sensor.find_bodies(f".*foot_{i}")[0][0])
            self.dof_idx.extend(vals)
        
        # Get limits and default positions from SPIDER_JOINT_INFO
        self.dof_min_limits = torch.tensor(list(SPIDER_JOINT_INFO["limit_min"].values()), device=self.device).repeat(6)
        self.dof_max_limits = torch.tensor(list(SPIDER_JOINT_INFO["limit_max"].values()), device=self.device).repeat(6)
        self.dof_default_pos = torch.tensor(list(SPIDER_JOINT_INFO["default_pos"].values()), device=self.device).repeat(6)

        # Pre-calculate the range of motion on either side of the default position
        self.positive_range = self.dof_max_limits - self.dof_default_pos
        self.negative_range = self.dof_default_pos - self.dof_min_limits
        
        self.feet_step_up_counters = self.cfg.feet_step_time_leeway * torch.ones(self.num_envs, len(self.feet_body_ids), device=self.device)
        self.feet_step_down_counters = self.cfg.feet_step_time_leeway * torch.ones(self.num_envs, len(self.feet_body_ids), device=self.device)


        self.avg_vel_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.vel_smoothing_alpha = 0.05 # ~20 steps

        # State of robot (patrol, attack, hide, search)
        self.robot_state = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # Different for each state
        self.state_timers = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._prev_total_staleness = torch.zeros(self.num_envs, device=self.device)

        # For storing data like last known player position or investigation point
        self.nav_targets = torch.zeros(self.num_envs, 3, device=self.device)
        
        self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        log_dir = self.cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.extras["log"] = dict()
        


        # Save env and config code for reproducibility
        current_file = inspect.getfile(inspect.currentframe())
        with open(os.path.join(log_dir, "env_code.py.txt"), "w") as f:
            with open(current_file, "r") as current_f:
                f.write(current_f.read())

        config_file = self.cfg._get_config_file_path()
        with open(os.path.join(log_dir, "env_config.py.txt"), "w") as f:
            with open(config_file, "r") as config_f:
                f.write(config_f.read())
                

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        #self._player = RigidObject(self.cfg.player)
        
        #self.scene.rigid_objects["player"] = self._player

        # we add a height scanner for perceptive locomotion
        self._lidar_sensor = RayCaster(self.cfg.lidar_scanner)
        self.scene.sensors["lidar_scanner"] = self._lidar_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # check if cameras exist
        if not hasattr(self.cfg, "cameras"):
            self.cfg.cameras = True
            print("No camera setting found in cfg, defaulting to cameras=True")


        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self._robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        terrain_gen = self.cfg.terrain.terrain_generator
        terrain_dims = (terrain_gen.num_rows * terrain_gen.size[0], terrain_gen.num_cols * terrain_gen.size[1])
        terrain_dims = (terrain_dims[0] + 2 * terrain_gen.border_width,
                        terrain_dims[1] + 2 * terrain_gen.border_width)
        
        self.map_manager = MapManager(self.cfg, self.num_envs, terrain_dims, self.device)

        if self.cfg.cameras and self.cfg.visualize_nav_data:
            self._create_debug_visualizers()

            self.loco_height_data = None
            self.nav_map_data = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        
        #self._update_player_movement()
        
        if self.cfg.cameras and self.cfg.visualize_nav_data:
            # Get data from MapManager
            # (Assuming you have robot_pos, robot_yaw, and lidar_hits)
            nav_map, loco_map, _, _ = self.map_manager.update(
                self._get_origins(),
                self._robot.data.root_pos_w,
                self._robot.data.heading_w.unsqueeze(-1), # Assuming you have this
                self._lidar_sensor.data.ray_hits_w # Assuming you have this
            )
            
            # Store for visualization
            self.nav_map_data = nav_map
            self.loco_height_data = loco_map
            self._visualize_markers()

    def _set_debug_actions(self) -> None:
        # --- MANUAL GAIT TEST MODE (Corrected) ---
        
        # 1. Setup Timing
        # Lower frequency slightly to make it easier to see individual leg movement
        freq = 3.0 
        t = self.common_step_counter * self.step_dt
        phase = t * freq * 2 * torch.pi

        # 2. Define Gait Parameters
        swing_amp = 0.5   # Swing forward/back
        lift_amp = 0.4    # Lift height
        
        # 3. Create Base Target from the ROBOT'S internal default, not our cached version.
        # This ensures 'find_joints' indices match this tensor perfectly.
        target_pos = self._robot.data.default_joint_pos.clone()

        # 4. Calculate Signals
        # Signal A: 1 to -1
        sig_A = torch.sin(torch.tensor(phase, device=self.device))
        # Signal B: Opposite of A
        sig_B = -sig_A
        
        # Lift Signal (Only lift when swinging forward)
        lift_sig_A = torch.clamp(torch.sin(torch.tensor(phase, device=self.device)), min=0)
        lift_sig_B = torch.clamp(torch.sin(torch.tensor(phase + torch.pi, device=self.device)), min=0)

        # 5. Define Groups
        legs_A = [0, 2, 4]
        legs_B = [1, 3, 5]

        # 6. Apply to Joints using String Searching
        for i in range(6):
            # Construct the specific joint names for this leg number
            hip_name = f"joint_body_leg_hip_{i}"
            upper_name = f"joint_leg_hip_leg_upper_{i}"
            
            # SEARCH for the index. 
            # find_joints returns ([indices], [names]). We take the first index.
            # This is slower than caching, but guarantees we hit the right joint.
            hip_ids, _ = self._robot.find_joints(hip_name)
            upper_ids, _ = self._robot.find_joints(upper_name)
            
            hip_idx = hip_ids[0]
            upper_idx = upper_ids[0]

            if i in legs_A:
                # Group A Logic
                target_pos[:, hip_idx] += swing_amp * sig_A
                target_pos[:, upper_idx] += lift_amp * lift_sig_A
                
            elif i in legs_B:
                # Group B Logic (Explicitly used now)
                target_pos[:, hip_idx] += swing_amp * sig_B
                target_pos[:, upper_idx] += lift_amp * lift_sig_B

        # 7. Send to Robot
        # We pass the full tensor, so we don't need to specify joint_ids
        self.processed_actions = target_pos
        self._robot.set_joint_position_target(self.processed_actions)

    def _apply_action(self) -> None:
        normalized_actions = self._actions.view(self._actions.shape[0], -1) * self.cfg.action_scale

        # For positive actions (0 to 1), scale by the positive range
        # For negative actions (-1 to 0), scale by the negative range
        action_range = torch.where(normalized_actions > 0, self.positive_range, self.negative_range)

        # Calculate the final joint positions
        self.processed_actions = self.dof_default_pos + normalized_actions * action_range
        
        """
        hip_joint = self.dof_idx.index(self._robot.find_joints("joint_body_leg_hip_1")[0][0])
        upper_joint = self.dof_idx.index(self._robot.find_joints("joint_leg_hip_leg_upper_1")[0][0])
        middle_joint = self.dof_idx.index(self._robot.find_joints("joint_leg_upper_leg_middle_1")[0][0])
        lower_joint = self.dof_idx.index(self._robot.find_joints("joint_leg_middle_leg_lower_1")[0][0])
        
        if self.common_step_counter <= 125:
            self.processed_actions = self._robot.data.default_joint_pos[:, self.dof_idx]
            # move the hip joint back
            self.processed_actions[:, upper_joint] += 3
        else:#elif self.common_step_counter <= 200:
            self.processed_actions = self._robot.data.default_joint_pos[:, self.dof_idx]
            # Slam the leg down
            self.processed_actions[:, upper_joint] -= 3
        """
        """   
        lower_joint_ids = self._robot.find_joints("joint_leg_middle_leg_lower_.*")[0]
        
        if self.common_step_counter % 500 <= 100:
            self.processed_actions = self._robot.data.default_joint_pos[:, self.dof_idx]
            self.processed_actions[:, lower_joint_ids] -= 1
            #even_joints = self._robot.find_joints(".*0.*|.*2.*|.*4.*")[0]
            #even_joints = [j for j in even_joints if j in self.dof_idx]
            #self.processed_actions[:, even_joints] = 0
        elif self.common_step_counter % 500 <= 200:
            self.processed_actions = self._robot.data.default_joint_pos[:, self.dof_idx]
        elif self.common_step_counter % 500 <= 300:
            self.processed_actions = 0
        elif self.common_step_counter % 500 <= 400:
            self.processed_actions = -self._robot.data.default_joint_pos[:, self.dof_idx]
        else:
            self.processed_actions = -self._robot.data.default_joint_pos[:, self.dof_idx]
            self.processed_actions[:, lower_joint_ids] += 1
        """

        self._robot.set_joint_position_target(self.processed_actions, joint_ids=self.dof_idx)
    
    def _can_see_player(self, player_pos_w: torch.Tensor) -> torch.Tensor:
        # Casts a ray from the robot to the player to check for line of sight
        ray_origins = self._robot.data.root_pos_w + self._up_dir * self.cfg.player_view_height_offset
        ray_directions = player_pos_w - ray_origins
        ray_directions_norm = torch.linalg.norm(ray_directions, dim=1, keepdim=True)
        ray_directions_unit = ray_directions / (ray_directions_norm + 1e-6)

        # Check for intersections with the environment
        hit_info = self.scene.ray_cast(ray_origins, ray_directions_unit)

        # Determine visibility based on raycast results
        can_see = hit_info["hit"] & (hit_info["distance"] < self.cfg.player_view_distance)
        return can_see


    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        self.is_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces, dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        )
        
        
        nav_data, height_data, far_staleness, self.last_exploration_bonus = self.map_manager.update(
            self._get_origins(),
            self._robot.data.root_pos_w,
            self._robot.data.heading_w.unsqueeze(-1),
            self._lidar_sensor.data.ray_hits_w,
        )


        # Concatenate the selected observations into a single tensor.
        obs = torch.cat(
            [
                # Robot state
                # Base info
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                # Joint info
                self._robot.data.joint_pos[:, self.dof_idx] - self._robot.data.default_joint_pos[:, self.dof_idx],
                self._robot.data.joint_vel[:, self.dof_idx],
                self._actions,
                self.is_contact[:, self.feet_contact_ids].float(),
                # Where it was moving
                self.avg_vel_b,

                # Player relative position
               # self._player.data.root_pos_w - self._robot.data.root_pos_w,

                # Staleness info
                far_staleness,
            ],
            dim=-1,
        )

        observations = {
            "observations": obs,
            "height_data": height_data.view(self.num_envs, self.cfg.loco_dim, self.cfg.loco_dim),
            "nav_data": nav_data,
        }
        
        # Need to clone because of torch.compile
        observations = {"policy": observations}# for rl_games, "critic": observations.clone()}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        
        # Reward for moving
        #movement_reward = torch.linalg.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)

    
        # Bonus for getting to target
        target_reward = torch.zeros(self.num_envs, device=self.device)
        #target_reward[reached_target_ids] = torch.log1p(self._targets_reached[reached_target_ids]) + 1


        # died if gravity is near positive (flipped over)
        died = self._robot.data.projected_gravity_b[:, 2] > 0.0
        base_contact_time = self._contact_sensor.data.current_contact_time[:, self.base_contact_ids].squeeze(-1)
        on_ground = base_contact_time > self.cfg.base_on_ground_time
        death_penalty = died.float() + on_ground.float()


        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        )
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, self.dof_idx]), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self.dof_idx]), dim=1)
        # dof velocity
        joint_vel = torch.sum(torch.square(self._robot.data.joint_vel[:, self.dof_idx]), dim=1)

        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=(1)#, 2)
        )
        
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self.feet_contact_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self.feet_contact_ids]
        feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)
        
        #first_air = self._contact_sensor.compute_first_air(self.step_dt)[
        #    :, self.feet_contact_ids
        #]
        #last_contact_time = self._contact_sensor.data.last_contact_time[:, self.feet_contact_ids]
        #feet_ground_time = torch.sum((last_contact_time - self.cfg.feet_ground_time_target) * first_air, dim=1)
        #feet_ground_time = torch.clamp(feet_ground_time, max=0)

        # undesired contacts
        undesired_contacts = torch.sum(self.is_contact[:, self.undesired_contact_ids], dim=1)
        #undesired_contact_time = torch.sum(
        #    self._contact_sensor.data.current_contact_time[:, self.undesired_contact_ids]
        #, dim=1)
        # If 3 or more feet are in contact, consider it stable
        #stable_contact = (torch.sum(self.is_contact[:, self.feet_contact_ids], dim=1) >= self.cfg.stable_contact_feet).float()

        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )


        # Body height reward
        #base_height = self._robot.data.body_pos_w[:, self.base_body_ids, 2]  # [envs, 1]
        #feet_height = self._robot.data.body_pos_w[:, self.feet_body_ids, 2] # [envs, num_feet]

        # Get lowest 3 feet
        #feet_height, _ = torch.topk(feet_height, 3, largest=False, dim=1)
        
        # Compute positive difference
        #body_relative_height = base_height - feet_height

        # Mean or sum over lowest 3 feet (you can adjust depending on desired strength)
        #body_height_reward = torch.mean(body_relative_height, dim=1)  # [envs]

        # Lower leg upright penalty
        #lower_leg_positions = self._robot.data.body_pos_w[:, self.lower_leg_body_ids]  # [envs, num_lower_legs, 3]
        #feet_pos = self._robot.data.body_pos_w[:, self.feet_body_ids] # [envs, num_feet, 3]

        # normalized from lower leg to foot
        #lower_to_foot_vectors = torch.nn.functional.normalize(feet_pos - lower_leg_positions, dim=2) # [envs, num_lower_legs, 3]
        #down_dir = torch.tensor([0, 0, -1.0], device=lower_to_foot_vectors.device)
        # Compute deviation from vertical (down direction)
        #lower_leg_reward = torch.mean(torch.mean(lower_to_foot_vectors * down_dir, dim=2), dim=1)  # [envs]


        # Hip joint deviation from neutral (0)
        hip_joint_positions = self._robot.data.joint_pos[:, self.hip_joint_ids]  # [envs, num_hips]
        hip_deviation = torch.abs(hip_joint_positions)  # absolute deviation from 0
        hip_penalty = torch.mean(hip_deviation, dim=1)

        # Feet under body penalty
        base_pos_xy = self._robot.data.body_pos_w[:, self.base_body_ids, :2].squeeze(1) # [envs, 2]
        feet_pos_xy = self._robot.data.body_pos_w[:, self.feet_body_ids, :2] # [envs, num_feet, 2]

        # Horizontal distance of each foot from body center
        feet_dist_to_base = torch.linalg.norm(feet_pos_xy - base_pos_xy.unsqueeze(1), dim=2)  # [envs, num_feet]

        # Feet that are within the body cylinder
        under_body_mask = feet_dist_to_base < self.cfg.body_penalty_radius

        # Compute how deep they are inside (closer to center = higher penalty)
        under_body_depth = torch.clamp(self.cfg.body_penalty_radius - feet_dist_to_base, min=0.0)

        # Mean penalty per environment (across all feet)
        feet_under_body_penalty = torch.mean(under_body_depth * under_body_mask.float(), dim=1)

        """
        # Reward for stepping up/down
        current_air_time = self._contact_sensor.data.current_air_time[:, self.feet_contact_ids]
        feet_in_contact = self.is_contact[:, self.feet_contact_ids]
        non_contact_count = torch.sum(~feet_in_contact, dim=1).clamp_min(1.0)  # at least 1 to avoid div by 0
        # + if moving up, - if moving down, 0 if contacting
        vel = self._robot.data.body_vel_w[:, self.feet_body_ids, 2]
        step_direction = torch.sign(vel) * torch.log1p(torch.abs(vel)) * (~feet_in_contact).float()

        # + if last_contact_time < cfg.step_up_time_end
        target_dir = torch.sign(self.cfg.step_up_time_end - current_air_time)

        # Don't penalize negative going against target
        step_values = torch.clamp(step_direction * target_dir, min=0.0) 

        # Remove reward if foot is above body
        relative_foot_height = self._robot.data.body_pos_w[:, self.feet_body_ids, 2] - self._robot.data.root_pos_w[:, 2].unsqueeze(1)
        under_body = (relative_foot_height < 0.0).float()
        step_values = step_values * under_body
        
        # Positive reward if foot moving up and last contact was recent
        #   or if foot moving down and last contact was a while ago
        step_reward = torch.sum(step_values, dim=1) / non_contact_count

        # Penalty for leg staying up too long
        step_length_penalty = torch.sum(
            torch.clamp(current_air_time - self.cfg.step_penalty_start, min=0.0, max=self.cfg.step_penalty_cap + self.cfg.step_penalty_start), dim=1
        )

        # Penalty for leg being down too long
        current_contact_time = self._contact_sensor.data.current_contact_time[:, self.feet_contact_ids]
        grounded_length_penalty = torch.sum(
            torch.clamp(current_contact_time - self.cfg.grounded_penalty_start, min=0.0, max=self.cfg.grounded_penalty_cap + self.cfg.grounded_penalty_start), dim=1
        )

        # Time since full step penalty
        # Subtract time from counters
        self.feet_step_up_counters -= self.step_dt * (~feet_in_contact).float()
        self.feet_step_down_counters -= self.step_dt * (feet_in_contact).float()
        # Recover time when foot is in desired state
        self.feet_step_up_counters += self.step_dt * (feet_in_contact).float() * self.cfg.feet_step_time_multiplier
        self.feet_step_down_counters += self.step_dt * (~feet_in_contact).float() * self.cfg.feet_step_time_multiplier
        # Clamp between -step_penalty_cap and 0
        self.feet_step_up_counters = torch.clamp(self.feet_step_up_counters, min=-self.cfg.feet_step_time_target, max=self.cfg.feet_step_time_leeway)
        self.feet_step_down_counters = torch.clamp(self.feet_step_down_counters, min=-self.cfg.feet_step_time_target, max=self.cfg.feet_step_time_leeway)

        self.feet_up_step_counter_penalty = -torch.mean(torch.clamp(self.feet_step_up_counters, max=0), dim=1)
        self.feet_down_step_counter_penalty = -torch.mean(torch.clamp(self.feet_step_down_counters, max=0), dim=1)


        # Penalty for leg joints having angle above 0 radians
        joint_pos = self._robot.data.joint_pos[:, self.dof_idx]  # [envs, num_joints]
        joint_default = self._robot.data.default_joint_pos[:, self.dof_idx]  # [num_joints] or 0 if centered
        joint_deviation = joint_pos - joint_default

        # Mean squared deviation
        joint_default_penalty = torch.mean(torch.square(joint_deviation), dim=1)

        """
        # TODO: instead of mask at end do math on specific states only
        # Patrol specific rewards
        patrol_mask = (self.robot_state == RobotState.PATROL).float()
        exploration_reward = self.last_exploration_bonus 
        # Distance from env_origin
        pos_w = self._robot.data.root_pos_w[:, :2]
        origin = self._get_origins()[:, :2]
        dist = torch.norm(pos_w - origin, dim=1)
        
        # Soft limit: Penalize (dist - radius)^2, but only if dist > radius
        excess_dist = torch.clamp(dist - self.cfg.patrol_size, min=0.0)
        boundary_penalty = torch.square(excess_dist)

        # 1. Update the Moving Average
        # We use Body Frame velocity (b) because we want it to commit to a direction relative to itself
        current_vel_xy = self._robot.data.root_lin_vel_b[:, :2]
        
        # Update equation: New_Avg = (Alpha * Current) + ((1-Alpha) * Old_Avg)
        self.avg_vel_b = (self.vel_smoothing_alpha * current_vel_xy) + \
                         ((1.0 - self.vel_smoothing_alpha) * self.avg_vel_b)

        # 2. Calculate Reward based on the SMOOTHED velocity
        # If it vibrates (+1, -1), avg_vel_b becomes ~0. Reward is low.
        # If it walks (+1, +1), avg_vel_b becomes ~1. Reward is high.
        avg_speed = torch.linalg.norm(self.avg_vel_b, dim=1)
        
        # Use the average speed for the penalty calculation instead of instantaneous
        velocity_matching = torch.square(avg_speed - self.cfg.patrol_target_velocity)

        rewards = {
            # Patrol specific rewards
            "patrol_exploration_reward": patrol_mask * exploration_reward * self.cfg.patrol_exploration_reward_scale * self.step_dt,
            "patrol_boundary_penalty": patrol_mask * boundary_penalty * self.cfg.patrol_boundary_penalty_scale * self.step_dt,
            "patrol_velocity_matching": patrol_mask * velocity_matching * self.cfg.patrol_velocity_matching_penalty_scale * self.step_dt,

            #"reach_target_reward": target_reward * self.cfg.reach_target_reward_scale * self.step_dt,
            #"death_penalty": death_penalty * self.cfg.death_penalty_scale * self.step_dt,
            #"movement_reward": movement_reward * self.cfg.movement_reward_scale * self.step_dt,
            "z_vel_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            #"dof_vel_l2": joint_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            #"feet_ground_time": feet_ground_time * self.cfg.feet_ground_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            #"undesired_contact_time": undesired_contact_time * self.cfg.undesired_contact_time_reward_scale * self.step_dt,
            #"desired_contacts": stable_contact * self.cfg.desired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            #"body_height_reward": body_height_reward * self.cfg.body_height_reward_scale * self.step_dt,
            #"lower_leg_reward": lower_leg_reward * self.cfg.lower_leg_reward_scale * self.step_dt,
            "hip_penalty": hip_penalty * self.cfg.hip_penalty_scale * self.step_dt,
            #"feet_under_body_penalty": feet_under_body_penalty * self.cfg.feet_under_body_penalty_scale * self.step_dt,
            #"step_reward": step_reward * self.cfg.step_reward_scale * self.step_dt,
            #"step_length_penalty": step_length_penalty * self.cfg.step_length_penalty_scale * self.step_dt,
            #"grounded_length_penalty": grounded_length_penalty * self.cfg.grounded_length_penalty_scale * self.step_dt,
            #"feet_up_step_counter_penalty": self.feet_up_step_counter_penalty * self.cfg.feet_up_step_time_penalty_scale * self.step_dt,
            #"feet_down_step_counter_penalty": self.feet_down_step_counter_penalty * self.cfg.feet_down_step_time_penalty_scale * self.step_dt,
            #"joint_default_penalty": joint_default_penalty * self.cfg.joint_default_penalty * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        
        # Logging
        if self.cfg.log:
            self._log_data("Episode_Reward/total_reward", torch.mean(reward))
            for key, value in rewards.items():
                episodic_sum_avg = torch.mean(value)
                self._log_data(f"Episode_Reward/{key}", episodic_sum_avg)
        
            """
            # Add the average and max amount of targets reached to log
            self._log_data(
                "Episode_Info/targets_reached_avg",
                self._last_targets_reached.float().mean(),
            )
            self._log_data(
                "Episode_Info/targets_reached_max", self._last_targets_reached.max()
            )
            # Counts for thresholds 1 through 8 in one go
            thresholds = torch.arange(1, self.cfg.log_targets_reached_max, self.cfg.log_targets_reached_step, device=self.device)
            counts = (self._last_targets_reached.unsqueeze(-1) >= thresholds).sum(dim=0) / self.num_envs

            for _, (t, c) in enumerate(zip(thresholds, counts), start=1):
                self._log_data(f"Episode_Info/targets_reached_{t}", c)
            """
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        full_time_out = self.episode_length_buf >= self.max_episode_length - 1

        timed_out = torch.zeros_like(full_time_out, dtype=torch.bool) #self._time_since_target > self._time_outs
        # change it so seperate time_outs
        
        died = self._robot.data.projected_gravity_b[:, 2] > 0.0
        base_contact_time = self._contact_sensor.data.current_contact_time[:, self.base_body_ids].squeeze(-1)
        on_ground = base_contact_time > self.cfg.base_on_ground_time
        fall_off_map = self._robot.data.root_pos_w[:, 2] < -3.0
        
        # Logging deaths/time outs per second
        if self.cfg.log:
            self._log_data("Episode_Termination/full_time_out", torch.count_nonzero(full_time_out))
            self._log_data("Episode_Termination/time_out", torch.count_nonzero(timed_out))
            self._log_data("Episode_Termination/died", torch.count_nonzero(died))
            self._log_data("Episode_Termination/on_ground", torch.count_nonzero(on_ground))
            self._log_data("Episode_Termination/fall_off_map", torch.count_nonzero(fall_off_map))

        self.terminated = died | on_ground | fall_off_map
        self.truncated = timed_out | full_time_out
        return self.terminated, self.truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)


        origins = self._get_origins()[env_ids]
        # Reset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        #default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :3] += origins
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset the player's position
        #self._player.reset(env_ids)
        #player_pos = self._player.data.default_root_state[env_ids, :7]
        #player_pos[:, :3] += origins
        #self._player.write_root_pose_to_sim(player_pos, env_ids=env_ids)
        #self._player_movement_angle[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * math.pi

        # Reset MapManager data
        self.map_manager.reset(env_ids)
        self.avg_vel_b[env_ids] = 0.0
        
        #if len(env_ids) == self.num_envs:
            # For initial randomize the initial timeout
            #self._time_since_target[:] = (-self.cfg.time_out_per_target + 
            #    torch.rand(self.num_envs, device=self.device) * self.cfg.time_out_per_target)

    def _get_origins(self) -> torch.Tensor:
        spawn_points = MultiBiomeTerrainCfg.spawns_positions
        loops = np.ceil(self.num_envs / spawn_points.shape[0])
        terrain_offsets = spawn_points.repeat(int(loops), 1)[: self.num_envs]
        return self._terrain.env_origins + terrain_offsets

    def _log_data(self, key, data) -> None:
        self.extras["log"][key] = data


    def _update_player_movement(self):
        velocity = self._player.data.root_vel_w

        # Update movement angle
        self._player_movement_angle += self.cfg.player_movement_angular_velocity * self.step_dt
        # Calculate new velocity components
        velocity[:, 0] = self.cfg.player_movement_speed * torch.cos(self._player_movement_angle)
        velocity[:, 1] = self.cfg.player_movement_speed * torch.sin(self._player_movement_angle)
        # 0 out the z velocity
        velocity[:, 2] = 0.0

        # Move in a circle
        # Write the target pose to the simulation
        self._player.write_root_velocity_to_sim(velocity)
        
        # set the rotation to be straight up
        position = self._player.data.root_pose_w
        position[:, 3:6] = 0
        position[:, 6] = 1
        
        self._player.write_root_pose_to_sim(position)

    def _get_random_colors(self, num_colors: int) -> list[tuple[float, float, float]]:
        colors = []
        for i in range(num_colors):
            # Evenly space hues for maximum color distinction
            hue = (i / min(19.0, num_colors)) % 1.0
            saturation = 0.9
            value = (i % 3) / 3.0 * 0.5 + 0.5  # Vary brightness
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb_color)
        return colors

    def _create_sphere_markers(
        self, num_markers: int, radius: float, prim_path: str, opacity: float = 1
    ) -> VisualizationMarkers:
        colors = self._get_random_colors(num_markers)
        markers = {}
        for i, color in enumerate(colors):
            marker_key = f"sphere{i}"
            markers[marker_key] = sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, opacity=opacity
                ),
            )
        marker_cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        return VisualizationMarkers(marker_cfg)

    def _create_arrow_markers(
        self, num_markers: int, prim_path: str
    ) -> VisualizationMarkers:
        colors = self._get_random_colors(num_markers)
        markers = {}
        for i, color in enumerate(colors):
            marker_key = f"arrow{i}"
            # Load the arrow mesh provided by Isaac Lab
            markers[marker_key] = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        marker_cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
        return VisualizationMarkers(marker_cfg)


    def _create_debug_visualizers(self):
        # Height map visualizers
        self.loco_pixel_size = self.cfg.loco_size / self.cfg.loco_dim
        
        loco_markers = {
            "height": sim_utils.SphereCfg(
                radius=self.loco_pixel_size * 5, # Sphere radius is half the pixel width
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.8),
            )
        }
        loco_cfg = VisualizationMarkersCfg(
            prim_path="/World/Debug/LocoHeightViz", 
            markers=loco_markers,
        )
        self.loco_height_viz = VisualizationMarkers(loco_cfg)

        # Loc map visualizers
        self.nav_pixel_size = self.cfg.nav_size / self.cfg.nav_dim
        
        nav_markers = {
            "staleness": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0), # Default 1m height
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 1.0), opacity=0.3),
            ),
            "density": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2), opacity=0.3),
            )
        }
        nav_cfg = VisualizationMarkersCfg(
            prim_path="/World/Debug/NavMapViz", 
            markers=nav_markers,
        )
        self.nav_map_viz = VisualizationMarkers(nav_cfg)

    def _get_egocentric_grid_points(self, dim, map_size, z_data, robot_pos, robot_quat):
        pixel_size = map_size / dim
        
        # Create grid indices (e.g., -12 to +12 for 25 dim)
        offset = (dim - 1) / 2.0
        indices = torch.arange(dim, device=self.device)
        
        # Create local X, Y coordinates
        # 'ij' indexing: y_grid is rows, x_grid is columns
        y_grid, x_grid = torch.meshgrid(indices - offset, indices - offset, indexing='ij')
        
        # Scale to meters
        x_local = x_grid * pixel_size
        y_local = y_grid * pixel_size
        
        # Combine with Z data
        local_points = torch.stack([x_local, y_local, z_data], dim=-1) # (Dim, Dim, 3)
        
        # Rotate and translate to world frame
        # quat_apply expects (N, 3), so we view
        local_points_flat = local_points.view(-1, 3)
        world_points = math_utils.quat_apply_yaw(robot_quat, local_points_flat) + robot_pos
        
        return world_points.view(dim, dim, 3)

    def _visualize_markers(self):
        # Skip if no data
        if self.nav_map_data is None or self.loco_height_data is None:
            return

        env_id = 0
        robot_pos = self._robot.data.root_pos_w[env_id]
        robot_quat = self._robot.data.root_quat_w[env_id]

        all_translations = []
        all_scales = []
        all_indices = []

        # Height map visualizers
        loco_data = self.loco_height_data[env_id, 0] # (25, 25)
        
        loco_points = self._get_egocentric_grid_points(
            self.cfg.loco_dim, 
            self.cfg.loco_size, 
            loco_data, 
            robot_pos, 
            robot_quat
        )
        
        # All these markers are "height" (index 0)
        num_loco_points = self.cfg.loco_dim ** 2
        all_translations.append(loco_points.view(-1, 3))
        all_indices.append(torch.full((num_loco_points,), 0, dtype=torch.int32, device=self.device))
        
        # Spheres have uniform scale
        sphere_scale = torch.full((num_loco_points, 3), self.loco_pixel_size / 2.0, device=self.device)
        all_scales.append(sphere_scale)

        # Navigation map visualizers
        stale_data = self.nav_map_data[env_id, 0]
        density_data = self.nav_map_data[env_id, 1] * 0.25 # Scaling down density for viz
        height_data = self.nav_map_data[env_id, 2] # Terrain Height relative to robot

        num_nav_points = self.cfg.nav_dim ** 2

        bar_width = self.nav_pixel_size / 8.0

        # Clamp and scale heights so they are visible
        stale_h = 0.5 * stale_data.clamp(min=0.01).view(-1, 1)
        density_h = 0.5 * density_data.clamp(min=0.01).view(-1, 1)

        scale_xy = torch.full((num_nav_points, 2), bar_width, device=self.device)
        
        stale_scales = torch.cat([scale_xy, stale_h], dim=1)
        density_scales = torch.cat([scale_xy, density_h], dim=1)

        # Put bar on top of terrain height
        z_stale = height_data + (stale_h.view(self.cfg.nav_dim, self.cfg.nav_dim) / 2.0)
        z_density = height_data + (density_h.view(self.cfg.nav_dim, self.cfg.nav_dim) / 2.0)

        # Generate base grid points (Center of the cell, correct height)
        stale_points = self._get_egocentric_grid_points(
            self.cfg.nav_dim, self.cfg.nav_size, z_stale, robot_pos, robot_quat
        ).view(-1, 3)
        
        density_points = self._get_egocentric_grid_points(
            self.cfg.nav_dim, self.cfg.nav_size, z_density, robot_pos, robot_quat
        ).view(-1, 3)

        # Put bars side by side
        offset_mag = bar_width / 1.5

        zeros = torch.zeros(num_nav_points, device=self.device)
        ones = torch.ones(num_nav_points, device=self.device)

        offset_local_stale = torch.stack([zeros, ones * offset_mag, zeros], dim=1)   # Shift Left
        offset_local_density = torch.stack([zeros, ones * -offset_mag, zeros], dim=1) # Shift Right

        # Rotate offsets to World Frame to match robot orientation
        quat_batch = robot_quat.repeat(num_nav_points, 1)
        
        offset_world_stale = math_utils.quat_apply_yaw(quat_batch, offset_local_stale)
        offset_world_density = math_utils.quat_apply_yaw(quat_batch, offset_local_density)

        # Apply offsets
        stale_points += offset_world_stale
        density_points += offset_world_density
        
        # Staleness (Index 1)
        all_translations.append(stale_points)
        all_scales.append(stale_scales)
        all_indices.append(torch.full((num_nav_points,), 0, dtype=torch.int32, device=self.device))

        # Density (Index 2)
        all_translations.append(density_points)
        all_scales.append(density_scales)
        all_indices.append(torch.full((num_nav_points,), 1, dtype=torch.int32, device=self.device))
        
        # Draw Loco Map
        self.loco_height_viz.visualize(
            translations=all_translations[0],
            scales=all_scales[0],
            marker_indices=all_indices[0]
        )
        
        # Draw Nav Map
        nav_translations = torch.cat([all_translations[1], all_translations[2]], dim=0)
        nav_scales = torch.cat([all_scales[1], all_scales[2]], dim=0)
        nav_indices = torch.cat([all_indices[1], all_indices[2]], dim=0)
        
        self.nav_map_viz.visualize(
            translations=nav_translations,
            scales=nav_scales,
            marker_indices=nav_indices
        )

        # Extract data and convert to numpy (CPU)
        env_id = 0
        
        loco_map = self.loco_height_data[env_id, 0].detach().cpu().float().numpy()
        stale_map = self.nav_map_data[env_id, 0].detach().cpu().float().numpy()
        density_map = self.nav_map_data[env_id, 1].detach().cpu().float().numpy()
        height_map = self.nav_map_data[env_id, 2].detach().cpu().float().numpy()

        # Lazy initialization of the figure (runs only once)
        if not hasattr(self, '_viz_fig'):
            plt.ion() # Interactive mode on
            self._viz_fig, self._viz_axs = plt.subplots(1, 4, figsize=(15, 4))
            self._viz_im_refs = [None, None, None, None]
            
            titles = ["Loco Height", "Nav Staleness", "Nav Density", "Nav Height"]
            for ax, title in zip(self._viz_axs, titles):
                ax.set_title(title)
                ax.axis('off') # Hide axis numbers for cleaner look

        # Update the images
        maps = [loco_map, stale_map, density_map, height_map]
        
        for i, data in enumerate(maps):
            if self._viz_im_refs[i] is None:
                # First time render
                # origin='lower' puts (0,0) at bottom-left (standard for grid maps)
                self._viz_im_refs[i] = self._viz_axs[i].imshow(data, origin='lower', cmap='viridis')
                self._viz_fig.colorbar(self._viz_im_refs[i], ax=self._viz_axs[i], fraction=0.046, pad=0.04)
            else:
                # Fast update
                self._viz_im_refs[i].set_data(data)
                # Auto-scale colors to min/max of current data
                self._viz_im_refs[i].set_clim(data.min(), data.max())

        # Refresh plot without blocking
        plt.draw()
        plt.pause(0.001)
