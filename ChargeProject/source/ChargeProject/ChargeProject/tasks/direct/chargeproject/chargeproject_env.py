from __future__ import annotations
import math
import colorsys
from time import time

import gymnasium as gym
import torch
import torch.compiler
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .spider_robot import SPIDER_JOINT_INFO

from .chargeproject_env_cfg import ChargeprojectEnvCfg

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

import inspect
import os


class ChargeprojectEnv(DirectRLEnv):
    cfg: ChargeprojectEnvCfg

    def __init__(
        self, cfg: ChargeprojectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Target positions
        self._desired_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._next_desired_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._time_since_target = torch.zeros(self.num_envs, device=self.device)
        self._targets_reached = torch.zeros(self.num_envs, device=self.device)
        self._time_outs = torch.zeros(self.num_envs, device=self.device)

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

        # Change to initialize with nulls
        self._distance_buffer = torch.zeros(self.num_envs, self.cfg.distance_lookback, device=self.device)

        log_dir = self.cfg.log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.extras["log"] = dict()

        # Save the variables in ChargeprojectEnvCfg to a text file for reference
        with open(os.path.join(log_dir, "env_config.txt"), "w") as f:
            for attr, value in vars(self.cfg).items():
                f.write(f"{attr}: {value}\n")

        current_file = inspect.getfile(inspect.currentframe())
        with open(os.path.join(log_dir, "env_code.py.txt"), "w") as f:
            with open(current_file, "r") as current_f:
                f.write(current_f.read())
        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # check if cameras exist
        if not hasattr(self.cfg, "cameras"):
            self.cfg.cameras = True
            print("No camera setting found in cfg, defaulting to cameras=True")

        if self.cfg.cameras:
            self.goal_pos_visualizer = self._create_sphere_markers(
                self.cfg.marker_colors, 0.25, "/Visuals/Command/goal_position"
            )
            self.identifier_visualizer = self._create_arrow_markers(
                self.cfg.marker_colors, "/Visuals/Command/identifier_arrow"
            )

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

        self._up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        
        target_distance = torch.linalg.norm(self._desired_pos - self._robot.data.root_pos_w[:, :2], dim=1)

        if self.cfg.cameras:
            self._visualize_markers(target_distance)

        index = self._sim_step_counter % self.cfg.distance_lookback
        
        self._distance_buffer[:, index] = target_distance.squeeze(-1)

    def _apply_action(self) -> None:
        normalized_actions = self._actions.view(self._actions.shape[0], -1)
        
        # For positive actions (0 to 1), scale by the positive range
        # For negative actions (-1 to 0), scale by the negative range
        # torch.where is perfect for this: condition, value_if_true, value_if_false
        action_range = torch.where(normalized_actions > 0, self.positive_range, self.negative_range)

        # Calculate the final joint positions
        self.processed_actions = self.dof_default_pos + normalized_actions * action_range
        
        # Optional but recommended: Explicitly clip to ensure no floating point errors exceed limits
        self.processed_actions = torch.clamp(self.processed_actions, self.dof_min_limits, self.dof_max_limits)

        

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
            
        lower_joint_ids = self._robot.find_joints("joint_leg_middle_leg_lower_.*")[0]
        # For testing standing on 3 legs
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
        # print the actions of the first env
        #if self._sim_step_counter % 2 == 0:
        #    print(self.processed_actions[0].shape, self.processed_actions[0])
        self._robot.set_joint_position_target(self.processed_actions, joint_ids=self.dof_idx)
        # self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)
    
    def _get_relative_target_info(
        self, target_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_target_pos_w = target_pos_w - self._robot.data.root_pos_w[:, :2]
        distance = torch.linalg.norm(relative_target_pos_w, dim=1, keepdim=True)

        # Pad the 2D world vector to 3D for rotation (z=0)
        relative_target_pos_w_3d = torch.cat(
            (relative_target_pos_w, torch.zeros_like(distance)), dim=1
        )

        # Rotate the world vector into the robot's local body frame
        relative_target_pos_b_3d = math_utils.quat_apply_inverse(
            self._robot.data.root_quat_w, relative_target_pos_w_3d
        )

        # Normalize the resulting 2D body-frame vector to get the unit vector
        unit_vector_b = relative_target_pos_b_3d[:, :2] / (distance + 1e-6)

        return unit_vector_b, distance
    
    #@torch.compile(mode="reduce-overhead")
    def _get_observations_impl(self, height_scanner_data) -> dict:
        target_unit_vector, target_distance = self._get_relative_target_info(
            self._desired_pos
        )

        next_target_unit_vector, next_target_distance = self._get_relative_target_info(
            self._next_desired_pos
        )

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
        
        height_data = (
            height_scanner_data.pos_w[:, 2].unsqueeze(1) - height_scanner_data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)

        # turn height data into a (16, 16) grid
        height_data = height_data.view(self.num_envs, 16, 16)
        
        # Concatenate the selected observations into a single tensor.
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                target_unit_vector,
                target_distance,
                next_target_unit_vector,
                next_target_distance,
                self._robot.data.joint_pos[:, self.dof_idx] - self._robot.data.default_joint_pos[:, self.dof_idx],
                self._robot.data.joint_vel[:, self.dof_idx],
                self._actions,
            ],
            dim=-1,
        )
        return {
            "observations": obs,
            "height_data": height_data
        }

        """
        base_quat_w = self._robot.data.root_quat_w

        quat_expanded = base_quat_w[:, None, :].expand(-1, len(self.feet_body_ids), -1)
        feet_pos = self._robot.data.body_pos_w[:, self.feet_body_ids] - (self._robot.data.root_pos_w).unsqueeze(1)
        foot_vectors_rotated = math_utils.quat_apply(quat_expanded, feet_pos)
        # Concatenate the selected observations into a single tensor.
        base_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                foot_vectors_rotated.view(self.num_envs, -1),
                target_unit_vector,
                target_distance,
                next_target_unit_vector,
                next_target_distance,
            ],
            dim=-1,
        )
        # Concatenate the selected observations into a single tensor.
        
        # hip positions (vec in direction)
        leg_obs = []
        for i in range(self.action_space.shape[1]):
            # rotate position vectors
            hip_pos = self._robot.data.body_pos_w[:, self.hip_body_ids[i]] - self._robot.data.root_pos_w
            hip_vec_rotated = math_utils.quat_apply(base_quat_w, hip_pos)
            cycled_ids = self.feet_body_ids[i:] + self.feet_body_ids[:i] # Cycle so current leg is first
            feet_pos = self._robot.data.body_pos_w[:, cycled_ids] - (self._robot.data.root_pos_w).unsqueeze(1)
            foot_vec_rotated = math_utils.quat_apply(quat_expanded, feet_pos)
            # offset all feet by current leg foot position (and remove current position because it's zero)
            foot_vec_rotated = foot_vec_rotated - foot_vec_rotated[:, 0:1, :]
            foot_vec_rotated = foot_vec_rotated[:, 1:, :]  # remove the zero vector for current foot

            leg_obs.append(torch.cat(
                [
                self._robot.data.joint_pos[:, self.leg_joint_ids[i]] - self._robot.data.default_joint_pos[:, self.leg_joint_ids[i]],
                self._robot.data.joint_vel[:, self.leg_joint_ids[i]],
                hip_vec_rotated, # hip position vector
                foot_vec_rotated.view(self.num_envs, -1), # relative foot positions of all feet
                self.actions[:, i],
                self.is_contact[:, self.feet_contact_ids[i]].float().unsqueeze(-1),
                self._contact_sensor.data.current_contact_time[:, self.feet_contact_ids[i]].float().unsqueeze(-1),
                self._contact_sensor.data.current_air_time[:, self.feet_contact_ids[i]].float().unsqueeze(-1),
                ],
                dim=-1,
            ))
        leg_obs = torch.stack(leg_obs, dim=1)

        return {
            "base_obs": base_obs,
            "leg_obs": leg_obs,
            "height_data": height_data
        }
        """

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        height_scanner_data = self._height_scanner.data

        observations = self._get_observations_impl(height_scanner_data)
        # Need to clone because of torch.compile
        observations = {"policy": observations}# for rl_games, "critic": observations.clone()}
        return observations

   #@torch.compile(mode="reduce-overhead")    
    def _get_rewards_impl(self, reached_target_ids) -> torch.Tensor:
        target_unit_vector, target_distance = self._get_relative_target_info(self._desired_pos)

        # - Rewards -
        # Reward for progress towards the target
        sum_valid = torch.nansum(self._distance_buffer, dim=1)
        count_valid = torch.clamp(torch.sum(~torch.isnan(self._distance_buffer), dim=1), min=1.0)
        previous_buffered_distance = sum_valid / count_valid
        difference = (previous_buffered_distance - target_distance.squeeze(-1)) * (1 + 0.5 * (count_valid-1))
        if self.cfg.progress_pow != 1.0: 
            progress_reward = torch.sign(difference) * torch.pow(torch.abs(difference), self.cfg.progress_pow)
        else:
            progress_reward = difference
        #progress_reward = difference
        #progress_reward *= torch.log1p(self._targets_reached/2) + 1

        # Reward for moving (average of buffer is = to this)
        movement_reward = torch.linalg.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)

        
        # Velocity alignment the target
        velocity_alignment_reward = torch.nn.functional.cosine_similarity(
           self._robot.data.root_lin_vel_w[:, :2], target_unit_vector, dim=1
        )

        # Bonus for getting to target
        target_reward = torch.zeros(self.num_envs, device=self.device)
        target_reward[reached_target_ids] = torch.log1p(self._targets_reached[reached_target_ids]) + 1


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
        feet_air_time = torch.sum((last_air_time - self.cfg.feet_air_time_target) * first_contact, dim=1)
        feet_air_time = torch.clamp(feet_air_time, max=0)
        
        first_air = self._contact_sensor.compute_first_air(self.step_dt)[
            :, self.feet_contact_ids
        ]
        last_contact_time = self._contact_sensor.data.last_contact_time[:, self.feet_contact_ids]
        feet_ground_time = torch.sum((last_contact_time - self.cfg.feet_ground_time_target) * first_air, dim=1)
        feet_ground_time = torch.clamp(feet_ground_time, max=0)

        # undesired contacts
        undesired_contacts = torch.sum(self.is_contact[:, self.undesired_contact_ids], dim=1)
        undesired_contact_time = torch.sum(
            self._contact_sensor.data.current_contact_time[:, self.undesired_contact_ids]
        , dim=1)
        # If 3 or more feet are in contact, consider it stable
        stable_contact = (torch.sum(self.is_contact[:, self.feet_contact_ids], dim=1) >= self.cfg.stable_contact_feet).float()

        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )


        # Body height reward
        base_height = self._robot.data.body_pos_w[:, self.base_body_ids, 2]  # [envs, 1]
        feet_height = self._robot.data.body_pos_w[:, self.feet_body_ids, 2] # [envs, num_feet]

        # Get lowest 3 feet
        feet_height, _ = torch.topk(feet_height, 3, largest=False, dim=1)
        
        # Compute positive difference
        body_relative_height = base_height - feet_height

        # Mean or sum over lowest 3 feet (you can adjust depending on desired strength)
        body_height_reward = torch.mean(body_relative_height, dim=1)  # [envs]

        # Lower leg upright penalty
        lower_leg_positions = self._robot.data.body_pos_w[:, self.lower_leg_body_ids]  # [envs, num_lower_legs, 3]
        feet_pos = self._robot.data.body_pos_w[:, self.feet_body_ids] # [envs, num_feet, 3]

        # normalized from lower leg to foot
        lower_to_foot_vectors = torch.nn.functional.normalize(feet_pos - lower_leg_positions, dim=2) # [envs, num_lower_legs, 3]
        down_dir = torch.tensor([0, 0, -1.0], device=lower_to_foot_vectors.device)
        # Compute deviation from vertical (down direction)
        lower_leg_reward = torch.mean(torch.mean(lower_to_foot_vectors * down_dir, dim=2), dim=1)  # [envs]


        # Hip joint deviation from neutral (0)
        hip_joint_positions = self._robot.data.joint_pos[:, self.hip_joint_ids]  # [envs, num_hips]
        hip_deviation = torch.abs(hip_joint_positions)  # absolute deviation from 0
        hip_penalty = torch.mean(hip_deviation, dim=1)

        # Feet under body penalty
        base_pos_xy = self._robot.data.body_pos_w[:, self.base_body_ids, :2].squeeze(1) # [envs, 2]
        feet_pos_xy = self._robot.data.body_pos_w[:, self.feet_body_ids, :2] # [envs, num_feet, 2]

        # Horizontal distance of each foot from body center
        feet_dist_to_base = torch.linalg.norm(feet_pos_xy - base_pos_xy.unsqueeze(1), dim=2)  # [envs, num_feet]

        # Feet that are within the body cylinder (radius < 0.25)
        under_body_mask = feet_dist_to_base < self.cfg.body_penalty_radius

        # Compute how deep they are inside (closer to center = higher penalty)
        under_body_depth = torch.clamp(self.cfg.body_penalty_radius - feet_dist_to_base, min=0.0)

        # Mean penalty per environment (across all feet)
        feet_under_body_penalty = torch.mean(under_body_depth * under_body_mask.float(), dim=1)


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


        return {
            "progress_reward": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            "velocity_alignment_reward": velocity_alignment_reward * self.cfg.velocity_alignment_reward_scale * self.step_dt,
            "reach_target_reward": target_reward * self.cfg.reach_target_reward_scale * self.step_dt,
            "death_penalty": death_penalty * self.cfg.death_penalty_scale * self.step_dt,
            "movement_reward": movement_reward * self.cfg.movement_reward_scale * self.step_dt,
            "z_vel_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "dof_vel_l2": joint_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "feet_ground_time": feet_ground_time * self.cfg.feet_ground_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "undesired_contact_time": undesired_contact_time * self.cfg.undesired_contact_time_reward_scale * self.step_dt,
            "desired_contacts": stable_contact * self.cfg.desired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "body_height_reward": body_height_reward * self.cfg.body_height_reward_scale * self.step_dt,
            "lower_leg_reward": lower_leg_reward * self.cfg.lower_leg_reward_scale * self.step_dt,
            "hip_penalty": hip_penalty * self.cfg.hip_penalty_scale * self.step_dt,
            "feet_under_body_penalty": feet_under_body_penalty * self.cfg.feet_under_body_penalty_scale * self.step_dt,
            "step_reward": step_reward * self.cfg.step_reward_scale * self.step_dt,
            "step_length_penalty": step_length_penalty * self.cfg.step_length_penalty_scale * self.step_dt,
            "grounded_length_penalty": grounded_length_penalty * self.cfg.grounded_length_penalty_scale * self.step_dt,
            "feet_up_step_counter_penalty": self.feet_up_step_counter_penalty * self.cfg.feet_up_step_time_penalty_scale * self.step_dt,
            "feet_down_step_counter_penalty": self.feet_down_step_counter_penalty * self.cfg.feet_down_step_time_penalty_scale * self.step_dt,
            "joint_default_penalty": joint_default_penalty * self.cfg.joint_default_penalty * self.step_dt,
        }


    def _get_rewards(self) -> torch.Tensor:

        # Check if distance is within tolerance
        target_distance = torch.linalg.norm(self._desired_pos - self._robot.data.root_pos_w[:, :2], dim=1)
        reached_target = target_distance.squeeze(-1) < self.cfg.success_tolerance
        reached_target_ids = reached_target.nonzero(as_tuple=False).squeeze(-1)

        # Generate a new target immediately for the environments that reached theirs
        if len(reached_target_ids) > 0:
            self._move_next_targets(reached_target_ids)

        rewards = self._get_rewards_impl(reached_target_ids)

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        

        # Logging
        if self.cfg.log:
            self._log_data("Episode_Reward/total_reward", torch.mean(reward))
            for key, value in rewards.items():
                episodic_sum_avg = torch.mean(value)
                self._log_data(f"Episode_Reward/{key}", episodic_sum_avg)
        
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

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        full_time_out = self.episode_length_buf >= self.max_episode_length - 1

        self._time_since_target += self.step_dt
        timed_out = self._time_since_target > self._time_outs
        # change it so seperate gtime_outs
        
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

        # Reset
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        #default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._last_targets_reached[env_ids] = self._targets_reached[env_ids].clone()

        # Set Next target positions to be reset position
        self._next_desired_pos[env_ids] = self._robot.data.root_pos_w[env_ids, :2].clone()
        self._move_next_targets(env_ids)
        self._move_next_targets(env_ids)  # call twice to initialize both current and next target positions
        if len(env_ids) == self.num_envs:
            # For initial randomize the initial timeout
            self._time_since_target[:] = (-self.cfg.time_out_per_target + 
                torch.rand(self.num_envs, device=self.device) * self.cfg.time_out_per_target)

        self._time_outs[env_ids] = self.cfg.time_out_per_target
        self._targets_reached[env_ids] = 0

    
    def _log_data(self, key, data) -> None:
        self.extras["log"][key] = data

    def _reached_target(self, env_ids):
        # Get robot and target positions (only x, y)
        robot_pos = self._robot.data.root_pos_w[env_ids, :2]
        target_pos = self._desired_pos[env_ids]
        # Calculate distance
        dist = torch.linalg.norm(robot_pos - target_pos, dim=1)

        # Check if distance is within tolerance
        return dist < self.cfg.success_tolerance

    def _move_next_targets(self, env_ids: Sequence[int]):
        # Update current position
        self._desired_pos[env_ids] = self._next_desired_pos[env_ids].clone()

        num_resets = len(env_ids)

        radius = self.cfg.point_max_distance
        radius += (
            self.cfg.point_min_distance - self.cfg.point_max_distance
        ) * torch.rand(num_resets, device=self.device)
        angle = 2 * math.pi * torch.rand(num_resets, device=self.device)
        new_target_pos_xy = self._desired_pos[env_ids] + torch.stack(
            [radius * torch.cos(angle), radius * torch.sin(angle)], dim=1
        )

        # Update current and next desired positions
        self._next_desired_pos[env_ids] = new_target_pos_xy

        # set _distance_buffer to nan and add current distance as element
        self._distance_buffer[env_ids] = torch.nan
        index = self._sim_step_counter % self.cfg.distance_lookback
        distance = torch.linalg.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._desired_pos[env_ids], dim=1
        )
        self._distance_buffer[env_ids, index] = distance

        # Reset timer
        self._time_since_target[env_ids] = 0.0
        self._targets_reached[env_ids] += 1
        self._time_outs[env_ids] = (
            self.cfg.time_out_per_target
            - self._targets_reached[env_ids] * self.cfg.time_out_decrease_per_target
        )

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

    def _visualize_markers_impl(self, target_distance):
        desired_pos_3d = torch.cat(
            [
                self._desired_pos,
                0.2 * torch.ones((self.num_envs, 1), device=self.device),
            ],
            dim=-1,
        )
        marker_indices = (
            torch.arange(self.num_envs, device=self.device) % self.cfg.marker_colors
        )

        # --- Identifier Arrow Marker ---
        robot_pos_2d = self._robot.data.root_pos_w[:, :2]
        relative_target_pos = self._desired_pos - robot_pos_2d
        yaw = torch.atan2(relative_target_pos[:, 1], relative_target_pos[:, 0])
        orientations = math_utils.quat_from_angle_axis(yaw, self._up_dir)

        # Linearly interpolate scale from 0.15 down to 0 as the robot gets closer
        # Clamp between 0.0 and 1.0 to handle cases where the robot is farther than max_dist
        # The maximum distance should be `self.cfg.point_max_distance`
        scale_factor = torch.clamp(
            target_distance / self.cfg.point_max_distance, 0.0, 1.0
        )

        # Base scale for the arrow's width and height
        arrow_thickness_scale = 0.15 * scale_factor.unsqueeze(1) + 0.05
        # Make the arrow's length (x-axis) a bit longer for better visibility, relative to thickness
        scales = torch.cat(
            [2 * arrow_thickness_scale, arrow_thickness_scale, arrow_thickness_scale],
            dim=1,
        ).squeeze(-1)

        arrow_positions = self._robot.data.root_pos_w + torch.tensor(
            [0.0, 0.0, 0.5], device=self.device
        )

        return desired_pos_3d, marker_indices, arrow_positions, orientations, scales

    def _visualize_markers(self, target_distance):
        (
            desired_pos_3d,
            marker_indices,
            arrow_positions,
            orientations,
            scales,
        ) = self._visualize_markers_impl(target_distance)
        self.goal_pos_visualizer.visualize(
            desired_pos_3d, marker_indices=marker_indices
        )
        self.identifier_visualizer.visualize(
            translations=arrow_positions,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )
