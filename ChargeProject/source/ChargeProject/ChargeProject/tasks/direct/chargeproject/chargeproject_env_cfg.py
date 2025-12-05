# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import math
from requests import patch
from sympy import prime
from trimesh import Trimesh
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa isort: skip
from isaaclab_assets.robots.spot import SPOT_CFG  # noqa
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

#from ChargeProject.tasks.direct.chargeproject.environments import MySceneCfg, ROBOT_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .natural_terrain import terrain_gen_cfg
from .spider_robot import SPIDER_CFG

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, TerrainGeneratorCfg  # isort: skip

from gymnasium import spaces
import numpy as np

import isaaclab.terrains as terrain_gen

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

SIMPLER_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(1.0, 1.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.05, 1.0), platform_width=2.0
        ),
    },
)


def generate_rays():
    # Configuration
    total_channels = 96  # High density
    power = 3.0          # Cubic distribution (focus on 0)
    min_angle = -90.0
    max_angle = 40.0

    # 1. Calculate ratio of Up vs Down rays based on angle size
    # This prevents the "sparse top / dense bottom" look
    span = abs(max_angle) + abs(min_angle)
    n_up = int(total_channels * (abs(max_angle) / span))
    n_down = total_channels - n_up

    # 2. Generate normalized curve (0.0 to 1.0) using Power function
    # We use (i / N)^power
    t_up = np.linspace(0, 1, n_up + 1)[1:] 
    t_down = np.linspace(0, 1, n_down + 1)[1:]

    # 3. Apply curves
    # Up goes from 0 to 40
    rays_up = max_angle * (t_up ** power)
    # Down goes from 0 to -90
    rays_down = min_angle * (t_down ** power)

    # 4. Combine and Sort
    # Using 'unique' ensures we don't have double 0.0s
    combined = np.unique(np.concatenate(([0.0], rays_up, rays_down)))
    return combined.tolist()



@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):

    def _get_config_file_path(self) -> str:
        return inspect.getfile(inspect.currentframe())

    #always should be on
    log = True

    # env
    episode_length_s = 120.0
    # - spaces definition
    action_space = 12 #24
    observation_space = spaces.Dict({
        "observations": spaces.Box(-math.inf, math.inf, shape=(63,), dtype=float),
        "height_data": spaces.Box(-math.inf, math.inf, shape=(25, 25), dtype=float),
        "nav_data": spaces.Box(-math.inf, math.inf, shape=(3, 33, 33), dtype=float)
    })
    #observation_space=48+17*11
    state_space = 0 #idk why this is here

    # simulation
    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=decimation,
        physx=PhysxCfg(
            gpu_collision_stack_size = 2**29,
            gpu_max_rigid_patch_count = 2**19
        ),
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # robot(s)
    #robot: ArticulationCfg = SPIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    events: EventCfg = EventCfg()
    # Spider Robot (base, leg_hip_i, leg_middle_i, leg_lower_i, leg_foot_i)
    #base_name = "body"
    #foot_names = "leg_foot_.*"
    #undesired_contact_body_names = "body|leg_upper_.*|leg_middle_.*|leg_lower_.*"
    #lower_leg_names = "leg_lower_.*"
    #lower_leg_joint_names = "joint_leg_middle_leg_lower_.*"
    #hip_joint_names = "joint_body_leg_hip_.*"
    #legs = 6

    # Anymal Robot
    base_name = "base"
    foot_names = ".*FOOT"
    undesired_contact_body_names = ".*THIGH"
    legs = 4
    

    
    player: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Player",
        spawn=sim_utils.CapsuleCfg(
            radius=0.25,
            height=1.4,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # Make it red
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_angular_velocity=0, angular_damping=1000.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=70.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.75)),
    )

    # Unitree Go2
    #base_name = "base"
    #foot_names = ".*_foot"
    #undesired_contact_body_names = ".*_thigh"

    # Spot
    # base_name = "body"
    # foot_names = ".*_foot"
    # undesired_contact_body_names = ".*_uleg"

    # Anymal
    # base_name = "base"
    # foot_names = ".*FOOT"
    # undesired_contact_body_names = ".*THIGH"

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    visualize_nav_data = True
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=int(1),
        env_spacing=4.0, 
        replicate_physics=True
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    
    robot_view_distance = 10.0
    robot_view_offset = 0.5

    lidar_scanner = RayCasterCfg(
        prim_path=f"/World/envs/env_.*/Robot/{base_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, robot_view_offset)),
        ray_alignment="yaw",
        max_distance=robot_view_distance,
        pattern_cfg=#patterns.LidarPatternCfg(
        #    horizontal_fov_range=(-180.0, 180.0), # Full 360-degree sweep
        #    vertical_fov_range=(-40, 30),  # Looks 40 deg up and 30 deg down
        #    horizontal_res=5.0,            # 
        #    channels=1 + int(1 + (40+30)/1),   # ring every 2.5 deg vertically
        #),
        patterns.BpearlPatternCfg(
            horizontal_fov=360,
            horizontal_res=5.0,
            vertical_ray_angles=generate_rays()
        ),
        debug_vis=visualize_nav_data, 
        mesh_prim_paths=["/World/ground"],
    )
    
    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.4, 2.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    #height_scanner = RayCasterCfg(
    #    prim_path="/World/envs/env_.*/Robot/base",
    #    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #    ray_alignment="yaw",
    #    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #    debug_vis=False,
    #    mesh_prim_paths=["/World/ground"],
    #)
    
    base_on_ground_time = 0.5 #seconds before death if base is on ground

    player_movement_angular_velocity = 0.5 # radians per second
    player_movement_speed = 1.0 # m/s

    patrol_size = 18.0 # Meters (Area the robot is expected to search)
    patrol_offset = 5.0 # Random offset from origin
    staleness_res = 1.0 # Meters (Resolution of staleness map)
    staleness_dim = int(patrol_size / staleness_res) # 64 pixels
    staleness_decay_rate = 1#/30 # 30 seconds from clean to fully stale

    nav_size = 24.0
    nav_dim = 33 # must line up with model input size
    nav_res = 0.75 # must be size / (dim - 1)

    # Locomotion Height Map (CNN Input)
    loco_size = 2.4
    loco_dim = 25 # must line up with model input size
    height_res = 0.1 # must be size / (dim - 1)
    
    marker_colors = 57


    speed_min = 0.5
    speed_max = 2.0

    # Final rewards
    action_scale = 0.5
    

    # --- Reward Scales ---
    patrol_exploration_reward_scale = 2.0
    patrol_boundary_penalty_scale = -0.25
    patrol_velocity_matching = 1.0

    attack_approach_reward_scale = 1.5
    attack_catch_reward_scale = 200.0

    # Multiplied by targets hit reward
    z_vel_reward_scale = -2.0 
    ang_vel_reward_scale = -0.05

    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    undesired_contact_time_reward_scale = -15
    flat_orientation_reward_scale = -5.0 /2

    # --- State Machine Settings ---
    # 0: Patrol, 1: Investigate, 2: Look, 3: Attack, 4: Hide
    num_states = 5



