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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .natural_terrain import terrain_gen_cfg
from .spider_robot import SPIDER_CFG

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, TerrainGeneratorCfg  # isort: skip

from gymnasium import spaces

import isaaclab.terrains as terrain_gen

SIMPLER_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
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

ANYMAL_JOINT_INFO = { 
    "default_pos": {
        ".*HAA": 0.0,
        ".*F_HFE": 0.4,
        ".*H_HFE": -0.4,
        ".*F_KFE": -0.8,
        ".*H_KFE": 0.8,
    },
    "limit_min": {
        # Left side HAA limits (LF_HAA, LH_HAA)
        ".*L._HAA": -0.72, 
        # Right side HAA limits (RF_HAA, RH_HAA)
        ".*R._HAA": -0.49,
        # Hip Flexion/Extension (All legs)
        ".*HFE": -1.5, #-9.42477796077,
        # Knee Flexion/Extension (All legs)
        ".*KFE": -2.5, #-9.42477796077,
    },
    "limit_max": {
        # Left side HAA limits (LF_HAA, LH_HAA)
        ".*L._HAA": 0.49,
        # Right side HAA limits (RF_HAA, RH_HAA)
        ".*R._HAA": 0.72,
        # Hip Flexion/Extension (All legs)
        ".*HFE": 1.5, #9.42477796077,
        # Knee Flexion/Extension (All legs)
        ".*KFE": -0.1, #9.42477796077,
    },
}


@configclass
class ChargeprojectEnvCfg(DirectRLEnvCfg):

    def _get_config_file_path(self) -> str:
        return inspect.getfile(inspect.currentframe())

    #always should be on
    log = True

    # env
    episode_length_s = 20.0
    # - spaces definition
    action_space = 12 #24
    """
    observation_space_old = spaces.Dict({
        "observations": spaces.Box(-math.inf, math.inf, shape=(97 - 2 - 12*3 - 4 + 3,), dtype=float),
        "height_data": spaces.Box(-math.inf, math.inf, shape=(25, 25), dtype=float),
        "nav_data": spaces.Box(-math.inf, math.inf, shape=(3, 33, 33), dtype=float)
    })"""
    observation_space = spaces.Dict({
        "observations": spaces.Box(-math.inf, math.inf, shape=(48,), dtype=float),
        "height_data": spaces.Box(-math.inf, math.inf, shape=(25, 25), dtype=float),
        #"nav_data": spaces.Box(-math.inf, math.inf, shape=(3, 33, 33), dtype=float)
    })
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
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # robot(s)
    robot: ArticulationCfg = SPIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
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
            radius=0.025,
            height=0.14,
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=int(1024*1.5),#0),
        env_spacing=4.0, 
        replicate_physics=True
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG #terrain_gen_cfg, # SIMPLER_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )


    visualize_nav_data = False
    lidar_scanner = RayCasterCfg(
        prim_path=f"/World/envs/env_.*/Robot/{base_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.4)),
        ray_alignment="base",
        max_distance=50.0,
        pattern_cfg=patterns.LidarPatternCfg(
            horizontal_fov_range=(-180.0, 180.0), # Full 360-degree sweep
            vertical_fov_range=(-40, 40),  # Looks 40 deg up and 40 deg down
            horizontal_res=2.5,            # 
            channels=1 + int((40+40)/2.5),   # ring every 2.5 deg vertically
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
    
    base_on_ground_time = 0.5 #seconds before death if base is on ground

    player_movement_angular_velocity = 0.5 # radians per second
    player_movement_speed = 1.0 # m/s

    patrol_size = 32.0 # Meters (Area the robot is expected to search)
    staleness_res = 0.5 # Meters (Resolution of staleness map)
    staleness_dim = int(patrol_size / staleness_res) # 64 pixels
    staleness_decay_rate = 1/30 # 30 seconds from clean to fully stale

    nav_size = 24.0
    nav_dim = 33 # must line up with model input size
    nav_res = 0.75 # must be size / (dim - 1)

    # Locomotion Height Map (CNN Input)
    loco_size = 2.4
    loco_dim = 25 # must line up with model input size
    height_res = 0.1 # must be size / (dim - 1)
    
    marker_colors = 57

    # Final rewards
    action_scale = 1.0
    

    # Training stages:
    # Removed scaling progress/velocity rewards by number of targets reached
    # init, 2k steps: Learning rate 1e-4 (learning rate may be able to start at 1e-3)
    #                 joint_leg_middle_leg_lower_ = 125 (may not be needed)
    #                 Stopped training after velocity/progress became noticeable (2025-10-22_19-49-57_ppo_torch, v1.0.0)
    # init, 3k steps: Learning rate to 1e-3
    #                 Stopped training after height ~equal init height (2025-10-22_20-42-36_ppo_torch, v1.0.1)
    # start, 5.5k steps: After init comments below
    #                   joint_leg_middle_leg_lower_ = 160 
    #                   Stopped after it starts "jumping" (2025-10-23_15-28-12_ppo_torch, v1.1.0)
    # main, 33k steps: Learning rate to 1e-4?
    #                  After start comments below (2025-10-23_13-09-24_ppo_torch, spiderbot_v1.2.0)
    #


    # x this means bad and ignore
    # new train on smaller keeps main changes (1.0e-03)
    # end after 3k steps, (2025-10-23_18-31-31_ppo_torchm, v3.0.0) 
    #   do --  
    # end after 3k steps, (2025-10-23_19-17-46_ppo_torch, v3.1.0)
    #   do --- changes when progress gets too dominant 
    # end after 4k steps, changes after seeing flaws in reward scales (2025-10-23_19-46-39_ppo_torch, v3.2.0)
    #   do ----, lr 1.0e-04
    # end after 8k steps, (2025-10-23_21-43-55_ppo_torch, spiderbot_v3.3.0)
    #   do ----- 
    # BAD didn't use this checkpoint for next, skipped loading this checkpoint ~~end After 110k steps, (2025-10-23_23-28-24_ppo_torch, spiderbot_v3.4.0)~~
    #   do =,
    # x change to rougher terrain
    # x end After 14k steps, (2025-10-24_09-37-04_ppo_torch, spiderbot_v3.4.1)
    #   do ==, divide torques

    #  1e-4 then set to 1e-3 for faster learning

    # --- Reward Scales ---
    patrol_exploration_reward_scale = 0.5 * 0.0
    patrol_boundary_penalty_scale = -10.0
    patrol_velocity_matching_penalty_scale = -0.25
    patrol_target_velocity = 1.0 # m/s
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # Multiplied by targets hit reward
    #reach_target_reward_scale = 1000 * 4 # == add * 4
    death_penalty_scale = -20
    #movement_reward_scale = 30 / 6 / 2 * 2 # -- add / 6 # ---- add / 2 # = add * 2
    z_vel_reward_scale = -2.0 
    ang_vel_reward_scale = -0.05

    joint_torque_reward_scale = -2.5e-5# / 2
    joint_accel_reward_scale = -2.5e-7# / 2 / 150
    #dof_vel_reward_scale = 0.0
    action_rate_reward_scale = -0.01# / 2

    feet_air_time_reward_scale = 0.5# / 1.5
    feet_air_time_target = 0.5 # set to 0.4 after start (was 0.7 but not sure if this matters) # ---- set 0.35 # ----- set to 0.6
    #feet_ground_time_reward_scale = 40
    #feet_ground_time_target = 0.5 # set to 0.4 after start (was 0.7 but not sure if this matters) # ---- set 0.35 # ----- set to 0.6
    
    undesired_contact_reward_scale = -1.0
    undesired_contact_time_reward_scale = -15
    #desired_contact_reward_scale = 10 * 4 / 8 # add (*4) after start ---- add / 8
    #stable_contact_feet = 2 # ---- set to 2 (was 3)
    flat_orientation_reward_scale = 0#-5.0
    #body_height_reward_scale = 114 / 2 / 2 # * 4 # Remove (*4) after init # add (/2) after start # ----- add / 2
    #lower_leg_reward_scale = 200 / 10 # add (/10) after start
    #hip_penalty_scale = 0#-30 / 5 # ---- Add / 5
    #feet_under_body_penalty_scale = -72000 * 2 / 8# add (*2) after start # == add / 8
    #body_penalty_radius = 0.175

    # rewards positive joint velocity when time from contact
    #step_reward_scale = 50 / 4 / 4 / 2# Set 50 after init (from 0) # Add (*4) after start # -- remove *4 #--- add / 4 # ---- add / 4
    #step_up_time_end = 0.3 # set to 0.2 after start (was 0.55)  # ----- set to 0.3
    # linear scale of penalty if leg doesn't step in this time
    #step_length_penalty_scale = -15
    #step_penalty_start = 1.3
    #step_penalty_cap = 2.0
    #grounded_length_penalty_scale = -15
    #grounded_penalty_start = 2.0
    #grounded_penalty_cap = 2.0
    #feet_up_step_time_penalty_scale = -160 * 2 / 3 # set -60 after start (from 0) # -- set -160  # ----- add * 2 # = / 3
    #feet_down_step_time_penalty_scale = -160 * 2 / 3 # set -60 after start (from 0) # -- set -160 # ----- add * 2 # = / 3
    #feet_step_time_multiplier = 1.5 * 2 # makes more going up than being on ground # == add * 2
    #feet_step_time_target = 0.4
    #feet_step_time_leeway = 2.0 # clamped out on positive  # ----- set to 0.8 # = set to 2.0

    #joint_default_penalty = 0


    # --- State Machine Settings ---
    # 0: Patrol, 1: Investigate, 2: Look, 3: Attack, 4: Hide
    num_states = 5
