
import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


"""
effort_mod = 0.1 # 0.025
stiffness_mod = 0.7
damping_mod = 1.0
SPIDER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        "joint_body_leg_hip_.*",
        "joint_leg_hip_leg_upper_.*",
        "joint_leg_upper_leg_middle_.*",
        "joint_leg_middle_leg_lower_.*",
    ],
    effort_limit_sim={
        "joint_body_leg_hip_.*": 208 * effort_mod,
        "joint_leg_hip_leg_upper_.*": 256.0 * effort_mod,
        "joint_leg_upper_leg_middle_.*": 176.0 * effort_mod,
        "joint_leg_middle_leg_lower_.*": 125.0 * effort_mod,
    },
    #velocity_limit_sim={
    #    "joint_body_leg_hip_.*": 3.0,
    #    "joint_leg_hip_leg_upper_.*": 3.0,
    #    "joint_leg_upper_leg_middle_.*": 3.0,
    #    "joint_leg_middle_leg_lower_.*": 3.0,
    #},
    stiffness={
        "joint_body_leg_hip_.*": 208 * effort_mod * stiffness_mod,
        "joint_leg_hip_leg_upper_.*": 256.0 * effort_mod * stiffness_mod,
        "joint_leg_upper_leg_middle_.*": 176.0 * effort_mod * stiffness_mod,
        "joint_leg_middle_leg_lower_.*": 125.0 * effort_mod * stiffness_mod,
    },
    damping={
        "joint_body_leg_hip_.*": 1.63 * math.sqrt(effort_mod) * damping_mod,
        "joint_leg_hip_leg_upper_.*": 1.81 * math.sqrt(effort_mod) * damping_mod,
        "joint_leg_upper_leg_middle_.*": 1.50 * math.sqrt(effort_mod) * damping_mod,
        "joint_leg_middle_leg_lower_.*": 1.39 * math.sqrt(effort_mod) * damping_mod,
    },
    friction={
        "joint_body_leg_hip_.*": 0.2,
        "joint_leg_hip_leg_upper_.*": 0.2,
        "joint_leg_upper_leg_middle_.*": 0.2,
        "joint_leg_middle_leg_lower_.*": 0.1,
    },
    armature = 0.005,
)
"""

# Make sure this is synced with CreateURDF and usd
SPIDER_JOINT_INFO = { 
    "default_pos": {
        "joint_body_leg_hip_.*": math.radians(0.0),
        "joint_leg_hip_leg_upper_.*": math.radians(30.0),
        "joint_leg_upper_leg_middle_.*": math.radians(-75.0),
        "joint_leg_middle_leg_lower_.*": math.radians(-45.0),
    },
    "limit_min": {
        "joint_body_leg_hip_.*": math.radians(-45.0),
        "joint_leg_hip_leg_upper_.*": math.radians(-10.0),
        "joint_leg_upper_leg_middle_.*": math.radians(-105.0),
        "joint_leg_middle_leg_lower_.*": math.radians(-85.0),
    },
    "limit_max": {
        "joint_body_leg_hip_.*": math.radians(45.0),
        "joint_leg_hip_leg_upper_.*": math.radians(60.0),
        "joint_leg_upper_leg_middle_.*": math.radians(-35.0),
        "joint_leg_middle_leg_lower_.*": math.radians(-5.0),
    },
}



SPIDER_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        "joint_body_leg_hip_.*",
        "joint_leg_hip_leg_upper_.*",
        "joint_leg_upper_leg_middle_.*",
        "joint_leg_middle_leg_lower_.*",
    ],
    effort_limit_sim={
        "joint_body_leg_hip_.*": 50,
        "joint_leg_hip_leg_upper_.*": 100.0,
        "joint_leg_upper_leg_middle_.*": 100.0,
        "joint_leg_middle_leg_lower_.*": 90.0,
    },
    stiffness={
        "joint_body_leg_hip_.*": 100,
        "joint_leg_hip_leg_upper_.*": 300,
        "joint_leg_upper_leg_middle_.*": 300,
        "joint_leg_middle_leg_lower_.*": 200,
    },
    velocity_limit_sim=50,
    damping={
        "joint_body_leg_hip_.*": 2.0,
        "joint_leg_hip_leg_upper_.*": 5.0,
        "joint_leg_upper_leg_middle_.*": 5.0,
        "joint_leg_middle_leg_lower_.*": 3.0,
    },
    friction=0.05,
)

# print args in SPIDER_ACTUATOR_CFG
print(SPIDER_ACTUATOR_CFG)

SPIDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path=f"../../../SpiderBot/spider/spider.usd", not tested go down
        usd_path=f"../SpiderBot/spider/spider.usd",
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.265),  # start above the ground
        # Default angles: body-hip=0°, hip-upper=30°, upper-middle=-65°, middle--lower=-55°
        joint_pos=SPIDER_JOINT_INFO["default_pos"],
    ),
    actuators={
        "legs": SPIDER_ACTUATOR_CFG,
    },
    soft_joint_pos_limit_factor=0.95,
)