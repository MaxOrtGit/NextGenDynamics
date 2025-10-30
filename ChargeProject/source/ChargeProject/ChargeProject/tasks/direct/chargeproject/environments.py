from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

"""
ROUGH_TERRAIN_CFG: terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=2.0,
    num_rows=20,
    num_cols=20,
    # horizontal_scale=0.1,
    # vertical_scale=0.005,
    # slope_threshold=0.5,
    # use_cache=False,
    color_scheme="height",
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=100, patch_radius=1.0, max_height_diff=0.5
                ),
            },
        ),
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.7,
            flat_patch_sampling={
                "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=100, patch_radius=1.0, max_height_diff=0.5
                ),
            },
        ),
    },
)


SMOOTH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(20.0, 20.0),
    # border_width=10.0,
    num_rows=1,
    num_cols=1,
    # horizontal_scale=0.1,
    # vertical_scale=0.005,
    # slope_threshold=0.5,
    # use_cache=False,
    color_scheme="random",
    sub_terrains={
        "smooth_hill": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(-0.5, 1.5),
            num_waves=2,
        ),
        # "plane": terrain_gen.MeshPlaneTerrainCfg(
        #     proportion=0.2,
        #     size=(2,2),
        #     flat_patch_sampling={
        #         "robot_spawn": terrain_gen.FlatPatchSamplingCfg(
        #             num_patches=100, patch_radius=1.0, max_height_diff=0.5
        #         ),
        #     },
        # ),
    },
)

ROBOT_CFG = UNITREE_GO2_CFG.replace(  # type: ignore
    prim_path="/World/envs/env_.*/Robot",
    init_state=UNITREE_GO2_CFG.init_state.replace(  # type: ignore
        pos=(0.0, 0.0, 0.5),
    ),
)

ROUGH_TERRAINS_CFG_TEMP: terrain_gen.TerrainGeneratorCfg = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)

TMPTMP = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG_TEMP,
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

@configclass
class MySceneCfg(InteractiveSceneCfg):
    num_envs = int(1024*32*4)
    env_spacing = 4.0
    replicate_physics = True

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG, #.replace(num_rows=10, num_cols=10),
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

    """
