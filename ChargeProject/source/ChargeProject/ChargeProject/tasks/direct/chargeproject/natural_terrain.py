import numpy as np
import trimesh
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg
import torch
from isaaclab.utils import configclass
from dataclasses import MISSING

def smooth_slope(difficulty: float, cfg: "SmoothTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")

    meshes_list = list()
    grid_width = cfg.grid_width
    
    width_m, length_m = cfg.size[0], cfg.size[1]
    
    # Calculate grid start position
    start_pos_x = (width_m / 2.0) - (cfg.spacing_m / 2.0)
    start_pos_y = (length_m / 2.0) - (cfg.spacing_m / 2.0)

    # Create spawn grid
    spawn_x = np.linspace(start_pos_x, start_pos_x + cfg.spacing_m * (cfg.num_spawns_per_side - 1), cfg.num_spawns_per_side)
    spawn_y = np.linspace(start_pos_y, start_pos_y + cfg.spacing_m * (cfg.num_spawns_per_side - 1), cfg.num_spawns_per_side)
    spawn_xx, spawn_yy = np.meshgrid(spawn_x, spawn_y)
    
    spawn_x_flat = spawn_xx.flatten()
    spawn_y_flat = spawn_yy.flatten()

    # Calculate the "Natural" height at these spawn points (before flattening)
    # We use this to determine the height of the platform
    spawn_z_ground_natural = (
        (spawn_x_flat * cfg.slope_grade) + 
        (np.sin(spawn_x_flat * cfg.wave_frequency_x) * cfg.wave_amplitude) + 
        (np.cos(spawn_y_flat * cfg.wave_frequency_y) * cfg.wave_amplitude)
    )


    num_boxes_x = int(cfg.size[0] / grid_width)
    num_boxes_y = int(cfg.size[1] / grid_width)
    
    # Create template box
    grid_dim = [grid_width, grid_width, cfg.terrain_height]
    grid_position = [0.5 * grid_width, 0.5 * grid_width, -cfg.terrain_height / 2]
    
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    template_vertices = template_box.vertices # (8, 3)
    
    # Create Grid of boxes
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    
    x_coords = torch.arange(0, num_boxes_x, device=device)
    y_coords = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
    xx_yy = torch.cat((xx.flatten().view(-1, 1), yy.flatten().view(-1, 1)), dim=1)
    offsets = grid_width * xx_yy
    vertices[:, :, :2] += offsets.unsqueeze(1)
    
    # Slope
    mask_top = vertices[:, :, 2] > -0.1
    top_x = vertices[:, :, 0][mask_top]
    top_y = vertices[:, :, 1][mask_top]
    
    slope_z = (
        (top_x * cfg.slope_grade) + 
        (torch.sin(top_x * cfg.wave_frequency_x) * cfg.wave_amplitude) + 
        (torch.cos(top_y * cfg.wave_frequency_y) * cfg.wave_amplitude)
    )
    
    # Apply natural slope to all top vertices
    vertices[:, :, 2][mask_top] += slope_z
    
    # Platforms
    half_plat = cfg.platform_width / 2.0
    
    for i in range(len(spawn_x_flat)):
        sx = spawn_x_flat[i]
        sy = spawn_y_flat[i]
        target_z = spawn_z_ground_natural[i] # Height of the platform
        
        mask_platform = (
            (vertices[:, :, 0] >= sx - half_plat) & 
            (vertices[:, :, 0] <= sx + half_plat) & 
            (vertices[:, :, 1] >= sy - half_plat) & 
            (vertices[:, :, 1] <= sy + half_plat) &
            mask_top
        )
        
        # Overwrite the Z value of these vertices to be perfectly flat
        vertices[:, :, 2][mask_platform] = float(target_z)

    # Create mesh
    vertices_np = vertices.reshape(-1, 3).cpu().numpy()
    faces = torch.tensor(template_box.faces, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    face_offsets = torch.arange(0, num_boxes_x * num_boxes_y, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces_np = faces.view(-1, 3).cpu().numpy()

    ground_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    meshes_list = [ground_mesh]

    # Add Random Blocks
    rng = np.random.default_rng(seed=cfg.seed)

    for _ in range(cfg.num_blocks):
        sx = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sy = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sz = rng.uniform(cfg.block_height_min, cfg.block_height_max)

        pos_x = rng.uniform(2.0, width_m - 2.0)
        pos_y = rng.uniform(2.0, length_m - 2.0)
        
        # Skip if on platform
        is_on_platform = False
        for i in range(len(spawn_x_flat)):
            dist_x = abs(pos_x - spawn_x_flat[i])
            dist_y = abs(pos_y - spawn_y_flat[i])
            if dist_x < (cfg.platform_width / 2 + sx/2) and dist_y < (cfg.platform_width / 2 + sy/2):
                is_on_platform = True
                break
        
        if is_on_platform:
            continue # Skip this block so platforms are clear

        # Calculate ground height for block
        ground_z = (
            (pos_x * cfg.slope_grade) + 
            (np.sin(pos_x * cfg.wave_frequency_x) * cfg.wave_amplitude) + 
            (np.cos(pos_y * cfg.wave_frequency_y) * cfg.wave_amplitude)
        )
        
        pos_z = ground_z + (sz / 2.0)
        
        box = trimesh.creation.box(extents=(sx, sy, sz))
        transform = np.eye(4)
        angle = rng.uniform(0, 2 * np.pi)
        rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        transform[:3, :3] = rot_matrix[:3, :3]
        transform[:3, 3] = [pos_x, pos_y, pos_z]
        
        box.apply_transform(transform)
        meshes_list.append(box)

    # Spawn Origins
    
    # Add the robot offset (feet to center of base)
    spawn_z_total = spawn_z_ground_natural

    # Offset spawns to center
    spawn_x_centered = spawn_x_flat# - (width_m / 2.0)
    spawn_y_centered = spawn_y_flat# - (length_m / 2.0)

    # Create the Nx3 array for return
    origins = np.stack([spawn_x_centered, spawn_y_centered, spawn_z_total], axis=1)
    
    # Global store spawn positions
    SmoothTerrainCfg.spawns_positions = torch.tensor(origins, device=device, dtype=torch.float32)
    
    return meshes_list, np.zeros(3)


# Define a config for your specific sub-terrain settings
@configclass
class SmoothTerrainCfg(HfTerrainBaseCfg):
    # --- Terrain Geometry and Resolution ---
    grid_width: float = 0.25         # Resolution of the underlying solid grid (m). Must be small.
    terrain_height: float = 1.0      # Thickness of the terrain volume (m) for collision reliability.

    # --- Slope and Wave Parameters (Controls the "Smoothness") ---
    # Z = slope_grade * X + wave_amp * sin(freq * X) + ...
    slope_grade: float = 0.05        # Steepness of the linear incline (e.g., 0.05 = 5% grade)
    wave_amplitude: float = 0.3      # Amplitude of the sine/cosine waves (m)
    wave_frequency_x: float = 0.4    # Frequency multiplier for X-axis waves
    wave_frequency_y: float = 0.6    # Frequency multiplier for Y-axis waves

    # --- Block Obstacle Parameters ---
    num_blocks: int = 80             # Total number of random blocks
    block_size_min: float = 0.5      # Minimum block side length (m)
    block_size_max: float = 1.5      # Maximum block side length (m)
    block_height_min: float = 1.0    # Minimum block height (m)
    block_height_max: float = 2.5    # Maximum block height (m)

    # --- Spawn Parameters ---
    num_spawns_per_side = 1
    spacing_m = 10.0  # Distance between robot centers (meters)
    # Spawns are stored here statically after generation
    spawns_positions: np.ndarray = None

    platform_width: float = 1.5  # Width of the flat spawn platform in meters

# Main Generator Config
terrain_gen_cfg = TerrainGeneratorCfg(
    seed=42,
    num_rows=1,
    num_cols=1,
    size=(50.0, 50.0),
    sub_terrains={
        "main": SmoothTerrainCfg(
            function=smooth_slope,
            proportion=1.0, 
        )
    }
)

terrain_spawn_origins = None
