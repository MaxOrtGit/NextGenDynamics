import numpy as np
import trimesh
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg
import torch
from isaaclab.utils import configclass

def smooth_slope(difficulty: float, cfg: "SmoothTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")

    meshes_list = list()
    grid_width = cfg.grid_width
    
    num_boxes_x = int(cfg.size[0] / grid_width)
    num_boxes_y = int(cfg.size[1] / grid_width)
    
    # Create template box (base stays at -terrain_height/2 to -terrain_height/2)
    grid_dim = [grid_width, grid_width, cfg.terrain_height]
    grid_position = [0.5 * grid_width, 0.5 * grid_width, -cfg.terrain_height / 2]
    
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    template_vertices = template_box.vertices
    
    # Repeat and offset vertices (standard method to create the grid)
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    # ... (x, y meshgrid creation and offset calculation remains the same) ...
    x_coords = torch.arange(0, num_boxes_x, device=device)
    y_coords = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
    xx_yy = torch.cat((xx.flatten().view(-1, 1), yy.flatten().view(-1, 1)), dim=1)
    offsets = grid_width * xx_yy
    vertices[:, :, :2] += offsets.unsqueeze(1)
    
    # Smooth Slope Modification
    mask_top = vertices[:, :, 2] > -0.1
    top_x = vertices[:, :, 0][mask_top]
    top_y = vertices[:, :, 1][mask_top]
    
    # Slope func
    slope_z = (
        (top_x * cfg.slope_grade) + 
        (torch.sin(top_x * cfg.wave_frequency_x) * cfg.wave_amplitude) + 
        (torch.cos(top_y * cfg.wave_frequency_y) * cfg.wave_amplitude)
    )
    
    # Update Z-coordinates of the top vertices
    vertices[:, :, 2][mask_top] += slope_z
    
    # Finalize ground mesh (vertices_np, faces_np creation remains the same)
    vertices_np = vertices.reshape(-1, 3).cpu().numpy()
    faces = torch.tensor(template_box.faces, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    face_offsets = torch.arange(0, num_boxes_x * num_boxes_y, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces_np = faces.view(-1, 3).cpu().numpy()

    ground_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    meshes_list = [ground_mesh]

    # Add scattered blocks
    rng = np.random.default_rng(seed=cfg.seed)
    width_m, length_m = cfg.size[0], cfg.size[1]

    for _ in range(cfg.num_blocks):
        sx = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sy = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sz = rng.uniform(cfg.block_height_min, cfg.block_height_max)

        box = trimesh.creation.box(extents=(sx, sy, sz))
        pos_x = rng.uniform(2.0, width_m - 2.0)
        pos_y = rng.uniform(2.0, length_m - 2.0)
        
        ground_z = (
            (pos_x * cfg.slope_grade) + 
            (np.sin(pos_x * cfg.wave_frequency_x) * cfg.wave_amplitude) + 
            (np.cos(pos_y * cfg.wave_frequency_y) * cfg.wave_amplitude)
        )
        
        pos_z = ground_z + (sz / 2.0) # Position block center
        
        # Transform (rotation and translation)
        transform = np.eye(4)
        angle = rng.uniform(0, 2 * np.pi)
        rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        transform[:3, :3] = rot_matrix[:3, :3]
        transform[:3, 3] = [pos_x, pos_y, pos_z]
        
        box.apply_transform(transform)
        meshes_list.append(box)

    center_x = width_m / 2.0
    center_y = length_m / 2.0
    
    # Height at center
    center_z = (
        (center_x * cfg.slope_grade) + 
        (np.sin(center_x * cfg.wave_frequency_x) * cfg.wave_amplitude) + 
        (np.cos(center_y * cfg.wave_frequency_y) * cfg.wave_amplitude)
    )
    
    # Spawn offset from CFG
    origin = np.array([center_x, center_y, center_z + cfg.spawn_offset_z])

    return meshes_list, origin


# Define a config for your specific sub-terrain settings
@configclass
class SmoothTerrainCfg(HfTerrainBaseCfg):
    # --- Terrain Geometry and Resolution ---
    grid_width: float = 0.25         # Resolution of the underlying solid grid (m). Must be small.
    terrain_height: float = 1.0      # Thickness of the terrain volume (m) for collision reliability.

    # --- Slope and Wave Parameters (Controls the "Smoothness") ---
    # Z = slope_grade * X + wave_amp * sin(freq * X) + ...
    slope_grade: float = 0.075        # Steepness of the linear incline (e.g., 0.05 = 5% grade)
    wave_amplitude: float = 0.6      # Amplitude of the sine/cosine waves (m)
    wave_frequency_x: float = 0.2    # Frequency multiplier for X-axis waves
    wave_frequency_y: float = 0.3    # Frequency multiplier for Y-axis waves

    # --- Block Obstacle Parameters ---
    num_blocks: int = 80             # Total number of random blocks
    block_size_min: float = 0.5      # Minimum block side length (m)
    block_size_max: float = 1.5      # Maximum block side length (m)
    block_height_min: float = 1.0    # Minimum block height (m)
    block_height_max: float = 2.5    # Maximum block height (m)

    # --- Spawn Parameter ---
    spawn_offset_z: float = 0.1      # Height above the ground surface for robot spawn (m)

# Main Generator Config
terrain_gen_cfg = TerrainGeneratorCfg(
    seed=42,
    num_rows=1,
    num_cols=1,
    size=(50.0, 50.0),
    sub_terrains={
        "my_mesh_terrain": SmoothTerrainCfg(
            function=smooth_slope,
            proportion=1.0, 
        )
    }
)