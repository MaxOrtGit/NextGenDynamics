import numpy as np
import trimesh
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg
import torch
from isaaclab.utils import configclass
from dataclasses import dataclass, field
from typing import List
from noise import pnoise2

@dataclass
class BiomeCfg:
    """Defines a specific terrain style."""
    weight: float = 1.0       # The "strength" of this biome in the competition
    step_size: float = 0.0    # 0.0 = Smooth. >0.0 = Stepped height.

def multi_biome_terrain(difficulty: float, cfg: "MultiBiomeTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if cfg.size[0] != cfg.size[1]:
        raise ValueError(f"The terrain must be square. Received size: {cfg.size}.")

    meshes_list = list()
    grid_width = cfg.grid_width
    width_m, length_m = cfg.size[0], cfg.size[1]
    
    # Grid setup
    start_pos_x = (width_m / 2.0) - ((cfg.num_spawns_per_side - 1) * cfg.spacing_m / 2.0)
    start_pos_y = (length_m / 2.0) - ((cfg.num_spawns_per_side - 1) * cfg.spacing_m / 2.0)

    # --- NOISE HELPERS ---
    v_pnoise = np.vectorize(pnoise2)

    def get_raw_height_noise(x_vals, y_vals):
        """Standard Perlin noise for the underlying geometry (Hills/Valleys)"""
        return v_pnoise(
            x_vals * cfg.noise_scale, 
            y_vals * cfg.noise_scale, 
            octaves=cfg.noise_octaves, 
            persistence=cfg.noise_persistence, 
            lacunarity=cfg.noise_lacunarity, 
            repeatx=1024, repeaty=1024, base=cfg.noise_seed
        ) * cfg.noise_height_scale

    def get_biome_indices(x_vals, y_vals):
        """
        DETERMINES THE WINNING BIOME FOR EACH POINT.
        Strategy: ArgMax Competition.
        We generate a noise value for EACH biome type. 
        The biome with the highest (Noise * Weight) at that specific x,y wins.
        """
        # Array to store scores: Shape (Num_Biomes, Num_Points)
        num_points = len(x_vals)
        scores = np.zeros((len(cfg.biomes), num_points))

        for i, biome in enumerate(cfg.biomes):
            # Generate a unique noise map for this biome
            # We offset the seed by 'i * 500' so they are completely uncorrelated
            noise_val = v_pnoise(
                x_vals * cfg.biome_blend_scale, 
                y_vals * cfg.biome_blend_scale, 
                octaves=1, # Keep blend noise simple/fast
                repeatx=1024, repeaty=1024, base=cfg.noise_seed + ((1+i) * 500)
            )
            
            # Normalize noise from [-1, 1] to [0, 1] roughly, so weights act as multipliers
            # Adding 1.0 makes it [0, 2].
            positive_noise = noise_val + 1.0
            
            # Calculate Score
            scores[i] = positive_noise * biome.weight

        # Returns the index (0 to N-1) of the winning biome for each point
        return np.argmax(scores, axis=0)
    
    # ---------------------

    # Setup Spawns
    spawn_x = np.linspace(start_pos_x, start_pos_x + cfg.spacing_m * (cfg.num_spawns_per_side - 1), cfg.num_spawns_per_side)
    spawn_y = np.linspace(start_pos_y, start_pos_y + cfg.spacing_m * (cfg.num_spawns_per_side - 1), cfg.num_spawns_per_side)
    spawn_xx, spawn_yy = np.meshgrid(spawn_x, spawn_y)
    spawn_x_flat = spawn_xx.flatten()
    spawn_y_flat = spawn_yy.flatten()

    # Calculate Spawn Heights
    spawn_raw_z = get_raw_height_noise(spawn_x_flat, spawn_y_flat)
    spawn_biome_indices = get_biome_indices(spawn_x_flat, spawn_y_flat)
    
    spawn_z_final = np.zeros_like(spawn_raw_z)
    
    # Apply biome logic to spawns
    for i in range(len(spawn_x_flat)):
        b_idx = spawn_biome_indices[i]
        biome = cfg.biomes[b_idx]
        
        if biome.step_size > 0.001:
            # Stepped
            spawn_z_final[i] = np.floor(spawn_raw_z[i] / biome.step_size) * biome.step_size
        else:
            # Smooth
            spawn_z_final[i] = spawn_raw_z[i]

    # Generate Grid Boxes
    num_boxes_x = int(cfg.size[0] / grid_width)
    num_boxes_y = int(cfg.size[1] / grid_width)
    
    grid_dim = [grid_width, grid_width, cfg.terrain_height]
    grid_position = [0.5 * grid_width, 0.5 * grid_width, -cfg.terrain_height / 2]
    
    template_box = trimesh.creation.box(grid_dim, trimesh.transformations.translation_matrix(grid_position))
    template_vertices = template_box.vertices 
    
    vertices = torch.tensor(template_vertices, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    
    x_coords = torch.arange(0, num_boxes_x, device=device)
    y_coords = torch.arange(0, num_boxes_y, device=device)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing="ij")
    xx_yy = torch.cat((xx.flatten().view(-1, 1), yy.flatten().view(-1, 1)), dim=1)
    offsets = grid_width * xx_yy
    vertices[:, :, :2] += offsets.unsqueeze(1)
    
    # --- APPLY BIOME HEIGHTS ---
    
    # Calculate Box Centers (for Biome Selection + Stepped Heights)
    box_centers = vertices.mean(dim=1)[:, :2].cpu().numpy()
    
    # Calculate Raw Height at Center
    raw_z_centers = get_raw_height_noise(box_centers[:, 0], box_centers[:, 1])
    
    # Determine Biome at Center
    # We use the box center to decide the biome for the whole box (avoids jagged box-splits)
    winning_indices = get_biome_indices(box_centers[:, 0], box_centers[:, 1])
    
    # Calculate Smooth Heights per Vertex (for Smooth Biomes)
    all_vertices_cpu = vertices.reshape(-1, 3)[:, :2].cpu().numpy()
    raw_z_vertices = get_raw_height_noise(all_vertices_cpu[:, 0], all_vertices_cpu[:, 1])
    raw_z_vertices = raw_z_vertices.reshape(-1, 8) # Shape back to (N, 8)
    
    # Construct Final Height Map
    final_z_vals = np.zeros_like(raw_z_vertices) # (N, 8)
    
    # Iterate through unique biomes to vectorize the assignment
    for b_idx, biome in enumerate(cfg.biomes):
        # Mask: Which boxes belong to this biome?
        mask_indices = (winning_indices == b_idx) # Boolean array of shape (N,)
        
        if not np.any(mask_indices):
            continue

        if biome.step_size > 0.001:
            # STEPPED: Use Center Z, quantize it, apply to all 8 verts
            z_centered = raw_z_centers[mask_indices]
            z_stepped = np.floor(z_centered / biome.step_size) * biome.step_size
            # Broadcast (M,) -> (M, 8)
            final_z_vals[mask_indices, :] = z_stepped[:, np.newaxis]
        else:
            # SMOOTH: Use per-vertex Z
            final_z_vals[mask_indices, :] = raw_z_vertices[mask_indices, :]

    final_z_torch = torch.tensor(final_z_vals, device=device, dtype=torch.float32).view(-1, 8)
    mask_top = vertices[:, :, 2] > -0.1
    vertices[:, :, 2] += final_z_torch * mask_top.float()

    # Platforms
    half_plat = cfg.platform_width / 2.0
    
    for i in range(len(spawn_x_flat)):
        sx = spawn_x_flat[i]
        sy = spawn_y_flat[i]
        target_z = spawn_z_final[i]
        
        mask_platform = (
            (vertices[:, :, 0] >= sx - half_plat) & 
            (vertices[:, :, 0] <= sx + half_plat) & 
            (vertices[:, :, 1] >= sy - half_plat) & 
            (vertices[:, :, 1] <= sy + half_plat) &
            mask_top
        )
        vertices[:, :, 2][mask_platform] = float(target_z)

    # Mesh
    vertices_np = vertices.reshape(-1, 3).cpu().numpy()
    faces = torch.tensor(template_box.faces, device=device).repeat(num_boxes_x * num_boxes_y, 1, 1)
    face_offsets = torch.arange(0, num_boxes_x * num_boxes_y, device=device).unsqueeze(1).repeat(1, 12) * 8
    faces += face_offsets.unsqueeze(2)
    faces_np = faces.view(-1, 3).cpu().numpy()

    ground_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    meshes_list = [ground_mesh]

    # --- BLOCKS (With Biome Awareness) ---
    rng = np.random.default_rng(seed=cfg.seed)
    
    for _ in range(cfg.num_blocks):
        sx = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sy = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sz = rng.uniform(cfg.block_height_min, cfg.block_height_max)
        pos_x = rng.uniform(2.0, width_m - 2.0)
        pos_y = rng.uniform(2.0, length_m - 2.0)
        
        # Check Platform
        is_on_platform = False
        for i in range(len(spawn_x_flat)):
            dist_x = abs(pos_x - spawn_x_flat[i])
            dist_y = abs(pos_y - spawn_y_flat[i])
            if dist_x < (cfg.platform_width / 2 + sx/2) and dist_y < (cfg.platform_width / 2 + sy/2):
                is_on_platform = True; break
        if is_on_platform: continue 

        # Determine Height based on Biome
        b_idx = get_biome_indices(np.array([pos_x]), np.array([pos_y]))[0]
        biome = cfg.biomes[b_idx]
        raw_z = get_raw_height_noise(np.array([pos_x]), np.array([pos_y]))[0]
        
        if biome.step_size > 0.001:
            ground_z = np.floor(raw_z / biome.step_size) * biome.step_size
        else:
            ground_z = raw_z

        pos_z = ground_z + (sz / 2.0)
        
        box = trimesh.creation.box(extents=(sx, sy, sz))
        transform = np.eye(4)
        rot_matrix = trimesh.transformations.rotation_matrix(rng.uniform(0, 2 * np.pi), [0, 0, 1])
        transform[:3, :3] = rot_matrix[:3, :3]
        transform[:3, 3] = [pos_x, pos_y, pos_z]
        box.apply_transform(transform)
        meshes_list.append(box)

    origins = np.stack([spawn_x_flat, spawn_y_flat, spawn_z_final], axis=1)
    MultiBiomeTerrainCfg.spawns_positions = torch.tensor(origins, device=device, dtype=torch.float32)
    
    return meshes_list, np.zeros(3)

@configclass
class MultiBiomeTerrainCfg(HfTerrainBaseCfg):
    grid_width: float = 0.25         
    terrain_height: float = 5.0 # needs to be high enough for noise range 

    # --- Terrain Shape (The Geometry) ---
    noise_seed: int = 1234
    noise_scale: float = 0.1       
    noise_height_scale: float = 2.5
    noise_octaves: int = 5
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0

    # --- Biome Distribution Settings ---
    biome_blend_scale: float = 0.10   # Higher = Choppier transitions. Lower = Larger continents.
    
    # --- THE BIOME LIST ---
    biomes: List[BiomeCfg] = field(default_factory=lambda: [
        BiomeCfg(weight=1.1, step_size=0.0), # Smooth
        BiomeCfg(weight=1.0, step_size=0.1),
        BiomeCfg(weight=0.9, step_size=0.2),
        BiomeCfg(weight=0.8, step_size=0.3),
        BiomeCfg(weight=0.7, step_size=0.5), # Giant cliffs
    ])

    # --- Objects ---
    num_blocks: int = 800             
    block_size_min: float = 0.5; block_size_max: float = 1.5      
    block_height_min: float = 1.0; block_height_max: float = 2.5    

    num_spawns_per_side = 5
    spacing_m = 20.0  
    spawns_positions: np.ndarray = None
    platform_width: float = 1.5  

terrain_gen_cfg = TerrainGeneratorCfg(
    seed=42,
    num_rows=1, num_cols=1, size=(150.0, 150.0),
    sub_terrains={
        "main": MultiBiomeTerrainCfg(
            function=multi_biome_terrain,
            proportion=1.0, 
        )
    }
)
terrain_spawn_origins = None