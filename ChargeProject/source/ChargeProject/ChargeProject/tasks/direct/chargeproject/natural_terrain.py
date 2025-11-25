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
    
    # --- 1. SETUP & CONSTANTS ---
    # Resolution of the ground mesh (meters). 
    # 0.1 provides steep "walls" for steps which physics engines like.
    res = 0.1 
    
    width_m, length_m = cfg.size[0], cfg.size[1]
    nx = int(width_m / res)
    ny = int(length_m / res)
    
    # Center offsets for noise calculations
    x_center_offset = (width_m / 2.0)
    y_center_offset = (length_m / 2.0)

    # --- 2. NOISE HELPERS (Vectorized for Mesh, Single for Blocks) ---
    v_pnoise = np.vectorize(pnoise2)

    def get_raw_height_noise(x_vals, y_vals):
        """Calculates base terrain height (hills/valleys) at specific coordinates."""
        return v_pnoise(
            x_vals * cfg.noise_scale, 
            y_vals * cfg.noise_scale, 
            octaves=cfg.noise_octaves, 
            persistence=cfg.noise_persistence, 
            lacunarity=cfg.noise_lacunarity, 
            repeatx=1024, repeaty=1024, base=cfg.noise_seed
        ) * cfg.noise_height_scale

    def get_biome_at_points(x_vals, y_vals):
        """Returns the biome index and weight for given coordinates."""
        num_p = len(x_vals)
        scores = np.zeros((len(cfg.biomes), num_p))

        for i, biome in enumerate(cfg.biomes):
            noise_val = v_pnoise(
                x_vals * cfg.biome_blend_scale, 
                y_vals * cfg.biome_blend_scale, 
                octaves=1, 
                repeatx=1024, repeaty=1024, base=cfg.noise_seed + ((1+i) * 500)
            )
            scores[i] = (noise_val + 1.0) * biome.weight
        
        return np.argmax(scores, axis=0)

    # --- 3. GENERATE GROUND MESH ---
    # Create Grid
    x = torch.linspace(0, width_m, nx, device=device)
    y = torch.linspace(0, length_m, ny, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    
    # Shift to noise coordinates (centered)
    x_np = (x_flat - x_center_offset).cpu().numpy()
    y_np = (y_flat - y_center_offset).cpu().numpy()

    # Calculate Terrain Heights
    raw_z = get_raw_height_noise(x_np, y_np)
    winning_biomes = get_biome_at_points(x_np, y_np)
    final_z = np.zeros_like(raw_z)

    # Apply Steps vs Smooth logic
    for i, biome in enumerate(cfg.biomes):
        mask = (winning_biomes == i)
        if not np.any(mask): continue
        z_chunk = raw_z[mask]
        
        if biome.step_size > 0.001:
            final_z[mask] = np.floor(z_chunk / biome.step_size) * biome.step_size
        else:
            final_z[mask] = z_chunk

    # --- 4. FLATTEN SPAWN PLATFORMS ---
    # Define spawn grid
    start_pos_x = (width_m / 2.0) - ((cfg.num_spawns_per_side - 1) * cfg.spacing_m / 2.0)
    start_pos_y = (length_m / 2.0) - ((cfg.num_spawns_per_side - 1) * cfg.spacing_m / 2.0)
    
    spawn_grid_x = np.linspace(start_pos_x, start_pos_x + cfg.spacing_m * (cfg.num_spawns_per_side - 1), cfg.num_spawns_per_side)
    spawn_grid_y = np.linspace(start_pos_y, start_pos_y + cfg.spacing_m * (cfg.num_spawns_per_side - 1), cfg.num_spawns_per_side)
    
    # We flatten the ground mesh at these locations
    half_plat = cfg.platform_width / 2.0
    spawn_origins_list = [] # For Isaac Lab config

    # Flatten mesh loops
    for sx in spawn_grid_x:
        for sy in spawn_grid_y:
            # Map spawn world pos -> Noise pos
            sx_noise = sx - x_center_offset
            sy_noise = sy - y_center_offset
            
            # Distance check against all mesh points (Optimized via mask)
            dx = np.abs(x_np - sx_noise)
            dy = np.abs(y_np - sy_noise)
            
            dist_mask = (dx < half_plat) & (dy < half_plat)
            
            if np.any(dist_mask):
                # Flatten this area to the average height
                center_val = np.mean(final_z[dist_mask]) 
                final_z[dist_mask] = center_val
                
                # Save for the robot spawn config later
                # z + 0.5 so the robot drops slightly
                spawn_origins_list.append([sx, sy, center_val]) 

    # --- 5. BUILD TERRAIN MESH ---
    # Vertices (x_flat is 0..width, y_flat is 0..length)
    vertices = np.stack([x_flat.cpu().numpy(), y_flat.cpu().numpy(), final_z], axis=1)

    # Faces (Grid Topology)
    ids = np.arange(nx * ny).reshape(nx, ny)
    f1 = np.stack([ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:]], axis=2).reshape(-1, 3)
    f2 = np.stack([ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]], axis=2).reshape(-1, 3)
    faces = np.vstack([f1, f2])

    ground_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    ground_mesh.fix_normals()
    
    meshes_list = [ground_mesh]

    # --- 6. GENERATE BLOCKS (The requested part) ---
    rng = np.random.default_rng(seed=cfg.seed)
    
    # Flatten spawn list for fast distance checking
    spawn_origins_arr = np.array(spawn_origins_list) if len(spawn_origins_list) > 0 else np.empty((0,3))

    for _ in range(cfg.num_blocks):
        # Random Size
        sx = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sy = rng.uniform(cfg.block_size_min, cfg.block_size_max)
        sz = rng.uniform(cfg.block_height_min, cfg.block_height_max)
        
        # Random Position (World Coords)
        pos_x = rng.uniform(2.0, width_m - 2.0)
        pos_y = rng.uniform(2.0, length_m - 2.0)

        # 6a. Check Platform Collision
        # We don't want to block the spawn pads
        is_on_platform = False
        if len(spawn_origins_arr) > 0:
            # Vectorized distance check against all spawn points
            dists_x = np.abs(pos_x - spawn_origins_arr[:, 0])
            dists_y = np.abs(pos_y - spawn_origins_arr[:, 1])
            # If inside any platform box
            if np.any((dists_x < (half_plat + sx)) & (dists_y < (half_plat + sy))):
                is_on_platform = True
        
        if is_on_platform: 
            continue 

        # 6b. Calculate Height at this specific spot
        # We must transform World Coords -> Noise Coords
        pos_x_noise = pos_x - x_center_offset
        pos_y_noise = pos_y - y_center_offset
        
        # Re-run the biome logic for this single point to get exact ground height
        # This ensures the block sits perfectly on steps or smooth slopes
        b_idx = get_biome_at_points(np.array([pos_x_noise]), np.array([pos_y_noise]))[0]
        biome = cfg.biomes[b_idx]
        raw_z_block = get_raw_height_noise(np.array([pos_x_noise]), np.array([pos_y_noise]))[0]
        
        if biome.step_size > 0.001:
            ground_z = np.floor(raw_z_block / biome.step_size) * biome.step_size
        else:
            ground_z = raw_z_block

        # 6c. Create and Position Block
        # Center of box Z = ground_z + half height
        # Note: If you want blocks slightly sunken to prevent bottom gaps, subtract 0.1
        final_block_z = ground_z + (sz / 2.0) - 0.05 
        
        box = trimesh.creation.box(extents=(sx, sy, sz))
        
        # Transform
        transform = np.eye(4)
        rot_matrix = trimesh.transformations.rotation_matrix(rng.uniform(0, 2 * np.pi), [0, 0, 1])
        transform[:3, :3] = rot_matrix[:3, :3]
        transform[:3, 3] = [pos_x, pos_y, final_block_z]
        
        box.apply_transform(transform)
        meshes_list.append(box)

    # --- 7. FINALIZE ---
    if len(spawn_origins_list) > 0:
        MultiBiomeTerrainCfg.spawns_positions = torch.tensor(spawn_origins_list, device=device, dtype=torch.float32)
    else:
        # Fallback if map is tiny
        MultiBiomeTerrainCfg.spawns_positions = torch.zeros((1,3), device=device)

    return meshes_list, np.zeros(3)

@configclass
class MultiBiomeTerrainCfg(HfTerrainBaseCfg):
    grid_width: float = 0.25         
    terrain_height: float = 5.0 # needs to be high enough for noise range 

    # --- Terrain Shape (The Geometry) ---
    noise_seed: int = 123
    noise_scale: float = 0.05       # Frequency of the Perlin noise (higher = more hills/valleys)
    noise_height_scale: float = 2.5 # Amplitude of the Perlin noise
    noise_octaves: int = 5
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0

    # --- Biome Distribution Settings ---
    biome_blend_scale: float = 0.10   # Higher = Choppier transitions. Lower = Larger continents.
    
    # --- THE BIOME LIST ---
    biomes: List[BiomeCfg] = field(default_factory=lambda: [
        BiomeCfg(weight=1.1, step_size=0.0), # Smooth
        BiomeCfg(weight=1.0, step_size=0.05),
        BiomeCfg(weight=1.0, step_size=0.1),
        BiomeCfg(weight=0.9, step_size=0.2),
        #BiomeCfg(weight=0.8, step_size=0.3),
        BiomeCfg(weight=0.7, step_size=0.3), # Giant cliffs
    ])

    # --- Objects ---
    num_blocks: int = 500             
    block_size_min: float = 0.5; block_size_max: float = 1.5      
    block_height_min: float = 1.0; block_height_max: float = 2.5    

    num_spawns_per_side = 5
    spacing_m = 20.0  
    spawns_positions: np.ndarray = None
    platform_width: float = 1.75  

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