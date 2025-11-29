import numpy as np
import trimesh
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains import TerrainGeneratorCfg
import torch
from isaaclab.utils import configclass
from dataclasses import dataclass, field
from typing import List, Tuple
from noise import pnoise2

@dataclass
class BiomeCfg:
    """Defines a specific terrain style (Smooth vs Stepped)."""
    weight: float = 1.0       
    step_size: float = 0.0    # 0.0 = Smooth slopes. >0.0 = Flat steps.

@dataclass
class BlockCfg:
    """Defines a category of blocks (e.g. Debris vs Large Obstacles)."""
    weight: float = 1.0
    # (min, max) for base dimensions (X and Y)
    width_range: Tuple[float, float] = (0.5, 1.0) 
    # (min, max) for height STICKING OUT of ground
    height_range: Tuple[float, float] = (0.2, 0.5)
    # (min, max) multiplier for both width and height
    scale_range: Tuple[float, float] = (1.0, 1.0)
    # How deep the object goes into the ground (meters)
    burial_depth: float = 4.0

def multi_biome_terrain(difficulty: float, cfg: "MultiBiomeTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # --- 1. SETUP (OPTIMIZED) ---
    res = cfg.grid_width 
    width_m, length_m = cfg.size[0], cfg.size[1]
    
    nx = int(np.ceil(width_m / res))
    ny = int(np.ceil(length_m / res))
    
    x_center_offset = (width_m / 2.0)
    y_center_offset = (length_m / 2.0)

    # --- 2. GENERATE RAW NOISE GRID (VERTEX CENTERED) ---
    x_verts = np.linspace(0, width_m, nx + 1)
    y_verts = np.linspace(0, length_m, ny + 1)
    xx_v, yy_v = np.meshgrid(x_verts, y_verts, indexing="ij")
    
    xx_noise = xx_v - x_center_offset
    yy_noise = yy_v - y_center_offset

    v_pnoise = np.vectorize(pnoise2)
    
    # Base Terrain Height
    raw_z_grid = v_pnoise(
        xx_noise * cfg.noise_scale, 
        yy_noise * cfg.noise_scale, 
        octaves=cfg.noise_octaves, 
        persistence=cfg.noise_persistence, 
        lacunarity=cfg.noise_lacunarity, 
        repeatx=1024, repeaty=1024, base=cfg.noise_seed
    ) * cfg.noise_height_scale

    # Biome Map (Cell Centered)
    xx_c = (xx_v[:-1, :-1] + xx_v[1:, 1:]) * 0.5
    yy_c = (yy_v[:-1, :-1] + yy_v[1:, 1:]) * 0.5
    xx_c_noise = xx_c - x_center_offset
    yy_c_noise = yy_c - y_center_offset

    scores = np.zeros((len(cfg.biomes), xx_c.size))
    xf = xx_c_noise.ravel(); yf = yy_c_noise.ravel()
    
    for i, b in enumerate(cfg.biomes):
        nv = v_pnoise(xf*cfg.biome_blend_scale, yf*cfg.biome_blend_scale, octaves=1, 
                      repeatx=1024, repeaty=1024, base=cfg.noise_seed + ((i+1)*500))
        scores[i] = (nv + 1.0) * b.weight
    
    biome_indices = np.argmax(scores, axis=0).reshape(nx, ny)
    
    is_stepped = np.zeros((nx, ny), dtype=bool)
    step_sizes = np.zeros((nx, ny), dtype=np.float32)
    
    for i, b in enumerate(cfg.biomes):
        mask = (biome_indices == i)
        if b.step_size > 0.001:
            is_stepped[mask] = True
            step_sizes[mask] = b.step_size

    # --- 3. APPLY PLATFORMS (Hybrid Fade) ---
    spawn_origins_list = []
    
    start_x = (width_m - (cfg.num_spawns_per_side-1)*cfg.spacing_m)/2.0
    start_y = (length_m - (cfg.num_spawns_per_side-1)*cfg.spacing_m)/2.0
    sp_x = np.linspace(start_x, start_x + cfg.spacing_m*(cfg.num_spawns_per_side-1), cfg.num_spawns_per_side)
    sp_y = np.linspace(start_y, start_y + cfg.spacing_m*(cfg.num_spawns_per_side-1), cfg.num_spawns_per_side)
    
    max_rad = cfg.platform_width / 2.0
    flat_rad = max_rad * cfg.platform_flat_ratio
    win = int(max_rad / res) + 2

    for sx in sp_x:
        for sy in sp_y:
            ix, iy = int(sx/res), int(sy/res)
            x0, x1 = max(0, ix-win), min(nx+1, ix+win+1)
            y0, y1 = max(0, iy-win), min(ny+1, iy+win+1)
            
            sub_x = xx_noise[x0:x1, y0:y1]
            sub_y = yy_noise[x0:x1, y0:y1]
            sub_z = raw_z_grid[x0:x1, y0:y1]
            
            sx_n, sy_n = sx - x_center_offset, sy - y_center_offset
            dists = np.sqrt((sub_x - sx_n)**2 + (sub_y - sy_n)**2)
            
            mask_circ = dists < max_rad
            if not np.any(mask_circ): continue
            
            center_h = np.mean(sub_z[mask_circ])
            
            mask_flat = dists <= flat_rad
            sub_z[mask_flat] = center_h
            
            mask_fade = (dists > flat_rad) & (dists < max_rad)
            if np.any(mask_fade):
                d = dists[mask_fade]
                z_old = sub_z[mask_fade]
                t = (d - flat_rad)/(max_rad - flat_rad)
                alpha = 0.5 * (1 + np.cos(t * np.pi))
                sub_z[mask_fade] = (center_h * alpha) + (z_old * (1.0 - alpha))
                
                cx0, cx1 = max(0, x0), min(nx, x1)
                cy0, cy1 = max(0, y0), min(ny, y1)
                is_stepped[cx0:cx1, cy0:cy1] = False

            raw_z_grid[x0:x1, y0:y1] = sub_z
            spawn_origins_list.append([sx, sy, center_h])

    # --- 4. CONSTRUCT HYBRID MESH ---
    z_bl = raw_z_grid[:-1, :-1]
    z_br = raw_z_grid[1:, :-1]
    z_tl = raw_z_grid[:-1, 1:]
    z_tr = raw_z_grid[1:, 1:]
    
    z_centers = z_bl.copy() 
    valid_step = step_sizes > 0.001
    z_centers[valid_step] = np.floor(z_centers[valid_step] / step_sizes[valid_step]) * step_sizes[valid_step]
    
    z_bl = np.where(is_stepped, z_centers, z_bl)
    z_br = np.where(is_stepped, z_centers, z_br)
    z_tl = np.where(is_stepped, z_centers, z_tl)
    z_tr = np.where(is_stepped, z_centers, z_tr)

    # FIX FOR MERGE VERTICES: 
    # Use exact grid slices for coordinates instead of adding 'res' manually.
    # This guarantees that the right edge of cell[i] is identical to the left edge of cell[i+1].
    x_bl = xx_v[:-1, :-1]; y_bl = yy_v[:-1, :-1]
    x_br = xx_v[1:, :-1];  y_br = yy_v[1:, :-1]
    x_tr = xx_v[1:, 1:];   y_tr = yy_v[1:, 1:]
    x_tl = xx_v[:-1, 1:];  y_tl = yy_v[:-1, 1:]
    
    cells_v = np.zeros((nx, ny, 4, 3), dtype=np.float32)
    cells_v[:,:,0,0] = x_bl; cells_v[:,:,0,1] = y_bl; cells_v[:,:,0,2] = z_bl
    cells_v[:,:,1,0] = x_br; cells_v[:,:,1,1] = y_br; cells_v[:,:,1,2] = z_br
    cells_v[:,:,2,0] = x_tr; cells_v[:,:,2,1] = y_tr; cells_v[:,:,2,2] = z_tr
    cells_v[:,:,3,0] = x_tl; cells_v[:,:,3,1] = y_tl; cells_v[:,:,3,2] = z_tl

    total_cells = nx * ny
    all_verts = cells_v.reshape(-1, 3) 
    
    ids = np.arange(0, total_cells * 4, 4)
    f1 = np.stack([ids, ids+1, ids+2], axis=1)
    f2 = np.stack([ids, ids+2, ids+3], axis=1)
    
    final_faces = [np.vstack([f1, f2])]
    final_verts = [all_verts]
    v_offset = all_verts.shape[0]

    # --- 5. GENERATE WALLS (GAP FIXER) ---
    # X-Walls
    c_left, c_right = cells_v[:-1, :], cells_v[1:, :]
    z_l_br, z_l_tr = c_left[:, :, 1, 2], c_left[:, :, 2, 2]
    z_r_bl, z_r_tl = c_right[:, :, 0, 2], c_right[:, :, 3, 2]
    
    gap_mask = (np.abs(z_l_br - z_r_bl) > 0.001) | (np.abs(z_l_tr - z_r_tl) > 0.001)
    
    if np.any(gap_mask):
        wx = c_left[gap_mask, 1, 0]
        wy_b, wy_t = c_left[gap_mask, 1, 1], c_left[gap_mask, 2, 1]
        h_l_b, h_l_t = z_l_br[gap_mask], z_l_tr[gap_mask]
        h_r_b, h_r_t = z_r_bl[gap_mask], z_r_tl[gap_mask]
        
        count = len(wx)
        wv = np.zeros((count, 4, 3), dtype=np.float32)
        wv[:, 0, 0] = wx; wv[:, 0, 1] = wy_b; wv[:, 0, 2] = h_l_b
        wv[:, 1, 0] = wx; wv[:, 1, 1] = wy_b; wv[:, 1, 2] = h_r_b
        wv[:, 2, 0] = wx; wv[:, 2, 1] = wy_t; wv[:, 2, 2] = h_r_t
        wv[:, 3, 0] = wx; wv[:, 3, 1] = wy_t; wv[:, 3, 2] = h_l_t
        
        w_ids = np.arange(v_offset, v_offset + count*4, 4)
        wf1 = np.stack([w_ids, w_ids+1, w_ids+2], axis=1)
        wf2 = np.stack([w_ids, w_ids+2, w_ids+3], axis=1)
        wf3 = np.stack([w_ids, w_ids+2, w_ids+1], axis=1)
        wf4 = np.stack([w_ids, w_ids+3, w_ids+2], axis=1)
        
        final_verts.append(wv.reshape(-1, 3))
        final_faces.append(np.vstack([wf1, wf2, wf3, wf4]))
        v_offset += count * 4

    # Y-Walls
    c_bott, c_top = cells_v[:, :-1], cells_v[:, 1:]
    z_b_tl, z_b_tr = c_bott[:, :, 3, 2], c_bott[:, :, 2, 2]
    z_t_bl, z_t_br = c_top[:, :, 0, 2], c_top[:, :, 1, 2]
    
    gap_mask_y = (np.abs(z_b_tl - z_t_bl) > 0.001) | (np.abs(z_b_tr - z_t_br) > 0.001)
    
    if np.any(gap_mask_y):
        wy = c_bott[gap_mask_y, 3, 1]
        wx_l, wx_r = c_bott[gap_mask_y, 3, 0], c_bott[gap_mask_y, 2, 0]
        h_b_l, h_b_r = z_b_tl[gap_mask_y], z_b_tr[gap_mask_y]
        h_t_l, h_t_r = z_t_bl[gap_mask_y], z_t_br[gap_mask_y]
        
        count = len(wy)
        wv = np.zeros((count, 4, 3), dtype=np.float32)
        wv[:, 0, 0] = wx_l; wv[:, 0, 1] = wy; wv[:, 0, 2] = h_b_l
        wv[:, 1, 0] = wx_l; wv[:, 1, 1] = wy; wv[:, 1, 2] = h_t_l
        wv[:, 2, 0] = wx_r; wv[:, 2, 1] = wy; wv[:, 2, 2] = h_t_r
        wv[:, 3, 0] = wx_r; wv[:, 3, 1] = wy; wv[:, 3, 2] = h_b_r
        
        w_ids = np.arange(v_offset, v_offset + count*4, 4)
        wf1 = np.stack([w_ids, w_ids+1, w_ids+2], axis=1)
        wf2 = np.stack([w_ids, w_ids+2, w_ids+3], axis=1)
        wf3 = np.stack([w_ids, w_ids+2, w_ids+1], axis=1)
        wf4 = np.stack([w_ids, w_ids+3, w_ids+2], axis=1)
        
        final_verts.append(wv.reshape(-1, 3))
        final_faces.append(np.vstack([wf1, wf2, wf3, wf4]))

    mesh_v = np.vstack(final_verts)
    mesh_f = np.vstack(final_faces)

    ground_mesh = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, process=False)
    
    meshes_list = [ground_mesh]

    # --- 6. DYNAMIC BLOCKS (UPDATED) ---
    rng = np.random.default_rng(seed=cfg.seed)
    spawn_origins_arr = np.array(spawn_origins_list) if len(spawn_origins_list) > 0 else np.empty((0,3))
    
    # Calculate type weights
    block_weights = np.array([b.weight for b in cfg.block_types])
    block_weights /= block_weights.sum() # Normalize
    
    # Max safe check
    max_block_radius = 0.0
    for b in cfg.block_types:
        w_max = b.width_range[1] * b.scale_range[1]
        max_block_radius = max(max_block_radius, w_max)
    
    safe_r_sq = (max_rad + max_block_radius)**2

    for _ in range(cfg.num_blocks):
        # 6a. Select Block Type
        b_type: BlockCfg = rng.choice(cfg.block_types, p=block_weights)
        
        # 6b. Generate Dimensions (Width, Height, Scale)
        # We generate random width (X) and length (Y) independently for variety, 
        # or share if you want perfect squares. Let's do independent.
        raw_sx = rng.uniform(b_type.width_range[0], b_type.width_range[1])
        raw_sy = rng.uniform(b_type.width_range[0], b_type.width_range[1])
        raw_h  = rng.uniform(b_type.height_range[0], b_type.height_range[1])
        
        global_scale = rng.uniform(b_type.scale_range[0], b_type.scale_range[1])
        
        sx = raw_sx * global_scale
        sy = raw_sy * global_scale
        h_above = raw_h * global_scale
        
        # 6c. Position
        pos_x = rng.uniform(2.0, width_m - 2.0)
        pos_y = rng.uniform(2.0, length_m - 2.0)

        # Check spawn distance
        if len(spawn_origins_arr) > 0:
            d_sq = (pos_x - spawn_origins_arr[:,0])**2 + (pos_y - spawn_origins_arr[:,1])**2
            if np.any(d_sq < safe_r_sq): continue

        # 6d. Height Calculation (Burial)
        ix = int(pos_x / res)
        iy = int(pos_y / res)
        ix = min(max(0, ix), nx-1)
        iy = min(max(0, iy), ny-1)
        
        cell_corners = cells_v[ix, iy, :, 2]
        ground_z = np.mean(cell_corners)

        # Logic: 
        # We want 'h_above' to be the amount sticking OUT.
        # We want 'b_type.burial_depth' to be the amount HIDDEN.
        # Total physical height of box = h_above + burial_depth.
        # Center Z = ground_z - burial_depth + (total_height / 2).
        
        total_h = h_above + b_type.burial_depth
        final_block_z = ground_z - b_type.burial_depth + (total_h / 2.0)
        
        box = trimesh.creation.box(extents=(sx, sy, total_h))
        transform = np.eye(4)
        rot = trimesh.transformations.rotation_matrix(rng.uniform(0, 2*np.pi), [0, 0, 1])
        transform[:3, :3] = rot[:3, :3]
        transform[:3, 3] = [pos_x, pos_y, final_block_z]
        box.apply_transform(transform)
        meshes_list.append(box)

    if len(spawn_origins_list) > 0:
        MultiBiomeTerrainCfg.spawns_positions = torch.tensor(spawn_origins_list, device=device, dtype=torch.float32)
    else:
        MultiBiomeTerrainCfg.spawns_positions = torch.zeros((1,3), device=device)

    return meshes_list, np.zeros(3)

@configclass
class MultiBiomeTerrainCfg(HfTerrainBaseCfg):
    grid_width: float = 0.125         
    terrain_height: float = 5.0 # needs to be high enough for noise range 

    # --- Terrain Shape (The Geometry) ---
    noise_seed: int = 123
    noise_scale: float = 0.03       # Frequency of the Perlin noise (higher = more hills/valleys)
    noise_height_scale: float = 4.0 # Amplitude of the Perlin noise
    noise_octaves: int = 5
    noise_persistence: float = 0.5
    noise_lacunarity: float = 2.0

    # --- Biome Distribution Settings ---
    biome_blend_scale: float = 0.10   # Higher = Choppier transitions. Lower = Larger continents.
    
    # --- THE BIOME list ---
    biomes: list[BiomeCfg] = field(default_factory=lambda: [
        BiomeCfg(weight=1.07, step_size=0.0), # Smooth
        BiomeCfg(weight=1.0, step_size=0.1),
        BiomeCfg(weight=0.9, step_size=0.2),
        #BiomeCfg(weight=0.8, step_size=0.3),
        BiomeCfg(weight=0.7, step_size=0.3), # Giant cliffs
    ])

    # --- Objects ---
    num_blocks: int = 4000
    block_types: list[BlockCfg] = field(default_factory=lambda: [
        # 1. Traversable Debris (Low, walkable)
        BlockCfg(
            weight=8.0,
            width_range=(0.3, 0.6),
            height_range=(0.05, 0.15), # Low height
            scale_range=(1.0, 1.0),
        ),
        # 2. Medium Obstacles (might be climbable)
        BlockCfg(
            weight=1.0,
            width_range=(0.4, 0.75),
            height_range=(0.3, 0.6),
            scale_range=(1.0, 1.5),
        ),
        # 3. Giant Monoliths (Block the path)
        BlockCfg(
            weight=0.5,
            width_range=(1.0, 2.0),
            height_range=(1.5, 3.0),
            scale_range=(1.0, 2.0),
        )
    ])

    num_spawns_per_side = 5
    spacing_m = 20.0  
    spawns_positions: np.ndarray = None
    platform_width: float = 4.0
    platform_flat_ratio: float = 0.5

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