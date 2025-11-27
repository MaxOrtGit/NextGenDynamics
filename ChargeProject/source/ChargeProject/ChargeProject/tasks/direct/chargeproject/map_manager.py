from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from pxr import UsdGeom
import omni.usd
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
import isaaclab.sim as sim_utils

class MapManager:
    def __init__(self, config, num_envs, terrain_dims, device):
        self.device = device
        self.num_envs = num_envs
        self.config = config
        
        # Global World Map (Shared Height)
        self.world_w = terrain_dims[0]
        self.world_h = terrain_dims[1]

        # --- Generate Circular Patrol Mask ---
        # Coordinates -1 to 1
        x = torch.linspace(-1, 1, self.config.staleness_dim, device=device)
        y = torch.linspace(-1, 1, self.config.staleness_dim, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Distance from center (Normalized 0 to 1)
        dist = torch.sqrt(grid_x**2 + grid_y**2)
        
        # Fade width: 15% of radius
        fade_width = 0.15
        
        self.patrol_mask = torch.clamp((1.0 - dist) / fade_width, min=0.0, max=1.0)
        
        # Reshape to (1, 1, H, W) for broadcasting
        self.patrol_mask = self.patrol_mask.view(1, 1, self.config.staleness_dim, self.config.staleness_dim)

        # --- Initialization ---
        ground_prim_path = "/World/ground"
        self.global_height_map = self._scan_entire_world(ground_prim_path)
        
        # Initialize per-env staleness (1.0 = Stale/Dusty)
        self.staleness_maps = torch.zeros(self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim, device=device)
        
        # Pre-calculate 8 cardinal relative offsets (Radius = 12.0m)
        # Angles: 0 (Front), 45, 90 (Left), 135, 180 (Back), etc.
        scan_radius = 12.0 
        angles = torch.arange(0, 2*np.pi, 2*np.pi/8, device=device)
        self.far_sensor_offsets = torch.stack([
            scan_radius * torch.cos(angles),
            scan_radius * torch.sin(angles)
        ], dim=1) # (8, 2)

    
    def _scan_entire_world(self, ground_prim_path):
        rows = int(self.world_h / self.config.height_res) + 1
        cols = int(self.world_w / self.config.height_res) + 1
        
        # Create a config container for the scan
        satellite_cfg = RayCasterCfg(
            prim_path="/World", # Not used, but required by Cfg
            mesh_prim_paths=[ground_prim_path],
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 100.0)),
            ray_alignment="world", # Rays point straight down
            pattern_cfg=patterns.GridPatternCfg(
                resolution=self.config.height_res,
                size=(self.world_w, self.world_h)
            )
        )
        
        # Manually generate the rays
        ray_starts, ray_dirs = satellite_cfg.pattern_cfg.func(satellite_cfg.pattern_cfg, self.device)
        
        # Manually apply the offset
        offset_pos = torch.tensor(list(satellite_cfg.offset.pos), device=self.device)
        ray_starts += offset_pos

        # Manually load the warp mesh
        # This logic is copied from RayCaster._initialize_warp_meshes
        mesh_prim = sim_utils.get_first_matching_child_prim(
            ground_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if mesh_prim is None or not mesh_prim.IsValid():
            # Fallback for PhysX Plane (which is not a "Mesh")
            mesh_prim = sim_utils.get_first_matching_child_prim(
                ground_prim_path, lambda prim: prim.GetTypeName() == "Plane"
            )
            if mesh_prim is not None:
                raise NotImplementedError("Scanning a PhysX Plane is not supported by this script, only 'Mesh' prims.")
            raise RuntimeError(f"Invalid mesh prim path: {ground_prim_path}")
        
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        
        # Apply world transform to vertices
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)

        # Manually perform the raycast
        with torch.no_grad():
            # raycast_mesh returns (hits, hit_dists, hit_normals)
            # We add a batch dim (1) to ray_starts/dirs
            hit_data, _, _, _ = raycast_mesh(
                ray_starts.unsqueeze(0),
                ray_dirs.unsqueeze(0),
                max_dist=satellite_cfg.max_distance,
                mesh=wp_mesh,
            )
        
        # hit_data shape is (1, Num_Rays, 3)
        ground_z = hit_data[0, :, 2]

        # Reshape to (1, 1, H, W)
        height_map = ground_z.view(1, 1, rows, cols)
        
        return height_map

    def update(self, env_origins, robot_pos_w, robot_yaw_w, lidar_hits_w, dt):
        # Update Staleness & Calculate Reward (Amount Cleared)
        cleared_value = self._update_staleness_map(lidar_hits_w, env_origins, dt)

        # Sample Far Sensors (8 Cardinal Directions)
        far_staleness = self._get_far_staleness(robot_pos_w, robot_yaw_w, env_origins)

        # Generate Standard Observations
        nav_map, loco_map = self._sample_egocentric_maps(robot_pos_w, robot_yaw_w, lidar_hits_w, env_origins)

        return nav_map, loco_map, far_staleness, cleared_value

    def _update_staleness_map(self, lidar_hits_w, env_origins, dt):
        # Decay (Everything gets dusty)
        self.staleness_maps += dt * self.config.staleness_decay_rate
        self.staleness_maps = torch.minimum(self.staleness_maps, self.patrol_mask)
        
        # Calculate hits relative to Env Origin
        rel_hits = lidar_hits_w - env_origins.unsqueeze(1)
        
        # Map to Pixel Coordinates
        half_size = self.config.patrol_size / 2
        col = ((rel_hits[..., 0] + half_size) / self.config.patrol_size * self.config.staleness_dim).long()
        row = ((rel_hits[..., 1] + half_size) / self.config.patrol_size * self.config.staleness_dim).long()
        
        # Filter valid hits within patrol zone
        mask = (col >= 0) & (col < self.config.staleness_dim) & (row >= 0) & (row < self.config.staleness_dim)
        
        # Identify indices to clear
        batch_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(col)
        # Linear index for flatten: B * (H*W) + Row * W + Col
        flat_idx = batch_ids * (self.config.staleness_dim**2) + row * self.config.staleness_dim + col
        
        valid_idx = flat_idx[mask]
        
        total_cleared_value = torch.zeros(self.num_envs, device=self.device)
        
        if valid_idx.numel() > 0:
            flat_map = self.staleness_maps.view(-1)
            
            # --- Exploration Reward Calculation ---
            # Get values before we clear them
            # Note: Multiple rays might hit the same cell. We should only count unique cells to prevent reward hacking
            # by staring at the same spot. 
            unique_idx, _ = torch.unique(valid_idx, return_inverse=True)
            
            # Map unique indices back to environment IDs
            # We know index = env_id * dim^2 + ...
            env_ids_for_unique = torch.div(unique_idx, (self.config.staleness_dim**2), rounding_mode='floor')
            
            values_to_clear = flat_map[unique_idx]
            
            # Sum values per environment
            total_cleared_value.index_add_(0, env_ids_for_unique, values_to_clear)
            
            # Clear map
            flat_map[valid_idx] = 0.0 
            self.staleness_maps = flat_map.view(self.num_envs, 1, self.config.staleness_dim, self.config.staleness_dim)
            
        return total_cleared_value

    def _get_far_staleness(self, robot_pos_w, robot_yaw_w, env_origins):
        """Samples staleness at 8 cardinal directions rotated by robot yaw."""
        cos = torch.cos(robot_yaw_w).squeeze(-1)
        sin = torch.sin(robot_yaw_w).squeeze(-1)
        
        # Rotate offsets by Robot Yaw
        x_off = self.far_sensor_offsets[:, 0]
        y_off = self.far_sensor_offsets[:, 1] 
        
        # Rotated offsets (N, 8)
        rx = x_off.unsqueeze(0) * cos.unsqueeze(1) - y_off.unsqueeze(0) * sin.unsqueeze(1)
        ry = x_off.unsqueeze(0) * sin.unsqueeze(1) + y_off.unsqueeze(0) * cos.unsqueeze(1)
        
        # Add to Robot Position to get World Position
        # (N, 1) + (N, 8)
        px = robot_pos_w[:, 0].unsqueeze(1) + rx
        py = robot_pos_w[:, 1].unsqueeze(1) + ry
        
        # Normalize to [-1, 1] grid coordinates relative to Patrol Box
        # grid = (pos - env_origin) / (patrol_size/2)
        rel_x = px - env_origins[:, 0].unsqueeze(1)
        rel_y = py - env_origins[:, 1].unsqueeze(1)
        
        norm_x = rel_x / (self.config.patrol_size / 2.0)
        norm_y = rel_y / (self.config.patrol_size / 2.0)
        
        # Stack for grid_sample: (N, 1, 8, 2) -> Treated as a "Line" image of width 8
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)
        
        # Sample
        # Output: (N, 1, 1, 8)
        samples = F.grid_sample(self.staleness_maps, grid, align_corners=False, padding_mode='border')
        
        return samples.view(self.num_envs, 8)

    def _sample_egocentric_maps(self, robot_pos_w, robot_yaw_w, lidar_hits_w, env_origins):
        cos = torch.cos(robot_yaw_w).squeeze(-1)
        sin = torch.sin(robot_yaw_w).squeeze(-1)
        
        # --- Prepare Affine Data for Global Height Map ---
        # World Normals [-1, 1]
        tx = (robot_pos_w[:, 0]) / (self.world_w / 2)
        ty = (robot_pos_w[:, 1]) / (self.world_h / 2)
        
        # Nav Height (33x33, 16m wide)
        zoom_nav = self.config.nav_size / self.world_w
        theta_nav = self._get_affine_matrix(zoom_nav, cos, sin, tx, ty)
        grid_nav = F.affine_grid(theta_nav, torch.Size((self.num_envs, 1, self.config.nav_dim, self.config.nav_dim)), align_corners=False)
        nav_height = F.grid_sample(self.global_height_map.expand(self.num_envs, -1, -1, -1), grid_nav, align_corners=False, padding_mode='border')
        
        # Loco Height (25x25, 2.4m wide)
        zoom_loco = self.config.loco_size / self.world_w
        theta_loco = self._get_affine_matrix(zoom_loco, cos, sin, tx, ty)
        grid_loco = F.affine_grid(theta_loco, torch.Size((self.num_envs, 1, self.config.loco_dim, self.config.loco_dim)), align_corners=False)
        loco_height = F.grid_sample(self.global_height_map.expand(self.num_envs, -1, -1, -1), grid_loco, align_corners=False, padding_mode='border')
        
        # Normalize Heights: Subtract Robot Z so feet are at ~0
        robot_z = robot_pos_w[:, 2].view(self.num_envs, 1, 1, 1)
        nav_height -= robot_z
        loco_height -= robot_z

        # --- Prepare Affine Data for Local Staleness ---
        # Normals relative to Patrol Zone [-1, 1]
        rel_pos = robot_pos_w - env_origins
        tx_s = rel_pos[:, 0] / (self.config.patrol_size / 2)
        ty_s = rel_pos[:, 1] / (self.config.patrol_size / 2)
        
        # Zoom: 16m Nav View inside 24m Patrol Map
        zoom_s = self.config.nav_size / self.config.patrol_size
        theta_s = self._get_affine_matrix(zoom_s, cos, sin, tx_s, ty_s)
        grid_s = F.affine_grid(theta_s, torch.Size((self.num_envs, 1, self.config.nav_dim, self.config.nav_dim)), align_corners=False)
        nav_staleness = F.grid_sample(self.staleness_maps, grid_s, align_corners=False, padding_mode='border')
        
        # --- Generate Lidar Density (33x33) ---
        # This needs to be in the Robot Frame
        # Transform hits to robot frame (Rotate by -Yaw)
        diff = lidar_hits_w - robot_pos_w.unsqueeze(1)
        x_local = diff[..., 0] * cos.unsqueeze(1) + diff[..., 1] * sin.unsqueeze(1)
        y_local = -diff[..., 0] * sin.unsqueeze(1) + diff[..., 1] * cos.unsqueeze(1)
        
        # Bin into 33x33
        half_nav = self.config.nav_size / 2
        x_idx = ((x_local + half_nav) / self.config.nav_size * self.config.nav_dim).long()
        y_idx = ((y_local + half_nav) / self.config.nav_size * self.config.nav_dim).long()
        
        mask_l = (x_idx >= 0) & (x_idx < self.config.nav_dim) & (y_idx >= 0) & (y_idx < self.config.nav_dim)
        
        nav_density = torch.zeros(self.num_envs * self.config.nav_dim**2, device=self.device)
        b_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(x_idx)
        flat_idx = b_ids * (self.config.nav_dim**2) + y_idx * self.config.nav_dim + x_idx
        
        nav_density.scatter_add_(0, flat_idx[mask_l], torch.ones_like(flat_idx[mask_l], dtype=torch.float))
        nav_density = nav_density.view(self.num_envs, 1, self.config.nav_dim, self.config.nav_dim)
        nav_density = torch.log1p(nav_density) # Log compress
        
        # Combine Nav Inputs: (N, 3, 33, 33)
        nav_combined = torch.cat([nav_staleness, nav_density, nav_height], dim=1)
        
        return nav_combined, loco_height

    def _get_affine_matrix(self, scale, cos, sin, tx, ty):
        theta = torch.zeros(self.num_envs, 2, 3, device=self.device)
        theta[:, 0, 0] = scale * cos
        theta[:, 0, 1] = -scale * sin
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = scale * sin
        theta[:, 1, 1] = scale * cos
        theta[:, 1, 2] = ty
        return theta
    
    def reset(self, env_ids):
        # Reset specific environments
        self.staleness_maps[env_ids] = 1.0