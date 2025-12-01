"""
Dynamic Octant Staleness Test
- Animates a robot moving & rotating
- Computes mean staleness per octant (yaw-rotated)
- Visualizes staleness map, octant boundaries, octant means
- Robust if robot leaves the patrol zone (no crash)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------
# Config
# -------------------------
class C:
    staleness_dim = 128
    patrol_size = 24.0    # meters (full width)
    fps = 20
    run_seconds = 40

config = C()
device = "cpu"

# -------------------------
# Synthetic staleness map factory
# -------------------------
def make_fake_staleness_map(dim):
    xs = torch.linspace(-1, 1, dim)
    ys = torch.linspace(-1, 1, dim)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(gx**2 + gy**2)

    blob1 = torch.exp(-((gx - 0.4) ** 2 + (gy + 0.2) ** 2) * 10)
    blob2 = torch.exp(-((gx + 0.25) ** 2 + (gy - 0.35) ** 2) * 12)
    blob3 = torch.exp(-(r ** 2) * 4)
    noise = 0.03 * torch.randn_like(gx)

    m = (blob1 + blob2 + 0.6 * blob3 + noise).clamp(min=0.0)
    # apply a circular patrol mask (soft fade)
    fade_width = 0.15
    mask = torch.clamp((1.0 - r) / fade_width, min=0.0, max=1.0)
    m = m * mask
    return m.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

# -------------------------
# Octant computation (vectorized, robust)
# -------------------------
def compute_octant_staleness(staleness_map, robot_pos, robot_yaw, env_origin, config):
    """
    staleness_map: (1,1,H,W) torch tensor
    robot_pos: (2,) tensor or array [x,y] world coords
    robot_yaw: scalar (radians) tensor or float
    env_origin: (2,) tensor or array world coords (center of patrol box)
    returns:
      (8,) tensor of mean staleness per octant,
      oct_idx (H,W) long tensor of octant assignment for visualization
    """
    H = W = config.staleness_dim
    half = config.patrol_size / 2.0

    # pixel-centered normalized coords in [-1,1]
    xs = torch.linspace(-1, 1, W, device=staleness_map.device)
    ys = torch.linspace(-1, 1, H, device=staleness_map.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)

    # map to world coords inside patrol box
    pixel_x = gx * half + env_origin[0]
    pixel_y = gy * half + env_origin[1]

    # vector from robot to pixel
    dx = pixel_x - robot_pos[0]
    dy = pixel_y - robot_pos[1]

    # rotate world vectors into robot frame (so octants align with robot yaw)
    cos = float(torch.cos(torch.tensor(robot_yaw)))
    sin = float(torch.sin(torch.tensor(robot_yaw)))
    rx = dx * cos + dy * sin
    ry = -dx * sin + dy * cos

    angles = torch.atan2(ry, rx)  # [-pi, pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # [0,2pi)

    # octant index 0..7
    oct_idx = (angles / (np.pi / 4.0)).long() % 8  # (H,W) long

    flat_map = staleness_map.view(-1)  # (H*W,)
    flat_idx = oct_idx.view(-1)        # (H*W,)

    out = torch.zeros(8, device=staleness_map.device)
    counts = torch.zeros(8, device=staleness_map.device)

    out.scatter_add_(0, flat_idx, flat_map)
    counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))

    mean = out / (counts + 1e-9)
    return mean, oct_idx

# -------------------------
# Helpers: world<->image coordinates for plotting
# -------------------------
def world_to_image(robot_pos, env_origin, config):
    """Return pixel coordinates (x_img, y_img) for plotting on imshow grid."""
    H = W = config.staleness_dim
    half = config.patrol_size / 2.0
    # normalized in [-1,1]
    nx = (robot_pos[0] - env_origin[0]) / half
    ny = (robot_pos[1] - env_origin[1]) / half
    # convert to image indices [0 .. W-1], origin='lower' is used
    ix = ((nx + 1.0) / 2.0) * (W - 1)
    iy = ((ny + 1.0) / 2.0) * (H - 1)
    return ix, iy

def clamp_to_image(ix, iy, W, H):
    return np.clip(ix, 0, W - 1), np.clip(iy, 0, H - 1)

# -------------------------
# Build test map + initial robot state
# -------------------------
staleness_map = make_fake_staleness_map(config.staleness_dim)  # (1,1,H,W)
map_display = staleness_map[0, 0].numpy()

env_origin = np.array([0.0, 0.0])      # patrol map centered at world origin
robot_pos = np.array([0.0, 0.0])       # start at center
robot_yaw = 0.0

# robot motion parameters (random-walk + slow rotation)
np.random.seed(1)
vel = np.array([0.06, 0.03])    # meters per frame baseline
rot_vel = 0.06                  # rad/frame baseline

# occasional random perturbations
def step_robot(pos, yaw, step):
    # base circular-ish motion plus small noise -> ensures leaving sometimes possible
    t = step / (config.fps)
    # move in a gentle figure-8 + drifting offset
    x = 6.0 * np.cos(0.5 * t) + 0.8 * np.sin(0.25 * t)
    y = 4.0 * np.sin(0.35 * t)
    # add a slow drift so sometimes the robot leaves the patrol zone
    drift = 8.0 * np.sin(0.12 * t)
    pos_next = np.array([x + drift * 0.05, y + drift * 0.02])
    yaw_next = yaw + rot_vel * 0.8 * np.cos(0.15 * t) + 0.02 * np.sin(0.3 * t)
    return pos_next, yaw_next

# -------------------------
# Matplotlib figure + animation setup
# -------------------------
H = W = config.staleness_dim
fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(map_display, origin="lower", cmap="magma", vmin=0.0, vmax=1.2)
ax.set_title("Dynamic Octant Staleness (Robot = dot)")

# robot marker and out-of-bounds text
robot_marker, = ax.plot([], [], marker="o", markersize=10, markeredgecolor="k", markerfacecolor="white")
oob_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="red", fontsize=12, va="top")

# octant boundary lines and labels
lines = []
labels = []
center = (W / 2.0, H / 2.0)
radius = max(W, H) * 0.7
for k in range(8):
    ln, = ax.plot([], [], 'w--', linewidth=1)
    lines.append(ln)
    lbl = ax.text(0, 0, "", color="white", fontsize=10, ha="center", va="center")
    labels.append(lbl)

# colorbar
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xlim(-0.5, W - 0.5)
ax.set_ylim(-0.5, H - 0.5)

# show octant mean text in top-right
oct_text = ax.text(0.99, 0.99, "", transform=ax.transAxes, color="white", fontsize=10, ha="right", va="top",
                   bbox=dict(facecolor='black', alpha=0.4))

# -------------------------
# Animation update function
# -------------------------
total_steps = int(config.run_seconds * config.fps)
step = 0

def update(frame):
    global robot_pos, robot_yaw, step
    step += 1

    # update robot state
    robot_pos, robot_yaw = step_robot(robot_pos, robot_yaw, step)

    # compute octant means
    robot_pos_t = torch.tensor(robot_pos, dtype=torch.float32)
    mean_oct, oct_idx = compute_octant_staleness(staleness_map, robot_pos_t, robot_yaw, torch.tensor(env_origin, dtype=torch.float32), config)
    mean_np = mean_oct.cpu().numpy()

    # draw octant lines (rotated by robot_yaw)
    for k in range(8):
        angle = robot_yaw + k * (np.pi / 4.0)
        x2 = center[0] + np.cos(angle) * radius
        y2 = center[1] + np.sin(angle) * radius
        lines[k].set_data([center[0], x2], [center[1], y2])

        # label positions halfway along each octant
        angle_mid = robot_yaw + (k + 0.5) * (np.pi / 4.0)
        lx = center[0] + np.cos(angle_mid) * (radius * 0.45)
        ly = center[1] + np.sin(angle_mid) * (radius * 0.45)
        labels[k].set_position((lx, ly))
        labels[k].set_text(f"{mean_np[k]:.2f}")

    # robot plotting (map image coordinates)
    ix, iy = world_to_image(robot_pos, env_origin, config)  # robot pixel coords

    for k in range(8):
        angle = robot_yaw + k * (np.pi / 4.0)

        # end of line in image coords
        length = max(W, H) * 0.6
        x2 = ix + np.cos(angle) * length
        y2 = iy + np.sin(angle) * length

        # update boundary line
        lines[k].set_data([ix, x2], [iy, y2])

        # label halfway along the line
        angle_mid = robot_yaw + (k + 0.5) * (np.pi / 4.0)
        lx = ix + np.cos(angle_mid) * (length * 0.5)
        ly = iy + np.sin(angle_mid) * (length * 0.5)

        labels[k].set_position((lx, ly))
        labels[k].set_text(f"{mean_np[k]:.2f}")

    # determine if inside patrol box
    inside = (abs(robot_pos[0] - env_origin[0]) <= config.patrol_size/2.0) and (abs(robot_pos[1] - env_origin[1]) <= config.patrol_size/2.0)

    if inside:
        robot_marker.set_markerfacecolor("white")
        robot_marker.set_markersize(10)
        robot_marker.set_data([ix], [iy])
        oob_text.set_text("")
    else:
        # clamp to image edge and show red marker & OUTSIDE text
        ix_c, iy_c = clamp_to_image(ix, iy, W, H)
        robot_marker.set_markerfacecolor("red")
        robot_marker.set_markersize(8)
        robot_marker.set_data([ix_c], [iy_c])
        oob_text.set_text("OUTSIDE PATROL ZONE")

    # update octant summary box
    lines_str = "\n".join([f"{i}: {mean_np[i]:.3f}" for i in range(8)])
    oct_text.set_text("Octant means:\n" + lines_str)

    return lines + labels + [robot_marker, oob_text, oct_text]

# -------------------------
# Run animation
# -------------------------
ani = animation.FuncAnimation(fig, update, frames=total_steps, interval=1000/config.fps, blit=False, repeat=False)
plt.show()
