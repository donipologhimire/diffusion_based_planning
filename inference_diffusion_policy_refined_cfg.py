import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
import math
import os
import pickle 

# --- Constants and Configuration ---
# !!! MUST MATCH CFG TRAINING SCRIPT !!!
# !!! Update path to the CFG-trained model !!!
MODEL_CHECKPOINT_PATH = 'diffusion_policy_unet_refined_cfg.pth'
# MODEL_CHECKPOINT_PATH = 'model_epoch_120.pth'

GRID_SIZE = 50
STATE_DIM = 2
MAX_TRAJ_LEN = int(GRID_SIZE * 1.5)

# Diffusion Hyperparameters (MUST MATCH TRAINING)
N_DIFFUSION_STEPS = 100
BETA_START = 1e-4
BETA_END = 0.02

# Model Hyperparameters (MUST MATCH TRAINING)
MODEL_BASE_DIM = 64
MODEL_DIM_MULTS = (1, 2, 4, 8)
MODEL_TIME_EMB_DIM = 128
MODEL_START_GOAL_DIM = 4
MODEL_OBSTACLE_EMB_DIM = 128
MODEL_COND_EMB_EXTRA_DIM = 128 # CHECK YOUR TRAINING SCRIPT VALUE
MODEL_ATTN_LEVELS = (2, 3)     # CHECK YOUR TRAINING SCRIPT VALUE

# !!! CFG Inference Hyperparameter !!!
GUIDANCE_SCALE = 3.0 # Tune this value (e.g., 1.0 = no guidance, higher values = stronger conditioning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Obstacle Map Definition (Copied - ensure identical) ---
def create_obstacle_map(grid_size=GRID_SIZE):
    obstacle_map = np.zeros((grid_size, grid_size), dtype=int)
    wall_width = 1; h1_start=max(0, grid_size // 3 - wall_width); h1_end=min(grid_size, grid_size // 3 + wall_width); h1_len=min(grid_size, grid_size // 2 - 8)
    if h1_len > 0: obstacle_map[h1_start:h1_end, :h1_len] = 1
    h2_start=max(0, 2 * grid_size // 3 - wall_width); h2_end=min(grid_size, 2 * grid_size // 3 + wall_width); h2_col_start=max(0, grid_size // 2 + 8); h2_col_end=min(grid_size, 43)
    if h2_col_start < h2_col_end: obstacle_map[h2_start:h2_end, h2_col_start:h2_col_end] = 1
    v1_row_start=max(0, 7); v1_row_end=min(grid_size, grid_size - 5); v1_start=max(0, grid_size // 3 - wall_width); v1_end=min(grid_size, grid_size // 3 + wall_width)
    if v1_row_start < v1_row_end: obstacle_map[v1_row_start:v1_row_end, v1_start:v1_end] = 1
    v2_row_start=max(0, grid_size // 2 + 7); v2_start=max(0, (2 * grid_size) // 3 - wall_width); v2_end=min(grid_size, (2 * grid_size) // 3 + wall_width)
    if v2_row_start < grid_size: obstacle_map[v2_row_start:, v2_start:v2_end] = 1
    obstacle_map[0, :]=1; obstacle_map[-1, :]=1; obstacle_map[:, 0]=1; obstacle_map[:, -1]=1
    return obstacle_map

# --- Coordinate Normalization/Denormalization (Copied) ---
def normalize_coordinates(coord, grid_size):
    if np.any(np.abs(coord) <= 1.0):
        if np.max(np.abs(coord)) <= 1.0 and np.min(coord)>=-1.0: return tuple(coord)
    row_norm = ((coord[0] / (grid_size - 1)) * 2 - 1); col_norm = ((coord[1] / (grid_size - 1)) * 2 - 1)
    return (row_norm, col_norm)

def denormalize_coordinates(norm_coord, grid_size):
     row = int(round(((norm_coord[0] + 1) / 2) * (grid_size - 1))); col = int(round(((norm_coord[1] + 1) / 2) * (grid_size - 1)))
     row = max(0, min(grid_size - 1, row)); col = max(0, min(grid_size - 1, col))
     return (row, col)

# --- Diffusion Utilities (Copied - with additions for sampling) ---
betas = torch.linspace(BETA_START, BETA_END, N_DIFFUSION_STEPS, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_log_variance_clipped = torch.log(torch.maximum(posterior_variance, betas))

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.long())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- Model Definition (MUST BE IDENTICAL TO CFG TRAINING SCRIPT) ---
# --- Paste the IDENTICAL model definition (ConditionalUNet1D and its sub-modules) here ---
# --- from your `train_diffusion_policy_refined_cfg.py`          ---
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# -- Building Blocks --
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ObstacleEncoderCNN(nn.Module):
    def __init__(self, grid_size, output_dim=128, base_channels=16):
        super().__init__()
        self.output_dim = output_dim
        layers = []
        in_channels = 1
        curr_size = grid_size
        curr_channels = base_channels
        while curr_size > 4:
            layers.extend([
                nn.Conv2d(in_channels, curr_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(8, curr_channels // 2 if curr_channels > 1 else 1), curr_channels),
                nn.SiLU()
            ])
            in_channels = curr_channels
            curr_channels *= 2
            curr_size //= 2
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        final_conv_channels = in_channels
        layers.append(nn.Linear(final_conv_channels, output_dim))
        layers.append(nn.SiLU())
        self.encoder = nn.Sequential(*layers)
    def forward(self, obstacle_map):
        if obstacle_map.dim() == 3:
            obstacle_map = obstacle_map.unsqueeze(1)
        return self.encoder(obstacle_map)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim, groups=8, dropout=0.1, activation=nn.SiLU):
        super().__init__()
        groups_in = min(groups, in_channels) if in_channels > 0 else 1
        while groups_in > 1 and in_channels % groups_in != 0: groups_in -=1
        groups_out = min(groups, out_channels) if out_channels > 0 else 1
        while groups_out > 1 and out_channels % groups_out != 0: groups_out -=1
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = activation()
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = activation()
        self.dropout = nn.Dropout(dropout)
        self.time_mlp = nn.Sequential(activation(), nn.Linear(time_emb_dim, out_channels))
        self.cond_mlp = nn.Sequential(activation(), nn.Linear(cond_emb_dim, out_channels))
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, time_emb, cond_emb):
        residual = self.skip_connection(x)
        h = self.norm1(x); h = self.act1(h); h = self.conv1(h)
        time_bias = self.time_mlp(time_emb).unsqueeze(-1)
        cond_bias = self.cond_mlp(cond_emb).unsqueeze(-1)
        h = h + time_bias + cond_bias
        h = self.norm2(h); h = self.act2(h); h = self.dropout(h); h = self.conv2(h)
        return h + residual

class AttentionBlock1D(nn.Module):
    def __init__(self, channels, num_heads=4, groups=8):
        super().__init__()
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads; self.head_dim = channels // num_heads; self.scale = self.head_dim ** -0.5
        groups = min(groups, channels)
        while groups > 1 and channels % groups != 0: groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.to_qkv = nn.Linear(channels, channels * 3); self.to_out = nn.Linear(channels, channels)
    def forward(self, x):
        B, C, T = x.shape; h = self.norm(x); h = h.transpose(-1, -2)
        qkv = self.to_qkv(h).chunk(3, dim=-1); q, k, v = map(lambda t: t.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3), qkv)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale; attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v); out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.to_out(out); out = out.transpose(-1, -2)
        return x + out

class Upsample1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(); self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False); self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    def forward(self, x): return self.conv(self.upsample(x))

class Downsample1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
       super().__init__(); self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
    def forward(self, x): return self.conv(x)

class ConditionalUNet1D(nn.Module):
    def __init__(self, in_channels, max_traj_len, grid_size, base_dim=64, dim_mults=(1, 2, 4, 8), time_emb_dim=128, start_goal_dim=4, obstacle_emb_dim=128, cond_emb_extra_dim=64, dropout=0.1, activation=nn.SiLU, attn_levels=(2,)):
        super().__init__(); self.max_traj_len = max_traj_len
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 4), activation(), nn.Linear(time_emb_dim * 4, time_emb_dim))
        self.obstacle_encoder = ObstacleEncoderCNN(grid_size=grid_size, output_dim=obstacle_emb_dim)
        self.start_goal_mlp = nn.Sequential(nn.Linear(start_goal_dim, time_emb_dim // 2), activation(), nn.Linear(time_emb_dim // 2, time_emb_dim // 2))
        combined_cond_input_dim = time_emb_dim + obstacle_emb_dim + (time_emb_dim // 2)
        self.combined_cond_mlp = nn.Sequential(nn.Linear(combined_cond_input_dim, cond_emb_extra_dim), activation(), nn.Linear(cond_emb_extra_dim, cond_emb_extra_dim)); final_cond_emb_dim = cond_emb_extra_dim
        dims = [base_dim] + [base_dim * m for m in dim_mults]; in_out_dims = list(zip(dims[:-1], dims[1:])); num_resolutions = len(in_out_dims)
        self.init_conv = nn.Conv1d(in_channels, base_dim, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([]); current_dim = base_dim
        for i, (dim_in, dim_out) in enumerate(in_out_dims):
            is_last = (i == num_resolutions - 1); use_attn = (i in attn_levels)
            self.downs.append(nn.ModuleList([
                ResidualBlock1D(dim_in, dim_out, time_emb_dim, final_cond_emb_dim, activation=activation, dropout=dropout),
                ResidualBlock1D(dim_out, dim_out, time_emb_dim, final_cond_emb_dim, activation=activation, dropout=dropout),
                AttentionBlock1D(dim_out) if use_attn else nn.Identity(),
                Downsample1DLayer(dim_out, dim_out) if not is_last else nn.Identity()
            ])); current_dim = dim_out
        mid_dim = current_dim; self.mid_block1 = ResidualBlock1D(mid_dim, mid_dim, time_emb_dim, final_cond_emb_dim, activation=activation, dropout=dropout); self.mid_attn = AttentionBlock1D(mid_dim); self.mid_block2 = ResidualBlock1D(mid_dim, mid_dim, time_emb_dim, final_cond_emb_dim, activation=activation, dropout=dropout)
        self.ups = nn.ModuleList([]); current_dim = mid_dim
        for i, (target_dim_out, _) in enumerate(reversed(in_out_dims)):
            down_level_index = num_resolutions - 1 - i; actual_skip_dim = dims[down_level_index + 1]; res1_in_dim = current_dim + actual_skip_dim; use_attn = (down_level_index in attn_levels)
            self.ups.append(nn.ModuleList([
                Upsample1DLayer(current_dim, current_dim) if i > 0 else nn.Identity(),
                ResidualBlock1D(res1_in_dim, target_dim_out, time_emb_dim, final_cond_emb_dim, activation=activation, dropout=dropout),
                ResidualBlock1D(target_dim_out, target_dim_out, time_emb_dim, final_cond_emb_dim, activation=activation, dropout=dropout),
                AttentionBlock1D(target_dim_out) if use_attn else nn.Identity(),
            ])); current_dim = target_dim_out
        final_groups = min(8, base_dim)
        while final_groups > 1 and base_dim % final_groups != 0: final_groups -=1
        self.final_norm = nn.GroupNorm(final_groups, base_dim); self.final_act = activation(); self.final_conv = nn.Conv1d(base_dim, in_channels, kernel_size=3, padding=1)
    def forward(self, x, timesteps, start_goal_coords, obstacle_maps):
        x = x.permute(0, 2, 1); t_emb = self.time_mlp(timesteps); o_emb = self.obstacle_encoder(obstacle_maps); sg_emb = self.start_goal_mlp(start_goal_coords)
        combined_cond = torch.cat([t_emb, o_emb, sg_emb], dim=-1); cond_emb = self.combined_cond_mlp(combined_cond)
        h = self.init_conv(x); skips = [h]
        for i, (res1, res2, attn, downsample) in enumerate(self.downs):
            h = res1(h, t_emb, cond_emb); h = res2(h, t_emb, cond_emb); h = attn(h); skips.append(h); h = downsample(h)
        h = self.mid_block1(h, t_emb, cond_emb); h = self.mid_attn(h); h = self.mid_block2(h, t_emb, cond_emb)
        skips = list(reversed(skips))
        for i, (upsample_layer, res1, res2, attn) in enumerate(self.ups):
             h = upsample_layer(h); skip = skips[i]
             if h.shape[-1] != skip.shape[-1]: h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
             combined = torch.cat((h, skip), dim=1); h = res1(combined, t_emb, cond_emb); h = res2(h, t_emb, cond_emb); h = attn(h)
        h = self.final_norm(h); h = self.final_act(h); out = self.final_conv(h)
        return out.permute(0, 2, 1)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- Sampling Function (Reverse Process with CFG) ---
@torch.no_grad()
def p_sample(model, x, t, t_index, condition_sg, condition_obs, null_sg, null_obs, guidance_scale):
    """ Samples x_{t-1} from x_t using the model prediction with CFG """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # --- CFG Prediction ---
    # Predict noise conditioned on provided start/goal and obstacles
    noise_pred_cond = model(x, t, condition_sg, condition_obs)

    # Predict noise unconditionally (using null conditions)
    noise_pred_uncond = model(x, t, null_sg, null_obs)

    # Combine predictions using the guidance scale
    # noise = unconditional_prediction + scale * (conditional_prediction - unconditional_prediction)
    predicted_noise = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    # ---------------------

    # Calculate x_{t-1} mean component using the CFG noise prediction
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean # No noise added at the last step
    else:
        # Add noise
        posterior_log_variance_t = extract(posterior_log_variance_clipped, t, x.shape)
        noise = torch.randn_like(x)
        model_std_dev = torch.exp(0.5 * posterior_log_variance_t)
        return model_mean + model_std_dev * noise

@torch.no_grad()
def p_sample_loop(model, shape, condition_sg, condition_obs, start_coord_norm, goal_coord_norm, guidance_scale):
    """ Performs the full reverse diffusion sampling loop with CFG """
    batch_size = shape[0]
    x_t = torch.randn(shape, device=DEVICE)

    # Apply initial start/goal constraints to the noise
    x_t[:, 0, :] = torch.tensor(start_coord_norm, dtype=torch.float32, device=DEVICE)
    x_t[:, -1, :] = torch.tensor(goal_coord_norm, dtype=torch.float32, device=DEVICE)

    # --- Create null conditions for CFG ---
    # Assuming batch_size = 1 for typical inference, adjust if needed
    null_sg = torch.zeros_like(condition_sg)
    null_obs = torch.zeros_like(condition_obs)
    # --------------------------------------

    print(f"Starting sampling loop with Guidance Scale: {guidance_scale}...")
    for i in tqdm(reversed(range(0, N_DIFFUSION_STEPS)), desc="Sampling", total=N_DIFFUSION_STEPS):
        t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
        x_t = p_sample(model, x_t, t, i, condition_sg, condition_obs, null_sg, null_obs, guidance_scale) # Pass null conds and scale

        # Apply Start/Goal Constraints HARD at each step
        x_t[:, 0, :] = torch.tensor(start_coord_norm, dtype=torch.float32, device=DEVICE)
        x_t[:, -1, :] = torch.tensor(goal_coord_norm, dtype=torch.float32, device=DEVICE)

    print("Sampling finished.")
    return x_t.cpu().numpy()


# --- Main Inference Script ---
if __name__ == "__main__":
    # --- 1. Define Start/Goal and Obstacles ---
    start_point_grid = (3, 5)
    goal_point_grid = (45, 45)
    # start_point_grid = (43, 5)
    # goal_point_grid = (45, 45)
    obstacle_map = create_obstacle_map(GRID_SIZE)

    # --- 2. Prepare Model Inputs ---
    start_norm = normalize_coordinates(start_point_grid, GRID_SIZE)
    goal_norm = normalize_coordinates(goal_point_grid, GRID_SIZE)
    start_goal_cond = torch.tensor(np.concatenate([start_norm, goal_norm]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    obstacle_cond = torch.tensor(obstacle_map, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    print(f"Start (Grid): {start_point_grid}, Start (Norm): {start_norm}")
    print(f"Goal (Grid): {goal_point_grid}, Goal (Norm): {goal_norm}")
    print(f"Start/Goal Condition Tensor Shape: {start_goal_cond.shape}")
    print(f"Obstacle Condition Tensor Shape: {obstacle_cond.shape}")

    # --- 3. Load CFG-Trained Model ---
    model = ConditionalUNet1D(
        in_channels=STATE_DIM,
        max_traj_len=MAX_TRAJ_LEN,
        grid_size=GRID_SIZE,
        base_dim=MODEL_BASE_DIM,
        dim_mults=MODEL_DIM_MULTS,
        time_emb_dim=MODEL_TIME_EMB_DIM,
        start_goal_dim=MODEL_START_GOAL_DIM,
        obstacle_emb_dim=MODEL_OBSTACLE_EMB_DIM,
        cond_emb_extra_dim=MODEL_COND_EMB_EXTRA_DIM, # Check value
        attn_levels=MODEL_ATTN_LEVELS            # Check value
    ).to(DEVICE)

    print(f"Loading CFG model checkpoint from: {MODEL_CHECKPOINT_PATH}")
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
        exit()

    try:
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state_dict from checkpoint.")
            # Optionally load other info if needed (epoch, loss, cfg_prob)
            if 'epoch' in checkpoint: print(f"Model trained for {checkpoint['epoch']+1} epochs.")
            if 'cfg_prob_uncond' in checkpoint: print(f"Trained with CFG prob: {checkpoint['cfg_prob_uncond']:.2f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict directly.")

        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        exit()

    # --- 4. Perform Sampling with CFG ---
    sampling_shape = (1, MAX_TRAJ_LEN, STATE_DIM) # Batch size = 1

    generated_traj_norm = p_sample_loop(
        model,
        shape=sampling_shape,
        condition_sg=start_goal_cond,
        condition_obs=obstacle_cond,
        start_coord_norm=start_norm,
        goal_coord_norm=goal_norm,
        guidance_scale=GUIDANCE_SCALE # Pass the guidance scale
    )

    final_trajectory_norm = generated_traj_norm[0]

    # --- 5. Visualize Result ---
    print("Denormalizing trajectory for plotting...")
    final_trajectory_denorm = np.array([denormalize_coordinates(p, GRID_SIZE) for p in final_trajectory_norm])

    plt.figure(figsize=(10, 10))
    plt.imshow(obstacle_map, cmap='gray_r', origin='lower', extent=(0, GRID_SIZE, 0, GRID_SIZE), alpha=0.8)
    path_rows, path_cols = final_trajectory_denorm[:, 0], final_trajectory_denorm[:, 1]
    plt.plot(path_cols + 0.5, path_rows + 0.5, marker='.', color='cyan', linestyle='-', linewidth=2, markersize=5, label='Generated Trajectory')
    plt.plot(start_point_grid[1] + 0.5, start_point_grid[0] + 0.5, 'go', markersize=12, label='Start', markeredgecolor='black')
    plt.plot(goal_point_grid[1] + 0.5, goal_point_grid[0] + 0.5, 'ro', markersize=12, label='Goal', markeredgecolor='black')

    # plt.title(f"Generated Trajectory (CFG Sampling, Scale={GUIDANCE_SCALE}, {N_DIFFUSION_STEPS} steps)")
    plt.title(f"Generated Trajectory ( {N_DIFFUSION_STEPS} steps)")

    plt.xlabel("Grid Column"); plt.ylabel("Grid Row")
    plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.xlim(0, GRID_SIZE); plt.ylim(0, GRID_SIZE); plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"generated_trajectory_cfg_scale{GUIDANCE_SCALE}.png") # Save with scale in name
    print(f"Plot saved to generated_trajectory_cfg_scale{GUIDANCE_SCALE}.png")
    plt.show()
