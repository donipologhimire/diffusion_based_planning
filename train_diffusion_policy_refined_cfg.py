import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import copy # For saving best model state

# --- Constants and Configuration ---
DATA_FILE = 'expert_trajectories.pkl'
# !!! New model save path !!!
MODEL_SAVE_PATH = 'diffusion_policy_unet_refined_cfg.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Add a constant at the top of your file with other constants
CHECKPOINT_FREQ = 1  # Save checkpoint every 10 epochs
CHECKPOINT_DIR = "checkpoints"  # Directory to store periodic checkpoints

# --- Diffusion Hyperparameters ---
N_DIFFUSION_STEPS = 100
BETA_START = 1e-4
BETA_END = 0.02

# --- Training Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
N_EPOCHS = 9
VALIDATION_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 20
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
# !!! CFG Hyperparameter !!!
CFG_PROB_UNCOND = 0.15 # Probability of dropping conditions during training (e.g., 10-20%)

# --- Diffusion Utilities (Precompute schedules) ---
betas = torch.linspace(BETA_START, BETA_END, N_DIFFUSION_STEPS, device=DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.long())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- Dataset (Identical to previous version) ---
class TrajectoryDataset(Dataset):
    def __init__(self, data_file):
        try:
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Data loaded successfully from {data_file}")
        except FileNotFoundError:
            print(f"Error: Data file {data_file} not found.")
            raise
        except Exception as e:
            print(f"Error loading data file {data_file}: {e}")
            raise

        required_keys = ['trajectories', 'conditions', 'grid_size', 'max_traj_len']
        if not all(key in self.data for key in required_keys):
            raise ValueError(f"Data file {data_file} is missing required keys: {required_keys}")

        self.trajectories = torch.tensor(self.data['trajectories'], dtype=torch.float32)
        self.conditions = self.data['conditions']
        self.grid_size = self.data['grid_size']
        self.max_traj_len = self.data['max_traj_len']

        if len(self.trajectories) != len(self.conditions):
             raise ValueError("Mismatch between number of trajectories and conditions.")

        self.state_dim = self.trajectories.shape[2]
        expected_traj_shape = (len(self.conditions), self.max_traj_len, self.state_dim)
        if self.trajectories.shape != expected_traj_shape:
             print(f"Warning: Trajectories shape {self.trajectories.shape} differs from expected {expected_traj_shape}")

        self.start_goal_coords = []
        self.obstacle_maps = []
        expected_obs_flat_len = self.grid_size * self.grid_size
        for i, cond in enumerate(self.conditions):
            if not isinstance(cond, dict) or 'start' not in cond or 'goal' not in cond or 'obstacles' not in cond:
                 raise ValueError(f"Condition at index {i} is malformed: {cond}")

            start = np.array(cond['start'], dtype=np.float32).flatten()
            goal = np.array(cond['goal'], dtype=np.float32).flatten()
            if start.shape != (2,) or goal.shape != (2,):
                raise ValueError(f"Start/Goal shape error at index {i}. Got {start.shape}, {goal.shape}. Expected (2,).")
            sg_vec = np.concatenate([start, goal])
            self.start_goal_coords.append(sg_vec)

            obstacles = np.array(cond['obstacles'], dtype=np.float32).flatten()
            if obstacles.shape[0] != expected_obs_flat_len:
                 raise ValueError(f"Obstacle map size error at index {i}. Got {obstacles.shape[0]} elements, expected {expected_obs_flat_len}.")
            obstacle_map_reshaped = obstacles.reshape(self.grid_size, self.grid_size)
            self.obstacle_maps.append(obstacle_map_reshaped)

        self.start_goal_coords = torch.tensor(np.array(self.start_goal_coords), dtype=torch.float32)
        self.obstacle_maps = torch.tensor(np.array(self.obstacle_maps), dtype=torch.float32)

        if self.start_goal_coords.shape != (len(self.conditions), 4):
             raise ValueError(f"Final start_goal_coords shape is incorrect: {self.start_goal_coords.shape}")
        if self.obstacle_maps.shape != (len(self.conditions), self.grid_size, self.grid_size):
             raise ValueError(f"Final obstacle_maps shape is incorrect: {self.obstacle_maps.shape}")

        print(f"Dataset initialized: {len(self.trajectories)} trajectories.")
        print(f"  Trajectory shape: {self.trajectories.shape}")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Max trajectory length: {self.max_traj_len}")
        print(f"  Grid size: {self.grid_size}")
        print(f"  Start/Goal Coords shape: {self.start_goal_coords.shape}")
        print(f"  Obstacle Maps shape: {self.obstacle_maps.shape}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.start_goal_coords[idx], self.obstacle_maps[idx]

# --- Model Architecture: Conditional 1D U-Net (Refined) ---

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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- Training and Validation Loops ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch_num, total_epochs, cfg_prob_uncond):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} [Train]", leave=False)
    for batch_traj, batch_sg, batch_obs in pbar:
        batch_traj, batch_sg, batch_obs = batch_traj.to(DEVICE), batch_sg.to(DEVICE), batch_obs.to(DEVICE)
        batch_size = batch_traj.shape[0]

        # Sample random timesteps
        t = torch.randint(0, N_DIFFUSION_STEPS, (batch_size,), device=DEVICE).long()

        # Sample noise and create noisy trajectory
        noise = torch.randn_like(batch_traj)
        x_noisy = q_sample(x_start=batch_traj, t=t, noise=noise)

        # --- Classifier-Free Guidance Training ---
        # Create null conditions (zeros)
        null_sg = torch.zeros_like(batch_sg)
        null_obs = torch.zeros_like(batch_obs)

        # Randomly decide which samples in the batch will be unconditional
        uncond_mask = torch.rand(batch_size, device=DEVICE) < cfg_prob_uncond

        # Select effective conditions based on the mask
        # Need to reshape mask for broadcasting with conditions
        effective_sg = torch.where(uncond_mask.view(batch_size, 1), null_sg, batch_sg)
        # Obstacle map needs mask shape (B, 1, 1) to broadcast to (B, H, W)
        effective_obs = torch.where(uncond_mask.view(batch_size, 1, 1), null_obs, batch_obs)
        # -----------------------------------------

        # Predict noise using the model with potentially dropped conditions
        predicted_noise = model(x_noisy, t, effective_sg, effective_obs) # Use effective conditions

        # Calculate standard diffusion loss
        loss = loss_fn(noise, predicted_noise)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_one_epoch(model, dataloader, loss_fn, epoch_num, total_epochs):
    # Validation can often be done without CFG dropout for simplicity,
    # measuring the model's raw conditional performance.
    # If you want validation loss to reflect CFG training, apply the same
    # condition dropping logic here.
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num}/{total_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch_traj, batch_sg, batch_obs in pbar:
            batch_traj, batch_sg, batch_obs = batch_traj.to(DEVICE), batch_sg.to(DEVICE), batch_obs.to(DEVICE)
            batch_size = batch_traj.shape[0]

            t = torch.randint(0, N_DIFFUSION_STEPS, (batch_size,), device=DEVICE).long()
            noise = torch.randn_like(batch_traj)
            x_noisy = q_sample(x_start=batch_traj, t=t, noise=noise)

            # Standard validation: Always use real conditions
            predicted_noise = model(x_noisy, t, batch_sg, batch_obs)

            loss = loss_fn(noise, predicted_noise)
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# --- Main Training Script ---
if __name__ == "__main__":
    # --- 1. Load and Prepare Data ---
    try:
        full_dataset = TrajectoryDataset(DATA_FILE)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        exit()

    val_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    if train_size <= 0 or val_size <= 0:
         raise ValueError("Dataset too small for the specified validation split.")
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True)
    print(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- 2. Initialize Model, Optimizer, Scheduler, Loss ---
    # Ensure these parameters match your intended architecture
    model = ConditionalUNet1D(
        in_channels=full_dataset.state_dim,
        max_traj_len=full_dataset.max_traj_len,
        grid_size=full_dataset.grid_size,
        base_dim=64,
        dim_mults=(1, 2, 4, 8),
        time_emb_dim=128,
        start_goal_dim=4,
        obstacle_emb_dim=128,
        cond_emb_extra_dim=128, # Make sure this matches definition
        attn_levels=(2, 3)      # Make sure this matches definition
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} parameters.")
    print(f"Training with Classifier-Free Guidance (Unconditional Probability: {CFG_PROB_UNCOND})")


    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    loss_fn = nn.MSELoss()

    # --- 3. Training Loop with Validation and Early Stopping ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    start_epoch = 0

    # Optional: Load existing model (ensure it was trained with CFG or restart)
    # if os.path.exists(MODEL_SAVE_PATH) and False: # Set to True to enable loading
    #     ... (loading logic - be careful if resuming non-CFG model)

    print(f"\nStarting CFG training from epoch {start_epoch} for max {N_EPOCHS} epochs...")
    for epoch in range(start_epoch, N_EPOCHS):
        # Training with CFG dropout probability
        train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch + 1, N_EPOCHS, CFG_PROB_UNCOND)
        history['train_loss'].append(train_loss)

        # Validation (standard conditional loss)
        val_loss = validate_one_epoch(model, val_dataloader, loss_fn, epoch + 1, N_EPOCHS)
        history['val_loss'].append(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1}/{N_EPOCHS} -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6e}")

        scheduler.step(val_loss)

        # Checkpoint Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'cfg_prob_uncond': CFG_PROB_UNCOND # Save CFG param too
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"   -> New best validation loss! Model checkpoint saved to {MODEL_SAVE_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"   -> Val loss did not improve ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE}). Best: {best_val_loss:.6f}")

        
        # Add periodic checkpoint saving
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            periodic_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            periodic_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Current model state, not necessarily the best
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,  # Current validation loss
                'history': history,
                'cfg_prob_uncond': CFG_PROB_UNCOND
            }
            torch.save(periodic_checkpoint, periodic_checkpoint_path)
            print(f"   -> Periodic checkpoint saved at epoch {epoch+1} to {periodic_checkpoint_path}")


        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    print("\nTraining finished.")
    print(f"Best Validation Loss achieved: {best_val_loss:.6f}")
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Best CFG model checkpoint saved to {MODEL_SAVE_PATH}")

    # --- 4. Plot Loss History ---
    plt.figure(figsize=(12, 6))
    # ... (Plotting code remains the same) ...
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE Loss"); ax1.set_title("Training and Validation Loss")
    ax1.legend(); ax1.grid(True)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(history['lr'], label='Learning Rate', color='orange')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate"); ax2.set_title("Learning Rate Schedule")
    ax2.legend(); ax2.grid(True); ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig('training_plot_refined_cfg.png') # New plot name
    print("Training plot saved to training_plot_refined_cfg.png")
    # plt.show()
