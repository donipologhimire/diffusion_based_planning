import numpy as np
import heapq
import matplotlib.pyplot as plt
import random
import pickle
import os

GRID_SIZE = 50
MAX_TRAJ_LEN = int(GRID_SIZE * 1.5) # Maximum length for padding/truncating
NUM_TRAJECTORIES = 5 # Number of expert trajectories to generate
OUTPUT_FILE = 'expert_trajectories.pkl'

# Define obstacle environment (Refined from user input)
def create_obstacle_map(grid_size=GRID_SIZE):
    """
    Create a binary obstacle map where 1 represents obstacles and 0 represents free space
    """
    obstacle_map = np.zeros((grid_size, grid_size), dtype=int)

    # Add walls (Ensure indices are integers and within bounds)
    wall_width = 1

    # Horizontal walls
    h1_start = max(0, grid_size // 3 - wall_width)
    h1_end = min(grid_size, grid_size // 3 + wall_width)
    h1_len = min(grid_size, grid_size // 2 - 8)
    if h1_len > 0 : obstacle_map[h1_start:h1_end, :h1_len] = 1

    h2_start = max(0, 2 * grid_size // 3 - wall_width)
    h2_end = min(grid_size, 2 * grid_size // 3 + wall_width)
    h2_col_start = max(0, grid_size // 2 + 8)
    h2_col_end = min(grid_size, 43)
    if h2_col_start < h2_col_end: obstacle_map[h2_start:h2_end, h2_col_start:h2_col_end] = 1

    # Vertical walls
    v1_row_start = max(0, 5)
    v1_row_end = min(grid_size, grid_size - 5)
    v1_start = max(0, grid_size // 3 - wall_width)
    v1_end = min(grid_size, grid_size // 3 + wall_width)
    if v1_row_start < v1_row_end: obstacle_map[v1_row_start:v1_row_end, v1_start:v1_end] = 1

    v2_row_start = max(0, grid_size // 2 + 7)
    v2_start = max(0, (2 * grid_size) // 3 - wall_width)
    v2_end = min(grid_size, (2 * grid_size) // 3 + wall_width)
    if v2_row_start < grid_size: obstacle_map[v2_row_start:, v2_start:v2_end] = 1

    # Add boundary walls
    obstacle_map[0, :] = 1
    obstacle_map[-1, :] = 1
    obstacle_map[:, 0] = 1
    obstacle_map[:, -1] = 1

    return obstacle_map

# --- A* Implementation ---
def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(obstacle_map, start, goal):
    """Finds a path using A* search"""
    grid_size = obstacle_map.shape[0]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                 (1, 1), (1, -1), (-1, 1), (-1, -1)] # 8-connectivity

    close_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    oheap = [(f_score[start], start)] # Priority queue

    while oheap:
        current_f, current = heapq.heappop(oheap)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start) # Add start node
            return path[::-1] # Return reversed path

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = g_score[current] + (1.414 if abs(i)+abs(j) == 2 else 1) # Cost is sqrt(2) for diagonal

            # Check bounds and obstacles
            if not (0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size):
                continue
            if obstacle_map[neighbor[0], neighbor[1]] == 1:
                continue

            if neighbor in close_set and tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (f_score[neighbor], neighbor))

    print(f"Warning: Path not found from {start} to {goal}")
    return None # Path not found

# --- Trajectory Processing ---
def normalize_coordinates(coord, grid_size):
    """Normalize coordinates to [-1, 1] range."""
    # Center origin at grid center, scale to [-1, 1]
    return ( (coord[0] / (grid_size - 1)) * 2 - 1,
             (coord[1] / (grid_size - 1)) * 2 - 1 )

def denormalize_coordinates(norm_coord, grid_size):
     """Denormalize coordinates from [-1, 1] back to grid indices."""
     row = int(round(((norm_coord[0] + 1) / 2) * (grid_size - 1)))
     col = int(round(((norm_coord[1] + 1) / 2) * (grid_size - 1)))
     # Clamp to grid boundaries
     row = max(0, min(grid_size - 1, row))
     col = max(0, min(grid_size - 1, col))
     return (row, col)

def process_trajectory(path, max_len, grid_size, start_point=None, goal_point=None):
    """Normalize, pad or truncate the trajectory, and ensure it starts/ends at specified points."""
    # Normalize
    normalized_path = [normalize_coordinates(p, grid_size) for p in path]
    
    # Ensure the path starts and ends at the exact normalized start/goal points if provided
    if start_point is not None:
        norm_start = normalize_coordinates(start_point, grid_size)
        normalized_path[0] = norm_start
    
    if goal_point is not None:
        norm_goal = normalize_coordinates(goal_point, grid_size)
        normalized_path[-1] = norm_goal

    current_len = len(normalized_path)
    processed_traj = np.zeros((max_len, 2), dtype=np.float32) # (row, col) format -> 2D

    if current_len >= max_len:
        # Truncate, but always keep start and goal
        if max_len >= 2:
            processed_traj[0] = normalized_path[0]  # Keep start
            if max_len > 2:
                # Include middle points if possible
                step = (current_len - 2) / (max_len - 2) if max_len > 2 else 0
                for i in range(1, max_len - 1):
                    idx = min(int(i * step) + 1, current_len - 2)
                    processed_traj[i] = normalized_path[idx]
            processed_traj[-1] = normalized_path[-1]  # Keep goal
        else:
            processed_traj = np.array(normalized_path[:max_len], dtype=np.float32)
    else:
        # Pad with the last state
        processed_traj[:current_len, :] = np.array(normalized_path, dtype=np.float32)
        processed_traj[current_len:, :] = normalized_path[-1] # Pad with last coordinate

    return processed_traj


# --- Main Generation Logic ---
if __name__ == "__main__":
    print("Generating expert trajectories...")
    obstacle_map = create_obstacle_map(GRID_SIZE)
    expert_trajectories = []
    conditions = [] # Store start, goal, maybe obstacles for conditioning

    free_cells = list(zip(*np.where(obstacle_map == 0)))
    if not free_cells:
        raise ValueError("No free cells available in the grid!")

    generated_count = 0
    attempts = 0
    max_attempts = NUM_TRAJECTORIES * 5 # Try harder to find paths

    while generated_count < NUM_TRAJECTORIES and attempts < max_attempts:
        attempts += 1
        # Select random valid start/goal points
        start_point = (4, 45) #random.choice(free_cells)
        goal_point = (43, 5)#random.choice(free_cells)

        # Ensure start != goal
        if start_point == goal_point:
            continue

        print(f"Attempt {attempts}: Finding path from {start_point} to {goal_point}...")
        path = a_star_search(obstacle_map, start_point, goal_point)

        if path and len(path) > 1: # Found a valid path
            processed_traj = process_trajectory(path, MAX_TRAJ_LEN, GRID_SIZE, 
                                                start_point=start_point, goal_point=goal_point)
            expert_trajectories.append(processed_traj)

            # Store conditions: normalized start/goal and flattened obstacle map
            norm_start = normalize_coordinates(start_point, GRID_SIZE)
            norm_goal = normalize_coordinates(goal_point, GRID_SIZE)
            # Flatten obstacle map (consider alternatives for large maps)
            flat_obstacles = obstacle_map.flatten().astype(np.float32)
            conditions.append({
                'start': np.array(norm_start, dtype=np.float32),
                'goal': np.array(norm_goal, dtype=np.float32),
                'obstacles': flat_obstacles
            })
            generated_count += 1
            print(f"  -> Path found and processed. Total generated: {generated_count}/{NUM_TRAJECTORIES}")
        # else:
            # print(f"  -> No path found or path too short.")

    if generated_count < NUM_TRAJECTORIES:
        print(f"\nWarning: Only generated {generated_count} trajectories out of the desired {NUM_TRAJECTORIES} after {max_attempts} attempts.")
    else:
        print(f"\nSuccessfully generated {generated_count} trajectories.")

    # Save the data
    if expert_trajectories:
        print(f"Saving data to {OUTPUT_FILE}...")
        # Check if directory exists, create if not (optional)
        # os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        data_to_save = {
            'trajectories': np.array(expert_trajectories), # Shape: (N, MAX_TRAJ_LEN, 2)
            'conditions': conditions, # List of dicts
            'grid_size': GRID_SIZE,
            'max_traj_len': MAX_TRAJ_LEN
        }
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(data_to_save, f)
        print("Data saved.")

        # --- Optional: Visualize one example ---
        plt.figure(figsize=(8, 8))
        plt.imshow(obstacle_map, cmap='gray_r', origin='lower', extent=(0, GRID_SIZE, 0, GRID_SIZE)) # Use 'lower' for (0,0) at bottom-left
        # Denormalize first trajectory for plotting
        example_traj_norm = expert_trajectories[0]
        example_traj_denorm = [denormalize_coordinates(p, GRID_SIZE) for p in example_traj_norm]
        path_rows, path_cols = zip(*example_traj_denorm)
        plt.plot(np.array(path_cols) + 0.5 , np.array(path_rows) + 0.5, marker='.', color='cyan', linestyle='-', linewidth=1.5) # Add 0.5 to center in cells
        start_idx = denormalize_coordinates(conditions[0]['start'], GRID_SIZE)
        goal_idx = denormalize_coordinates(conditions[0]['goal'], GRID_SIZE)
        plt.plot(start_idx[1] + 0.5, start_idx[0] + 0.5, 'go', markersize=10, label='Start')
        plt.plot(goal_idx[1] + 0.5, goal_idx[0] + 0.5, 'ro', markersize=10, label='Goal')
        plt.title(f"Example Generated Expert Trajectory (A*)")
        plt.xlabel("Grid Column")
        plt.ylabel("Grid Row")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(0, GRID_SIZE)
        plt.ylim(0, GRID_SIZE)
        plt.show()
    else:
        print("No trajectories were generated. Cannot save or visualize.")