import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch
from learning import NeuralNetwork

# Importo librerie acados/modelli
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from lidar import get_lidar_hits_2d, min_cube_select_base

# def simulate_lidar_2d(drone_x, drone_y, obstacles, num_rays=36, max_range=5.0):
#     """Simulate a 360-degree LIDAR and return the distance to the nearest obstacle."""
#     min_dist_global = max_range
#     angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
#     for angle in angles:
#         dx = np.cos(angle)
#         dy = np.sin(angle)
#         ray_min_dist = max_range
#         for obs in obstacles:
#             x_min, x_max, y_min, y_max = obs
#             t_x1 = (x_min - drone_x) / (dx + 1e-6)
#             t_x2 = (x_max - drone_x) / (dx + 1e-6)
#             t_y1 = (y_min - drone_y) / (dy + 1e-6)
#             t_y2 = (y_max - drone_y) / (dy + 1e-6)
#             t_min_x, t_max_x = min(t_x1, t_x2), max(t_x1, t_x2)
#             t_min_y, t_max_y = min(t_y1, t_y2), max(t_y1, t_y2)
#             t_enter = max(t_min_x, t_min_y)
#             t_exit = min(t_max_x, t_max_y)
#             if t_enter <= t_exit and t_exit >= 0:
#                 dist = t_enter if t_enter > 0 else 0
#                 if dist < ray_min_dist:
#                     ray_min_dist = dist
#         if ray_min_dist < min_dist_global:
#             min_dist_global = ray_min_dist
#     return min_dist_global

def get_ambiente(id_stanza):
    """Return obstacles, target, and title for the selected room."""
    if id_stanza == 1:
        # ROOM 1: Hallway (free space above and below)
        obstacles = [
            [1.5, 4.5, 1.0, 3.0],    # Ceiling
            [1.5, 4.5, -5.0, -3.5]   # Floor
        ]
        x_ref = np.array([5.0, -1.5, 0.0, 0.0, 0.0, 0.0])
        titolo = "Hallway"
         
    elif id_stanza == 2:
        # ROOM 2: Central front wall (dead end)
        obstacles = [
            [2.0, 3.0, -3.0, 2.0]    # Large central wall
        ]
        x_ref = np.array([4.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        titolo = "Front Wall"
         
        
    elif id_stanza == 3:
        # ROOM 3: Staggered walls (slalom)
        obstacles = [
            [1.0, 2.0, -1.0, 3.0],   # Descending wall
            [3.0, 4.0, -5.0, -1.5]   # Ascending wall
        ]
        x_ref = np.array([5.0, -2.5, 0.0, 0.0, 0.0, 0.0])
        titolo = "Staggered Walls"

    elif id_stanza == 4:
        # ROOM 4: Oblique wall (simulated with an AABB block sequence)
        # The wall spans diagonally from X=3, Z=1 to X=6, Z=5
        obstacles = []
        num_blocchi = 500 # More blocks make the surface look smoother (and harder for the AABB)
        x_vals = np.linspace(1.5, 3.0, num_blocchi)
        z_vals = np.linspace(-1.5, 1.5, num_blocchi)
         
        for i in range(num_blocchi):
            # Build small 0.02x0.02 cubes along the diagonal
            # Format: [x_min, x_max, y_min, y_max, z_min, z_max]
            obstacles.append([x_vals[i]-0.01, x_vals[i]+0.01, z_vals[i]-0.01, z_vals[i]+0.01])

        x_ref = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])   
        titolo = "Room 4: Oblique Wall"
    else:
        obstacles = []
        x_ref = np.array([5.0, -1.5, 0.0, 0.0, 0.0, 0.0])
        titolo = "Empty Room"
        
    return obstacles, x_ref, titolo

def plot_mappa_2d(x_history, box_history, obstacles, x_ref, x0, titolo):
    """Draw the full 2D map with trajectory and obstacles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Draw obstacles
    for obs in obstacles:
        x_min, x_max, z_min, z_max = obs
        width = x_max - x_min
        height = z_max - z_min
        rect = patches.Rectangle((x_min, z_min), width, height, linewidth=1, 
                                 edgecolor='black', facecolor='gray', alpha=0.6)
        ax.add_patch(rect)
        
    # 2. Draw the trajectory
    x_hist = np.array(x_history)
    ax.plot(x_hist[:, 0], x_hist[:, 1], color='blue', label='Trajectory', 
            linewidth=1.5, marker='o', markersize=4)
    
    # 3. Draw the asymmetric box computed by Max (every 10 steps to avoid overload)
    for t in range(0, len(box_history), 10):
        b_xmin, b_xmax, b_zmin, b_zmax = box_history[t]
        width = b_xmax - b_xmin
        height = b_zmax - b_zmin
        drone_box = patches.Rectangle((b_xmin, b_zmin), width, height, 
                                      linewidth=1, edgecolor='green', facecolor='lime', alpha=0.1)
        ax.add_patch(drone_box)
        
    # Start and Target
    ax.scatter(x0[0], x0[1], color='green', s=120, label='Start', zorder=5)
    ax.scatter(x_ref[0], x_ref[1], color='red', s=120, label='Target', marker='X', zorder=5)

    ax.set_aspect('equal', 'box')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-6, 4])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(titolo)
    ax.grid(True, linestyle=':', alpha=0.7)
    #ax.legend(loc='upper right')
    plt.show()

def main():
    print("--- Initializing Room Test Campaign ---")
    params = Parameters('sth')
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)

    DT = params.dt
    SIM_TIME = 5.5
    N_SIM = int(SIM_TIME / DT)

    # LOOP OVER THE 4 ROOMS
    for STANZA_ID in [1, 2, 3, 4]:
        obstacles, x_ref, titolo_stanza = get_ambiente(STANZA_ID)
        
        print(f"\n=============================================")
        print(f"TEST: {titolo_stanza}")
        print(f"=============================================")

        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        current_x = x0.copy()

        x_history = [x0]
        box_history = []  # Now save the full box, not just alpha

        u_hover = (model.mass * 9.81) / (2.0 * model.cf)
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x0, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        # stall detection
        contatore_stallo = 0
        MAX_STALLO_ITER = 50

        for t in range(N_SIM):
            
            # 1. Simulate LIDAR and create tangent rays
            hits, radii = get_lidar_hits_2d(current_x[0], current_x[1], obstacles, num_rays=360, max_range=2.0)
            
            # 2. Max algorithm
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]

            # -- NEW: Compute target relative position --
            target_rel_x = x_ref[0] - current_x[0]
            target_rel_z = x_ref[1] - current_x[1]

            # Pass relative parameters to the function
            xMin_rel, xMax_rel, zMin_rel, zMax_rel, _ = min_cube_select_base(
                Q_rel, 
                radii, 
                target_rel_x, 
                target_rel_z, 
                drone_radius=0.1
            )
            
            # 3. Convert the box to absolute coordinates
            box_abs = np.array([
                xMin_rel + current_x[0],
                xMax_rel + current_x[0],
                zMin_rel + current_x[1],
                zMax_rel + current_x[1]
            ])

            

            # 4. MPC computes using the 4 absolute room sides
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, box_abs)
            
            # # Compute minimum distance to walls for terminal logging
            # dist_x_min = current_x[0] - box_abs[0]
            # dist_x_max = box_abs[1] - current_x[0]
            # dist_z_min = current_x[1] - box_abs[2]
            # dist_z_max = box_abs[3] - current_x[1]
            # min_dist_to_wall = min(dist_x_min, dist_x_max, dist_z_min, dist_z_max)

            if status in [3, 4]:
                # print(f"\n⚠️ Teleported wall (Box Flip)! Solver panicked. Starting Recovery Mode...")
                
                # # Reset solver memory (Warm Start)
                controller.ocp_solver.reset()
                # Tell it to assume the drone stays still where it is
                controller.x_guess = np.tile(current_x, (controller.N, 1))
                controller.u_guess = np.full((controller.N, model.nu), u_hover)
                
                # Retry solving with a clean local estimate
                x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, box_abs)
                
                if status in [3, 4]:
                    print(f"❌ Recovery failed. The drone is physically trapped. Closing.")
                    break

            current_x = x_sol[1] 
            x_history.append(current_x)
            box_history.append(box_abs)

        # # ==========================================
        # # STALL HANDLING (Local Minimum Escape)
        # # ==========================================
        # # Check movement compared to previous step


        # if t > 0:
        #     spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            
        #     if spostamento < 0.01:  # If it moved less than 1 cm
        #         contatore_stallo += 1
        #     else:
        #         contatore_stallo = 0  # Reset the counter if it escapes

        #     # If stuck for 50 iterations, act
        #     if contatore_stallo >= MAX_STALLO_ITER:
        #         print(f"\n⚠️ STALL DETECTED (Step {t})! The drone is trapped in a local minimum.")
        #         print("   -> Perturb the local target upward by 20 cm...")
                
        #         # Raise the current waypoint Z by 20 cm
        #         waypoints[target_idx][1] += 0.20 
                
        #         # Update the variable used in the loop for the next step
        #         x_ref_attuale = waypoints[target_idx] 
                
        #         # Reset the counter to give it time to move
        #         contatore_stallo = 0 
        # # ==========================================


        # Plot the room before moving to the next one
        plot_mappa_2d(x_history, box_history, obstacles, x_ref, x0, titolo_stanza)

if __name__ == '__main__':
    main()