import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Import your libraries
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from lidar import get_lidar_hits_2d, get_lidar_hits_2d_qualsiasi, min_cube_select_base

# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for all remaining text
})

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches



def genera_caverna():
    """Defines the cave obstacles and waypoints."""

    # --- OBSTACLE DEFINITION ---
    ostacoli = []
    ostacoli.append([-2.0, 28.0, -2.0, -1.0])  # Floor
    ostacoli.append([-2.0, 28.0, 7.0, 8.0])    # Ceiling
    ostacoli.append([3.6, 4.5, -1.0, 3.6])     # Stalactite 1
    ostacoli.append([6.0, 10.0, 4.2, 7.0])     # Stalactite 2
    ostacoli.append([5.0, 12.0, -1.0, 3.0])    # Stalactite 3
    ostacoli.append([11.0, 15.0, 4.7, 7.0])    # Stalactite 4
    ostacoli.append([13.0, 14.5, -1.0, 3.6])   # Stalactite 5
    ostacoli.append([15.6, 18.0, 4.2, 7.0])    # Stalactite 6
    ostacoli.append([15.0, 20.0, -1.0, 3.1])   # Stalactite 7

    # --- WAYPOINT DEFINITION (Line of Sight) ---
    waypoints = [
        # np.array([5.2, 3.6, 0.0, 0.0, 0.0, 0.0]),   # WP1
        # np.array([10.0, 3.5, 0.0, 0.0, 0.0, 0.0]),  # WP2
        # np.array([15.0, 4.0, 0.0, 0.0, 0.0, 0.0]),  # WP3
        np.array([22.0, 3.0, 0.0, 0.0, 0.0, 0.0]),  # WP4
    ]
    return ostacoli, waypoints

# def genera_labirinto():
#     """ Complex 2D maze with a long sliding section """
#     ostacoli = []
    
#     # Outer edges (Floor and Ceiling) - extended to 45m
#     ostacoli.append([-2.0, 45.0, -2.0, 0.0])   # Floor
#     ostacoli.append([-2.0, 45.0, 10.0, 12.0])  # Ceiling

#     # Obstacle 1: Stalactite (forces passage below)
#     ostacoli.append([3.0, 4.0, 4.0, 10.0])
    
#     # Obstacle 2: Stalagmite (forces passage above)
#     ostacoli.append([7.0, 8.0, 0.0, 6.0])

#     # Obstacle 3: Floating central block (we choose to pass underneath)
#     ostacoli.append([11.0, 14.0, 4.0, 6.0])

#     # Obstacle 4: Funnel-shaped horizontal constriction
#     ostacoli.append([17.0, 19.0, 0.0, 3.0])    # Constriction floor
#     ostacoli.append([17.0, 19.0, 7.0, 10.0])   # Constriction ceiling

#     # Obstacle 5: Very tall wall (forces ceiling-skimming flight)
#     ostacoli.append([22.0, 23.0, 0.0, 8.0])

#     # Obstacle 6: Very low wall (forces low-level flight near the floor)
#     ostacoli.append([26.0, 27.0, 2.0, 10.0])

#     # Obstacle 7: Long final tunnel
#     ostacoli.append([30.0, 36.0, 0.0, 4.0])    # Raised floor
#     ostacoli.append([30.0, 36.0, 6.0, 10.0])   # Lowered ceiling

#     # --- WAYPOINT DEFINITION (Simulated global path planning) ---
#     waypoints = [
#         np.array([1.5, 2.0, 0.0, 0.0, 0.0, 0.0]),   # WP1: Descend through the first gap
#         np.array([5.0, 2.0, 0.0, 0.0, 0.0, 0.0]),   # WP2: Beyond the first stalactite
#         np.array([6.0, 8.0, 0.0, 0.0, 0.0, 0.0]),   # WP3: Climb above the stalagmite
#         np.array([9.0, 8.0, 0.0, 0.0, 0.0, 0.0]),   # WP4: Beyond the stalagmite
#         np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # WP5: Descend to pass under the island
#         np.array([15.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # WP6: Under the floating island
#         np.array([16.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP7: Climb and center for the constriction
#         np.array([20.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP8: Out of the constriction
#         np.array([21.0, 9.0, 0.0, 0.0, 0.0, 0.0]),  # WP9: Ceiling-skimming climb
#         np.array([24.0, 9.0, 0.0, 0.0, 0.0, 0.0]),  # WP10: Beyond the tall wall
#         np.array([25.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # WP11: Fast low-level dive
#         np.array([28.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # WP12: Beyond the low wall
#         np.array([29.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP13: Climb and center for the long tunnel
#         np.array([37.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP14: Tunnel exit
#         np.array([42.0, 5.0, 0.0, 0.0, 0.0, 0.0])   # WP15: Final target
#     ]

#     return ostacoli, waypoints

def genera_labirinto():
    """2D maze with narrow corridors and 90-degree turns (Pipe Maze)."""
    ostacoli = []
    
    # Outer edges (Floor and Ceiling)
    ostacoli.append([-2.0, 22.0, -2.0, -0.5])   # Base floor
    ostacoli.append([-2.0, 22.0, 9.5, 11.0])    # Base ceiling

    # BLOCK 1: Closes top and bottom at the start, leaving only the initial corridor
    ostacoli.append([-2.0, 3.5, -0.5, 4.0])
    ostacoli.append([-2.0, 3.5, 6.0, 9.5])

    # WELL 1 (X: 3.5 -> 5.0) - The drone must descend here

    # BLOCK 2: Central wall forcing the drone into the lower tunnel
    ostacoli.append([5.0, 9.0, 2.5, 9.5])
    # BLOCK 3: Raised floor in the lower tunnel
    ostacoli.append([5.0, 9.0, -0.5, 0.5])

    # WELL 2 (X: 9.0 -> 10.5) - The drone must climb back up here

    # BLOCK 4: Central wall forcing the drone into the upper tunnel
    ostacoli.append([10.5, 15.0, -0.5, 6.5])
    # BLOCK 5: Lowered ceiling in the upper tunnel
    ostacoli.append([10.5, 15.0, 8.5, 9.5])

    # WELL 3 (X: 15.0 -> 16.5) - Final 90-degree drop

    # BLOCK 6: Final closure
    ostacoli.append([16.5, 20.0, 4.5, 9.5])
    ostacoli.append([16.5, 20.0, -0.5, 1.5])

    # --- WAYPOINT DEFINITION (90-degree navigation) ---
    waypoints = [
        # np.array([4.25, 5.0, 0.0, 0.0, 0.0, 0.0]),   # WP1
        np.array([7.25, 1.5, 0.0, 0.0, 0.0, 0.0]),   # WP2
        #np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0]),   # WP3
        # np.array([12.0, 7.2, 0.0, 0.0, 0.0, 0.0]),  # WP3
        # np.array([9.75, 7.5, 0.0, 0.0, 0.0, 0.0]),  # WP4
        # np.array([15.75, 5.5, 0.0, 0.0, 0.0, 0.0]),  # WP5
        np.array([15.75, 2.5, 0.0, 0.0, 0.0, 0.0]),  # WP6
        np.array([25.0, 3.0, 0.0, 0.0, 0.0, 0.0]) 
    ]

    return ostacoli, waypoints


def genera_ambiente_obliquo():
    """
    Environment with generic obstacles (inclined walls, diamonds, triangles).
    Returns a flat list of segments (walls) and the waypoints.
    """
    # Define obstacles as closed polygons (list of vertices [X, Z])
    poligoni = [
        # 1. Irregular floor (slight slope toward the end)
        [[-2.0, -2.0], [25.0, -2.0], [25.0, 2.0], [-2.0, -0.5]],
        
        # 2. Oblique ceiling (descends then rises)
        [[-2.0, 10.0], [8.0, 7.0], [25.0, 9.0], [25.0, 9.0], [-2.0, 12.0]],
        
        # 3. Central obstacle: floating diamond/lozenge
        [[10.0, 3.5], [11.5, 5.0], [10.0, 6.5], [8.5, 5.0]],
        
        # 4. A skewed triangular stalactite
        [[16.0, 4.5], [14.0, 3.0], [17.5, 4.0]]
    ]
    
    # Break polygons into independent segments for the LiDAR
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            # Connect the current vertex to the next one (modulo % closes the shape)
            punto_A = poli[i]
            punto_B = poli[(i + 1) % n]
            segments.append([punto_A, punto_B])
            
    # Waypoints for navigation through these irregular obstacles
    waypoints = [
        # np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
        # np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # Pass under the diamond
        # np.array([16.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # Avoid the skewed stalactite
        np.array([22.0, 5.0, 0.0, 0.0, 0.0, 0.0])   # Final target
    ]
    
    return segments, waypoints

def main():
    print("--- Avvio Navigazione Multi-Target (Waypoints) ---")
    params = Parameters('sth')
    params.act = 'gelu'
    params.build = True 

    model = Model(params)
    controller = MpcController(model)

    DT = params.dt
    SIM_TIME = 40.0 # Extended time to cover all targets
    N_SIM = int(SIM_TIME / DT)

    
    target_idx = 0
    TOLLERANZA_WAYPOINT = 0.20

    # maze initial state
    x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    # cave state
    #x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    ostacoli, waypoints = genera_ambiente_obliquo()
    current_x = x0.copy()

    x_history = [current_x]
    box_history = []
    u_history = []
    
    # Solver initialization
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    # stall detection
    contatore_stallo = 0
    MAX_STALLO_ITER = 50

    # recovery management
    in_recovery = False
    timer_recovery = 0
    target_recovery = None

    ghost_waypoints = []       # Remember the old waypoints displaced during a stall
    mode_history = ['normal']  # Remember whether the system was in recovery at that instant

    print(f"Inizio volo verso Waypoint {target_idx + 1}...")

    for t in range(N_SIM):
       # 0. Select the current target
        if in_recovery:
            x_ref_attuale = target_recovery
            timer_recovery -= 1
            if timer_recovery <= 0:
                in_recovery = False
                # END OF EMERGENCY:
                # Restore the neural network constraints
                controller.ocp_solver.constraints_set(controller.N, "lh", np.zeros(4))
                
                lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                
                # Restore velocity bounds
                lbx_e_curr[3:] = [-1.0, -1.0, -1.0]
                ubx_e_curr[3:] = [ 1.0,  1.0,  1.0]
                
                controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)

                print(f"\n🔄 END RECOVERY: Mission restored toward WP {target_idx + 1}. Safety constraints reactivated.")
        else:
            x_ref_attuale = waypoints[target_idx]

        # 1. LiDAR and Safe-Box
        hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], ostacoli, num_rays=360, max_range=2.0)
        # max range 1.5 for cave without recovery help; 2.0 is fine if infeasibility handling is present; for maze 2.0 avoids seeing too close
        Q_rel = hits.copy()
        if len(hits) > 0:
            Q_rel[:, 0] -= current_x[0]
            Q_rel[:, 1] -= current_x[1]
        
        # Pass the current target's relative position to guide box expansion
        target_rel_x = x_ref_attuale[0] - current_x[0]
        target_rel_z = x_ref_attuale[1] - current_x[1]

        xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_base(
            Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1
        )
        box_abs = np.array([xMin_r + current_x[0], xMax_r + current_x[0], zMin_r + current_x[1], zMax_r + current_x[1]])

        box_history.append(box_abs.copy())

        # ==========================================
        # DIAGNOSTIC BLOCK (MPC DEBUGGER)
        # ==========================================
        if t % 10 == 0: # Print every 10 steps
            print(f"\n--- DEBUG STEP {t} ---")
            print(f"1. Drone Position : X={current_x[0]:.2f}, Z={current_x[1]:.2f}")
            print(f"2. Green Box (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Z in [{box_abs[2]:.2f}, {box_abs[3]:.2f}]")
            print(f"3. Local Target  : X={x_ref_attuale[0]:.2f}, Z={x_ref_attuale[1]:.2f}")
            
            
            # Compute the distance between the drone and the local target
            dist_to_local = np.linalg.norm(current_x[:2] - np.array([target_rel_x, target_rel_z])[:2])
            print(f"5. Distance remaining inside the box: {dist_to_local:.3f} meters")
        # ==========================================

        # 2. SOLVE MPC
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

        # ==========================================
        # INFEASIBILITY HANDLING (Status 3 or 4)
        # ==========================================

        if (status in [3, 4]):
            # ==========================================
            # PLAN A: Reset memory (Hovering Guess)
            # ==========================================
            controller.ocp_solver.reset()
            controller.x_guess = np.tile(current_x, (controller.N, 1))
            controller.u_guess = np.full((controller.N, model.nu), u_hover)
            
            x_sol, u_sol, alpha_curr, status =  controller.solve_step(current_x, x_ref_attuale, box_abs)

            # margin_safety = min_dist_to_wall - alpha_curr
            # danger = (margin_safety < 0.15) and (status in [0, 2])

            # ==========================================
            # PLAN B (NEW): Historical Warm-Start
            # ==========================================
            if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                print(f"\n⚠️ PLAN A FAILED. Starting PLAN B (backtracking through past controls)...")
                
                # Go backward from the last saved control to the first
                for i in range(len(u_history) - 1, -1, -1):
                    past_u = u_history[i]
                    
                    controller.ocp_solver.reset()
                    controller.x_guess = np.tile(current_x, (controller.N, 1))
                    controller.u_guess = past_u
                    
                    x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                    
                    if status in [0, 2]:
                        passi_indietro = len(u_history) - i
                        print(f"✅ PLAN B Succeeded! Found a valid warm-start {passi_indietro} steps ago.")
                        break

            # ==========================================
            # PLAN C: Retreat to Center
            # ==========================================
            
            if (status in [3, 4]) and not in_recovery:
                print(f"\n⚠️ PLAN B FAILED. Starting PLAN C (Retreat to center with relaxed alpha)...")
                
                passi_indietro = 10  # Number of steps back to take the safe box from
                if len(box_history) > passi_indietro:
                    box_sicuro = box_history[-passi_indietro]
                else:
                    box_sicuro = box_history[0]  # If it fails immediately, return to the start
                
                # Compute the center of the previous box
                center_x = (box_sicuro[0] + box_sicuro[1]) / 2.0
                center_z = (box_sicuro[2] + box_sicuro[3]) / 2.0
                
                target_recovery = np.array([center_x, center_z, 0.0, 0.0, 0.0, 0.0])
                x_ref_attuale = target_recovery
                
                in_recovery = True
                timer_recovery = 40 
                
                # ==========================================
                # CONSTRAINT RELAXATION (Realistic)
                # ==========================================
                controller.ocp_solver.constraints_set(controller.N, "lh", np.full(4, -0.10))
               
                lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                
                lbx_e_curr[3:] = [-5.0, -5.0, -5.0]
                ubx_e_curr[3:] = [ 5.0,  5.0,  5.0]
               
                controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                
                x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                
                if status in [3, 4]:
                    print("❌ RECOVERY 2 FAILED. The physical space is completely unavailable. Closing.")
                    break
                else:
                    print("✅ RECOVERY 2 Succeeded! The drone retreats toward the center temporarily ignoring the network.")
            
            elif status in [3, 4] and in_recovery:
                print("❌ THE DRONE IS IN A PHYSICAL TRAP. Cannot stabilize. Closing.")
                break
        # ==========================================

        if u_sol is not None:
            u_history.append(u_sol.copy())

        current_x = x_sol[1]
        x_history.append(current_x)

        mode_history.append('recovery' if in_recovery else 'normal')

        # ==========================================
        # STALL HANDLING (Local Minimum Escape)
        # ==========================================

        if t > 0:
            spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            
            if spostamento < 0.01:  # If moved less than 1 cm
                contatore_stallo += 1
            else:
                contatore_stallo = 0  # Reset the counter if it moves

            # If it remains stuck for 50 iterations, apply an escape
            if contatore_stallo >= MAX_STALLO_ITER:
                print(f"\n⚠️ STALL DETECTED (Step {t})! The drone is trapped in a local minimum.")
                print("   -> Perturbing the local target upward")
                
                # save the old constraint only for plotting
                ghost_waypoints.append(waypoints[target_idx].copy())

                # Raise the current waypoint Z
                # maze x
                #waypoints[target_idx][1] += 0.50 
                # for cave
                waypoints[target_idx][1] += 0.20 
                
                # Update the waypoint
                x_ref_attuale = waypoints[target_idx] 
                
                contatore_stallo = 0 
        # ==========================================

        # ==========================================
        # 3. TARGET SWITCHING LOGIC
        # ==========================================
        if not in_recovery:
            dist_al_target = np.linalg.norm(current_x[:2] - waypoints[target_idx][:2])

            if dist_al_target < TOLLERANZA_WAYPOINT:
                if target_idx < len(waypoints) - 1:
                    target_idx += 1
                    print(f"\n✅ Waypoint raggiunto! Passaggio al Target {target_idx + 1} a {waypoints[target_idx][:2]}")
                else:
                    print(f"\n🎯 MISSIONE COMPLETATA! Ultimo target raggiunto al passo {t}.")
                    break # End mission
        # ==========================================

    # ==========================================
    # FINAL PLOT OF TRAJECTORY AND OBSTACLES
    # ==========================================

    x_h = np.array(x_history)
    plt.figure(figsize=(15, 6))
    
    # ==========================================
    # DRAW OBSTACLES
    # ==========================================

    if len(ostacoli) > 0:
        # Determine whether we are using the new segment definition or the old rectangle one
        # If the first element is a list of lists (e.g. [[x1, z1], [x2, z2]]), it is a segment
        if isinstance(ostacoli[0][0], (list, np.ndarray)):
            # Draw generic oblique walls as thick black lines
            for seg in ostacoli:
                (x_A, z_A), (x_B, z_B) = seg
                plt.plot([x_A, x_B], [z_A, z_B], color='black', linewidth=4, zorder=2)
        else:
            # Draw the old parallel obstacles as filled gray rectangles
            for obs in ostacoli:
                w = obs[1] - obs[0]
                h = obs[3] - obs[2]
                plt.gca().add_patch(patches.Rectangle((obs[0], obs[2]), w, h, color='dimgray', alpha=0.7, zorder=2))
    # ==========================================
    
    # --- DRAW THE GREEN BOXES ---
    # Use a step (e.g. i[::5]) to avoid drawing a box at every single instant (it would become too dark green)
    # If you want to see ALL of them, remove the [::5] from the for loop.
    for i, box in enumerate(box_history[::5]):
        box_w = box[1] - box[0]
        box_h = box[3] - box[2]
        # Add the label only to the first box for the legend
        label = 'Safe-Box (AABB)' if i == 0 else ""
        plt.gca().add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, 
                                              edgecolor='lime', facecolor='none', 
                                              linewidth=1.0, alpha=0.3, label=label))
    
    # # Draw the trajectory
    # plt.plot(x_h[:, 0], x_h[:, 1], color='cyan', linewidth=2.5, label='Multi-Target Flight')
    
    # # Draw all the Waypoints
    # for i, wp in enumerate(waypoints):
    #     color = 'red' if i == target_idx else 'orange'
    #     plt.scatter(wp[0], wp[1], color=color, marker='X', s=150, zorder=6, label=f'WP {i+1}' if i==0 else "")

    # ==========================================
    # DRAW THE TRAJECTORY (Two-tone)
    # ==========================================
    for i in range(1, len(x_h)):
        # If the system was in recovery at that moment, use magenta (or orange), otherwise cyan
        colore_tratto = 'magenta' if mode_history[i] == 'recovery' else 'cyan'
        plt.plot(x_h[i-1:i+1, 0], x_h[i-1:i+1, 1], color=colore_tratto, linewidth=2.5)
        
    # Create two invisible lines just to make the legend clean
    plt.plot([], [], color='cyan', linewidth=2.5, label='Standard Navigation')
    plt.plot([], [], color='magenta', linewidth=2.5, label='Recovery Maneuver')

    # ==========================================
    # DRAW TARGETS (Current and Ghost)
    # ==========================================
    # 1. Draw old perturbed targets (faded and smaller)
    for g_wp in ghost_waypoints:
        plt.scatter(g_wp[0], g_wp[1], color='red', marker='X', s=80, alpha=0.25)
    
    if ghost_waypoints: # Add to the legend only if stalls occurred
        plt.scatter([], [], color='red', marker='X', s=80, alpha=0.25, label='Perturbed Targets (Stall)')

    # 2. Draw the actual current waypoints
    for i, wp in enumerate(waypoints):
        colore_wp = 'red' if i == target_idx else 'orange'
        testo_label = f'Target Finale' if i == target_idx else ""
        plt.scatter(wp[0], wp[1], color=colore_wp, marker='X', s=150, zorder=6, label=testo_label)
    
    plt.scatter(x0[0], x0[1], color='lime', s=100, label='Start', zorder=6)
    plt.title('Autonomous Navigation')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    
    # Show the legend outside the plot or in a corner
    #plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.show()

    

if __name__ == '__main__':
    main()