
# ==========================================================================================================================================================
"""Run a 13-target position-navigation campaign in a fixed polygonal map.

Each run starts from the same point, maps arbitrary wall segments with LiDAR,
builds a local safe box, and uses feedback MPC with stall handling and
infeasibility recovery. The campaign records navigation and recursive-box
feasibility outcomes across the predefined target set.

WARNING: set the state-cost weights ``Q_cost`` in ``mpc_abstract_obs.py`` for
position tracking before running this test. The state order is
``[x, z, theta, vx, vz, wy]``; increase the position weights for this case.
This script does not change the MPC weights automatically. Its local
``Q_COST`` is only used to report closed-loop cost. The tuning notes below
record historical runs and do not necessarily match the current settings. The
controller reports the NN margin but does not include it in the normal
terminal-clearance constraint; this test uses it when relaxing that constraint
during recovery.
"""
# ==========================================================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import matplotlib.animation as animation

from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from lidar import min_cube_select_base, get_lidar_hits_2d_qualsiasi, min_cube_select_directional, force_trajectory_in_box, min_cube_warm_start


# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for all other text
})


def genera_ambiente_2d_test():
    poligoni = [
        # Fixed bases (floor, ceiling, final wall)
        [[-2.0, -4.0], [25.0, -4.0], [25.0, -5.0], [-2.0, -5.0]], 
        [[-2.0,  5.0], [25.0,  5.0], [25.0,  6.0], [-2.0,  6.0]], 
        
        [[3.0, 1.0], [5.0, 3.0], [6.0, 1.0], [5.0, 0.0]],         # Left diamond
        [[7.0, -3.0], [9.0, -3.0], [9.0, -0.5], [7.0, -0.5]],     # Lower square
        
        [[7.8, 3.9], [10.0, 3.9], [9.6, 0.8]],                     # Upper left triangle
        [[11.0, -0.4], [13.9, -0.6], [14.3, -1.7], [11.9, -2.8]], # Lower slanted rectangle
        [[12.0, 2.5], [12.6, 3.3], [13.6, 3.3], [14.1, 2.4], 
         [14.1, 1.2], [13.4, 0.9], [12.4, 1.0]],                  # Central hexagon
        [[16.3, 4.1], [19.3, 4.2], [19.3, 2.5], [16.3, 2.5]],     # Upper right rectangle
        [[15.0, -1.0], [18.0, -2.0], [19.0, 0.0], [16.0, 1.0]],   # Lower right rectangle
    ]

    roof = 5.0
    
    # Final right wall
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            segments.append([poli[i], poli[(i + 1) % n]])
    segments.append([[21.0, -4.0], [21.0, -2.0]])

    # targets
    targets = [
        np.array([4.8, -2.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([6.2,  3.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([7.6,  0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([10.2,-2.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([11.0, 0.7, 0.0, 0.0, 0.0, 0.0]),
        np.array([11.2, 4.30, 0.0, 0.0, 0.0, 0.0]),
        np.array([15.1, 3.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([14.8, 0.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([15.5,-2.6, 0.0, 0.0, 0.0, 0.0]),
        np.array([18.2, 1.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([19.6,-2.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([20.1, 3.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([22.5, 0.4, 0.0, 0.0, 0.0, 0.0])
    ]
    
    return poligoni, segments, targets, roof





def main_statistico():
    
    risultati = {"Successes": 0, "Timeout": 0, "Crashes": 0}
    traiettorie_riuscita = []
    all_box = []
    totale_recovery_attivate = 0
    crashes_per_feasibility = 0
    tutte_le_traiettorie = []
    registro_violazioni_globale = []
    tempi_computazionali_globali = []

    params = Parameters("sth") 
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)

    DT = params.dt
    SIM_TIME = 40.0 # Extended time to cover all targets
    N_SIM = int(SIM_TIME / DT)
    
    poligoni, segmenti, targets, roof = genera_ambiente_2d_test()
    N_TESTS = len(targets)

    print(f"--- AVVIO TEST DI COPERTURA: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):
        # Fixed starting point
        current_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        x_history = [current_x.copy()]
        u_history = []
        box_history = []
        
        recovery_in_questo_test = 0
        esito = "Timeout" 

        contatore_stallo = 0
        MAX_STALLO_ITER = 50
        in_recovery = True

        timer_recovery = 0
        target_recovery = None
        current_target = target_base.copy()
        x_ref_attuale = current_target.copy()

        controller.ocp_solver.reset()
        controller.x_guess = np.tile(current_x, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)


        x_sol_prev = None
        box_abs_prev = None
        

        # ==========================================
        # SET UP REAL-TIME VIDEO DISPLAY AND FRAME CAPTURE.
        # ==========================================
        plt.ion() # Enable real-time video mode
        fig_anim, ax_anim = plt.subplots(figsize=(12, 7))
        
        # Draw the fixed environment only once
        for poli in poligoni:
            ax_anim.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        ax_anim.scatter(target_base[0], target_base[1], color='red', marker='X', s=200, zorder=8)
        ax_anim.scatter(0.0, 0.0, color='blue', s=150, zorder=8)
        
        linea_traj, = ax_anim.plot([], [], color='black', linewidth=2, zorder=6)
        linea_pred, = ax_anim.plot([], [], color='orange', linewidth=2, zorder=5)
        punto_drone = ax_anim.scatter([], [], color='green', s=150, zorder=7)
        box_corrente = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=3, alpha=0.3, zorder=4)
        ax_anim.add_patch(box_corrente)
        
        ax_anim.set_xlim(-2, 25)
        ax_anim.set_ylim(-6, 6)
        ax_anim.grid(True, linestyle='--', alpha=0.5)

        frames_animazione = []
        # ==========================================

        iterazioni_violazione_box = []


        for t in range(N_SIM):

            # 0. Recovery timer management
            if in_recovery:
                x_ref_attuale = target_recovery
                timer_recovery -= 1
                if timer_recovery <= 0:
                    in_recovery = False
                    
                    # Restore constraints
                    controller.ocp_solver.constraints_set(controller.N, "lh", np.zeros(4))
                    
                    lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                    ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")

                    lbx_e_curr[3:] = [-1.0, -1.0, -1.0]
                    ubx_e_curr[3:] = [ 1.0,  1.0,  1.0]

                    controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                    controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                    
                    current_target = target_base.copy() 
                    x_ref_attuale = current_target.copy()
            else:
                # Restore the base target if it was changed by a previous stall
                if contatore_stallo == 0:
                    x_ref_attuale = current_target.copy()

            t_start_step = time.perf_counter()

            # 1. LiDAR e Safe-Box
            hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=3.0)
            
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                
            # Direction signal from the current local position target (Case 1).
            target_rel_x = x_ref_attuale[0] - current_x[0]
            target_rel_z = x_ref_attuale[1] - current_x[1]

            dx = target_rel_x
            dz = target_rel_z


            # Choose the direction signal used by the safe-box selector.
            # If both alternatives remain commented, the local target is used.
            # !!! The selection is ininfluent for Cases (2,3,4), have W=0 !!!

            # # --- Direction calculation: use current velocity (Case 5) ---
            # # Alternative: use current velocity when significant; otherwise
            # # fall back to the local target direction.
            # dx = current_x[3] if abs(current_x[3]) > 0.1 else target_rel_x
            # dz = current_x[4] if abs(current_x[4]) > 0.1 else target_rel_z

            # --- Direction calculation: use predicted trajectory (Case 6) ---
            if x_sol_prev is not None and len(x_sol_prev) > 5:
                
                dx = x_sol_prev[5][0] - current_x[0]
                dz = x_sol_prev[5][1] - current_x[1]
                

                # # If vectors are too small (e.g. perfect hovering), direct movement toward the target
                # if abs(dx) < 0.05 and abs(dz) < 0.05:#0.01
                #     dx, dz = target_rel_x, target_rel_z

            else:
                # At first step (t=0)
                dx = target_rel_x
                dz = target_rel_z


        

            # # WARM START SAFE BOX (OLD VERSION)
            # if box_abs_prev is not None:
            #     box_prev_rel = [
            #         box_abs_prev[0] - current_x[0],
            #         box_abs_prev[1] - current_x[0],
            #         box_abs_prev[2] - current_x[1],
            #         box_abs_prev[3] - current_x[1]
            #     ]
            # else:
            #     box_prev_rel = None

            # ==========================================
            # WARM START SAFE BOX (MODIFIED)
            # ==========================================
            if x_sol_prev is not None:
                # 1. Calculate the bounding box (min and max) of the ENTIRE predicted trajectory at the previous step
                traj_xmin = np.min(x_sol_prev[:, 0])
                traj_xmax = np.max(x_sol_prev[:, 0])
                traj_zmin = np.min(x_sol_prev[:, 1])
                traj_zmax = np.max(x_sol_prev[:, 1])
                
                # 2. Convert this Bounding Box to relative coordinates with respect to the current position of the drone
                margin = 0.15
                box_prev_rel = [
                    traj_xmin - current_x[0]-margin,
                    traj_xmax - current_x[0]+margin,
                    traj_zmin - current_x[1]-margin,
                    traj_zmax - current_x[1]+margin
                ]
            else:
                box_prev_rel = None
            # ==========================================



            # CHOOSE BETWEEN WARM START METHOD (EXPANSION), DIRECTIONAL (EXPANSION) case 2-3-4 OR CLASSICAL METHOD case 1-5-6
            # warm start
            xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_warm_start(
                Q_rel, radii, dx, dz, target_rel_x, target_rel_z, drone_radius=0.1, box_prev=box_prev_rel, 
                expand_mode='directional',  # 'general', 'directional or 'score'
                W=50, rel=0.1
            )

            # # directional
            # xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_directional(
            #      Q_rel, radii, dx, dz, drone_radius=0.1,
            # expand_mode='score',   # 'general', 'directional' or 'score'
            # W=50, rel=0.1)


            # # classical
            # xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_base(
            #      Q_rel, radii, dx, dz, drone_radius=0.1, W=50, rel=0.1)
            
            
            box_abs = np.array([
                xMin_r + current_x[0], xMax_r + current_x[0], 
                zMin_r + current_x[1], zMax_r + current_x[1]
            ])
            box_history.append(box_abs.copy())
            


            # ==========================================
            # FORZATURA DEL BOX (Test)
            # Optional experiment: expand the mapped safe box to contain the
            # previous MPC-predicted trajectory.
            # ==========================================
            # if x_sol_prev is not None:
            #     box_abs = force_trajectory_in_box(box_abs, x_sol_prev)
            # ==========================================



            # ==========================================
            # CHECK RECURSIVE FEASIBILITY
            # Check whether the previous predicted trajectory lies inside the
            # safe box mapped at the current step.
            # ==========================================
            if x_sol_prev is not None:
                is_outside = False
                for p in x_sol_prev:
                    if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or 
                        p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3):
                        is_outside = True
                        break
                
                if is_outside:
                    iterazioni_violazione_box.append(t)
            # ==========================================


            # 2. MPC Solve
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

            t_end_step = time.perf_counter()

            if status in [0, 2]:
                tempi_computazionali_globali.append(t_end_step - t_start_step)


            # # ==========================================
            # # DIAGNOSTIC BLOCK (MPC DEBUGGER)
            # # ==========================================
            # if t % 10 == 0: # Print every 10 steps
            #     print(f"\n--- DEBUG STEP {t} ---")
            #     print(f"1. Drone Position : X={current_x[0]:.2f}, Z={current_x[1]:.2f}")
            #     print(f"2. Green Box (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Z in [{box_abs[2]:.2f}, {box_abs[3]:.2f}]")
            #     print(f"3. Local Target  : X={x_ref_attuale[0]:.2f}, Z={x_ref_attuale[1]:.2f}")
                
                
            #     # Compute distance between drone and local target
            #     dist_to_local = np.linalg.norm(current_x[:2] - np.array([target_rel_x, target_rel_z])[:2])
            #     print(f"5. Distance to local target: {dist_to_local:.3f} m")
            # # ==========================================


            # ==========================================
            # INFEASIBILITY HANDLING
            # ==========================================
            if status in [3, 4]:
                recovery_in_questo_test += 1
                totale_recovery_attivate += 1
                if alpha_curr is None: alpha_curr = 0.1
                
                # PLAN A: reset memory
                controller.ocp_solver.reset()
                controller.x_guess = np.tile(current_x, (controller.N, 1))
                controller.u_guess = np.full((controller.N, model.nu), u_hover)
                x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

                # PLAN B: historical warm start
                if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                    for i in range(len(u_history) - 1, -1, -1):
                        past_u = u_history[i]
                        controller.ocp_solver.reset()
                        controller.x_guess = np.tile(current_x, (controller.N, 1))
                        controller.u_guess = past_u
                        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                        if status in [0, 2]:
                            break

                # PLAN C: retreat to center
                if (status in [3, 4]) and not in_recovery:
                    print("Piano C avviato")
                    passi_indietro = 10
                    if len(box_history) > passi_indietro:
                        box_sicuro = box_history[-passi_indietro]
                    else:
                        box_sicuro = box_history[0]
                    
                    center_x = (box_sicuro[0] + box_sicuro[1]) / 2.0
                    center_z = (box_sicuro[2] + box_sicuro[3]) / 2.0
                    
                    target_recovery = np.array([center_x, center_z, 0.0, 0.0, 0.0, 0.0])
                    x_ref_attuale = target_recovery
                    
                    in_recovery = True
                    timer_recovery = 40 
                    
                    # Relax constraints
                    controller.ocp_solver.constraints_set(controller.N, "lh", np.full(4, -alpha_curr))
                   
                    lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                    ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                    lbx_e_curr[3:] = [-5.0, -5.0, -5.0]
                    ubx_e_curr[3:] = [ 5.0,  5.0,  5.0]
                    controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                    controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                    
                    x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                    
                    if status in [3, 4]:
                        esito = "Crashes"
                        in_recovery = False

                        # --- START CRASH AUTOPSY CHECK ---
                        if x_sol_prev is not None:
                            punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3))
                            print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                            if punti_fuori > 0:
                                crashes_per_feasibility += 1
                        # --- END CHECK ---

                        break
                
                elif status in [3, 4] and in_recovery:
                    esito = "Crashes"
                    in_recovery = False

                    # --- CHECK POINTS OF PRED_TRAJ OUTSIDE ON THE SAFE BOX ---
                    if x_sol_prev is not None:
                        punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3))
                        print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                        if punti_fuori > 0:
                                crashes_per_feasibility += 1
                    break

            if u_sol is not None:
                u_history.append(u_sol.copy())

            current_x = x_sol[1]
            x_history.append(current_x.copy())

            x_sol_prev = x_sol.copy()
            box_abs_prev = box_abs.copy()


            # ==========================================
            # VIDEO FRAME UPDATE
            # ==========================================
            # Update the trajectory line
            traj_attuale = np.array(x_history)
            linea_traj.set_data(traj_attuale[:, 0], traj_attuale[:, 1])

            if x_sol is not None:
                traj_pred_array = np.array(x_sol)
                linea_pred.set_data(traj_pred_array[:, 0], traj_pred_array[:, 1])
            
            # Update the drone's position
            punto_drone.set_offsets([[current_x[0], current_x[1]]])
            
            # Update the coordinates and dimensions of the green safe-box
            box_w = box_abs[1] - box_abs[0]
            box_h = box_abs[3] - box_abs[2]
            box_corrente.set_xy((box_abs[0], box_abs[2]))
            box_corrente.set_width(box_w)
            box_corrente.set_height(box_h)
            
            # Change the color to red if the solver goes into Status 4 (Crash/Infeasibility)
            if status in [3, 4]:
                box_corrente.set_edgecolor('red')
                box_corrente.set_facecolor('red')
            else:
                box_corrente.set_edgecolor('lime')
                box_corrente.set_facecolor('lime')
            
            ax_anim.set_title(f"Target {test_idx +1} | Step: {t} | Status MPC: {status}")
            
            # Draw the frame on the screen
            fig_anim.canvas.draw()
            fig_anim.canvas.flush_events()
            
            plt.pause(0.1) 

            image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
            frames_animazione.append(image)
            # ==========================================
            # ==========================================


            # Alternative stall policy: raise the local target, then return to
            # the original target after reaching the raised waypoint.

            # # ==========================================
            # # STALL HANDLING
            # # ==========================================
            # if t > 0:
            #     displacement = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            #     if displacement < 0.01:
            #         contatore_stallo += 1
            #     else:
            #         contatore_stallo = 0 

            #     if contatore_stallo >= MAX_STALLO_ITER:
            #         print(f"\n⚠️ STALL DETECTED (Step {t})! The drone is trapped in a local minimum.")
            #         print("   -> Perturb the local target upward")
                    
            #         current_target[1] += 0.20 
            #         x_ref_attuale = current_target.copy()
            #         contatore_stallo = 0 
            

            # ==========================================
            # STALL HANDLING AND RETURN TO BASE TARGET
            # ==========================================
            # 1. End the trial when the base target is reached.
            dist_target_reale = np.linalg.norm(current_x[:2] - target_base[:2])
            if dist_target_reale < 0.3:
                esito = "Successes"
                break
                            
            if t > 0:
                # 2. NORMAL STALL HANDLING
                spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
                if spostamento < 0.01:
                    contatore_stallo += 1
                else:
                    contatore_stallo = 0 

                if contatore_stallo >= MAX_STALLO_ITER:
                    print(f"\n⚠️ STALL DETECTED (Step {t})! Perturb the target upward.")
                    
                    Z_MAX_SICURA = roof - 0.2  # Keep the waypoint 0.2 m below the ceiling.
                    current_target[1] = min(current_target[1] + 0.20, Z_MAX_SICURA)
                    x_ref_attuale = current_target.copy()
                    contatore_stallo = 0

                # 3. Return to the base target after reaching the raised waypoint.
                dist_al_locale = np.linalg.norm(current_x[:2] - current_target[:2])
                if dist_al_locale < 0.3:
                    if not np.array_equal(current_target, target_base):
                        print(f"\n✅ Elevated target reached. Descend toward the final ground target.")
                        current_target = target_base.copy()
                        x_ref_attuale = current_target.copy()
                        contatore_stallo = 0


        risultati[esito] += 1
        if esito == "Successes" :
            traiettorie_riuscita.append(np.array(x_history))


        all_box.append(box_history)
        tutte_le_traiettorie.append((np.array(x_history), esito))



        print(f"Test {test_idx+1}/{N_TESTS} -> Esito: {esito} (Recovery usate: {recovery_in_questo_test})")

        print(f"\n--- FINAL RESULTS ---")
        print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")


        # ==========================================
        # FINAL REPORT OF THE SINGLE TARGET
        # ==========================================
        print(f"\n[{'='*40}]")
        print(f"🎯 REPORT TARGET {test_idx+1}/{N_TESTS}")
        print(f"Esito finale: {esito} (Iterazione di stop: {t})")
        print(f"Recovery attivate: {recovery_in_questo_test}")
        
        if len(iterazioni_violazione_box) > 0:
            print(f"⚠️ PROBLEMA: La traiettoria k è uscita dal box k+1 per {len(iterazioni_violazione_box)} volte.")
            print(f"Iterazioni esatte in cui è successo: {iterazioni_violazione_box}")
            
            # Correlation Analysis:
            if esito == "Crashes":
                if iterazioni_violazione_box[-1] >= t - 5:
                    print("--> 🔴 FORTE CORRELAZIONE: L'ultima uscita dal box è avvenuta a ridosso dello schianto!")
                else:
                    print("--> 🟠 CORRELAZIONE DEBOLE: Il drone è uscito dal box in passato, ma si è schiantato molto dopo.")
        else:
            print("✅ OTTIMO: La traiettoria predetta è rimasta SEMPRE all'interno del box al passo k+1.")
        print(f"[{'='*40}]\n")


        # --- Rescue in global register ---
        tipo_correlazione = "Nessuna"
        if len(iterazioni_violazione_box) > 0:
            if esito == "Crashes":
                if iterazioni_violazione_box[-1] >= t - 5:
                    tipo_correlazione = "FORTE (a ridosso dello schianto)"
                else:
                    tipo_correlazione = "DEBOLE (molto prima dello schianto)"
            else:
                tipo_correlazione = f"ININFLUENTE (Test finito in {esito})"
                
        registro_violazioni_globale.append({
            "target": test_idx + 1,
            "esito": esito,
            "iterazione": t,
            "num_violazioni": len(iterazioni_violazione_box),
            "correlazione": tipo_correlazione
        })
        # ----------------------------------------


        # ==========================================
        # END OF VIDEO
        # ==========================================
        plt.ioff()
        plt.close(fig_anim)

        if len(frames_animazione) > 0:
            cartella_video = "video_traiettorie_tempi"
            os.makedirs(cartella_video, exist_ok=True)
            
            print(f"Generazione del video per il Target {test_idx+1} in corso...")
            fig_movie = plt.figure(figsize=(12, 7))
            ax_movie = fig_movie.add_subplot(111)
            ax_movie.axis('off')
            
            # Show the first frame
            im = ax_movie.imshow(frames_animazione[0])
            
            def update_frame(i):
                im.set_data(frames_animazione[i])
                return [im]
            
            # Create the animation from saved images
            ani = animation.FuncAnimation(fig_movie, update_frame, frames=len(frames_animazione), blit=True)
            
            nome_video = os.path.join(cartella_video, f"Video_Target_N10_{test_idx+1:02d}_{esito}.mp4")
            
            # Save
            try:
                ani.save(nome_video, writer='ffmpeg', fps=10)
                print(f"🎬 Video salvato con successo: {nome_video}")
            except Exception as e:
                print("Nota: 'ffmpeg' non trovato, provo a salvare in formato .gif...")
                nome_video_gif = nome_video.replace(".mp4", ".gif")
                ani.save(nome_video_gif, writer='pillow', fps=10)
                print(f"🎬 Video (GIF) salvato con successo: {nome_video_gif}")
                
            plt.close(fig_movie)
        # ==========================================
        # ==========================================

    # ==========================================
    # STATISTICAL PLOT
    # ==========================================
    
    # # Optional per-target plot for inspecting the mapped safe boxes.
    # plt.figure(figsize=(10, 6))
    # for poli in poligoni:
    #     plt.gca().add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        
    # # Plot sampled green boxes (one every 8 steps to avoid clutter)
    # for box in box_history[::8]:
    #     box_w = box[1] - box[0]
    #     box_h = box[3] - box[2]
    #     plt.gca().add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, edgecolor='lime', facecolor='none', linewidth=0.8, alpha=0.4))
            
    # traj = np.array(x_history)
    # plt.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=2.5, label=f'Trajectory for target {test_idx+1}')
    # plt.scatter(target_base[0], target_base[1], color='red', marker='X', s=150, zorder=5, label='Target')
    # plt.scatter(0.0, 0.0, color='blue', s=100, label='Start')
    # plt.xlim(-2, 25)
    # plt.ylim(-6, 6)
    # plt.title(f"Flight analysis - target {test_idx+1} (outcome: {esito})")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend(loc='upper right', fontsize=10)
    # plt.show()

    



#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

#     labels = list(risultati.keys())
#     sizes = list(risultati.values())
#     colors = ['#4CAF50', '#FFC107', '#F44336']
#     explode = (0.1, 0, 0) 

#     ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
#     ax1.set_title('Success rate')

#     for poli in poligoni:
#         poly_patch = patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5)
#         ax2.add_patch(poly_patch)

#     for i, traj in enumerate(traiettorie_riuscita):
#         ax2.plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=f'Traj {i+1}')
    
#     ax2.scatter(target_base[0], target_base[1], color='red', marker='X', s=100, label='Target', zorder=5)
#     ax2.set_xlim(-1, 20)
#     ax2.set_ylim(-6, 6)
#     ax2.set_title('Top 10 successful trajectories')
#     ax2.legend(loc='upper right', fontsize=8)
#     ax2.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.show()




# ==========================================
# FINAL STATISTICAL PLOT (REACHABILITY MAP)
# ==========================================
    print(f"\n--- RISULTATI FINALI ---")
    print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")
    
    # --- GLOBAL FEASIBILITY SCOREBOARD PRINT ---
    print("\n" + "="*70)
    print("📊 RESOCONTO GLOBALE RECURSIVE FEASIBILITY (Traiettoria k in Box k+1)")
    print("="*70)
    for row in registro_violazioni_globale:
        print(f"Target {row['target']:02d} | Esito: {row['esito']:<9} | Step Stop: {row['iterazione']:<4} | "
              f"Uscite dal box: {row['num_violazioni']:<3} | Correlazione: {row['correlazione']}")
    print("="*70 + "\n")

    tempi_array = np.array(tempi_computazionali_globali)
    if len(tempi_array) > 0:
        print("\n" + "="*70)
        print(f"⏱  STATISTICHE TEMPI COMPUTAZIONALI (N={controller.N}, LiDAR=3.0m)")
        print("="*70)
        print(f"Tempo Medio:     {np.mean(tempi_array):.5f} secondi")
        print(f"99-Percentile:   {np.percentile(tempi_array, 99):.5f} secondi")
        print(f"Tempo Massimo:   {np.max(tempi_array):.5f} secondi")
        print("="*70 + "\n")


    # print(f"\n--- RISULTATI FINALI ---")
    # print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")
    # print(f"Total recovery attempts: {totale_recovery_attivate}")
    # print(f"Crashes associated with lost feasibility (trajectory outside box): {crashes_per_feasibility}/{risultati['Crashes']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

    # 1. PIE CHART
    labels = list(risultati.keys())
    sizes = list(risultati.values())
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0) 

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax1.set_title('Success Rate')

    # 2. REACHABILITY MAP (coverage map)
    for poli in poligoni:
        ax2.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))

    # box plots
    for bh in all_box:
        if len(bh) > 0:
            # 1. Draw intermediate boxes (one every 10 or 5, excluding the last)
            for box in bh[:-1:10]: 
                box_w = box[1] - box[0]
                box_h = box[3] - box[2]
                ax2.add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, 
                                                edgecolor='lime', facecolor='none', 
                                                linewidth=0.8, alpha=0.3))
            
            # 2. Draw the LAST BOX in a special way (e.g., Magenta)
            ultimo_box = bh[-1]
            box_w = ultimo_box[1] - ultimo_box[0]
            box_h = ultimo_box[3] - ultimo_box[2]
            ax2.add_patch(patches.Rectangle((ultimo_box[0], ultimo_box[2]), box_w, box_h, 
                                            edgecolor='magenta', facecolor='magenta', 
                                            linewidth=2.5, alpha=0.4, zorder=4))

    for i, traj in enumerate(traiettorie_riuscita):
        ax2.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=1.5, alpha=0.7)

    #trajectories plots
    # for traj, esito_traj in tutte_le_traiettorie:
    #     if esito_traj == "Successes":
    #         colore = 'cyan'
    #         z_ord = 3
    #     elif esito_traj == "Crashes":
    #         colore = 'red'
    #         z_ord = 4  # Keep crash trajectories above other plot elements.
    #     else: # Timeout
    #         colore = 'orange'
    #         z_ord = 3
            
    #     ax2.plot(traj[:, 0], traj[:, 1], color=colore, linewidth=1.5, alpha=0.7, zorder=z_ord)
    
    for tb in targets:
        ax2.scatter(tb[0], tb[1], color='red', marker='X', s=100, zorder=5)
        
    ax2.scatter(0.0, 0.0, color='blue', s=120, zorder=6, label='Start')
    
    ax2.plot([], [], color='cyan', linewidth=1.5, label='Traiettorie di Volo')
    ax2.scatter([], [], color='red', marker='X', s=100, label='Target Testati')

    ax2.set_xlim(-2, 25)
    ax2.set_ylim(-6, 6)
    ax2.set_title('Reachability Map')
    #ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_statistico()

