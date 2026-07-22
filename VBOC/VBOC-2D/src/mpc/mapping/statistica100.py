import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

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


from matplotlib.path import Path

# Aggiungi questa piccola funzione di supporto appena SOPRA genera_ambiente_2d_test()
def point_to_segment_dist(p, a, b):
    """Calcola la distanza minima tra un punto 'p' e un segmento definito dai punti 'a' e 'b'."""
    p, a, b = np.array(p), np.array(a), np.array(b)
    ab = b - a
    ap = p - a
    norm_ab_sq = np.dot(ab, ab)
    if norm_ab_sq == 0: 
        return np.linalg.norm(ap)
    t = max(0.0, min(1.0, np.dot(ap, ab) / norm_ab_sq))
    projection = a + t * ab
    return np.linalg.norm(p - projection)


def genera_ambiente_2d_test():
    """Nuova mappa basata sullo schizzo con ostacoli blu e target verdi."""
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
    
    # Final right wall
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            segments.append([poli[i], poli[(i + 1) % n]])
    segments.append([[21.0, -4.0], [21.0, -2.0]])
            
    # ==============================================================
    # NUOVA GENERAZIONE TARGET RANDOMICI (Monte Carlo)
    # ==============================================================
    num_targets = 100
    clearance = 0.4  # Distanza di sicurezza minima dagli ostacoli in metri
    targets = []
    
    # Usiamo Path di matplotlib per capire se un punto è DENTRO un ostacolo
    paths = [Path(poli) for poli in poligoni]

    np.random.seed(42)
    
    while len(targets) < num_targets:
        # 1. Campionamento nell'area utile (Escludo dietro lo start e oltre il muro finale)
        tx = np.random.uniform(1.0, 24.0)
        tz = np.random.uniform(-3.5, 4.5)
        pt = [tx, tz]
        
        # 2. Controllo: Il punto è DENTRO un ostacolo?
        is_inside = any(path.contains_point(pt) for path in paths)
        if is_inside:
            continue # Scarta e riprova
            
        # 3. Controllo: Il punto è TROPPO VICINO a un muro o spigolo?
        too_close = False
        for seg in segments:
            dist = point_to_segment_dist(pt, seg[0], seg[1])
            if dist < clearance:
                too_close = True
                break
                
        if too_close:
            continue # Scarta e riprova
            
        # Se passa tutti i test, è un punto valido!
        targets.append(np.array([tx, tz, 0.0, 0.0, 0.0, 0.0]))
    # ==============================================================
    
    return poligoni, segments, targets





def main_statistico():
    
    risultati = {"Successes": 0, "Timeout": 0, "Crashes": 0}
    traiettorie_riuscita = []
    all_box = []
    totale_recovery_attivate = 0
    crashes_per_feasibility = 0
    tutte_le_traiettorie = []
    registro_violazioni_globale = []

    params = Parameters("sth") 
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)

    DT = params.dt
    SIM_TIME = 40.0 # Extended time to cover all targets
    N_SIM = int(SIM_TIME / DT)
    
    poligoni, segmenti, targets = genera_ambiente_2d_test()
    N_TESTS = len(targets)


    # ==========================================================
    # PLOT INIZIALE: CONTROLLO DISTRIBUZIONE TARGET
    # ==========================================================
    print("Visualizzazione della distribuzione dei target...")
    print("CHIUDI LA FINESTRA DEL GRAFICO PER AVVIARE I TEST STATISTICI.")
    plt.figure(figsize=(12, 6))
    
    # Disegna gli ostacoli
    for poli in poligoni:
        plt.gca().add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
    
    # Disegna i 100 target
    for tb in targets:
        plt.scatter(tb[0], tb[1], color='red', marker='X', s=50, alpha=0.7)
    
    # Disegna lo start
    plt.scatter(0.0, 0.0, color='blue', s=120, label='Start (0,0)', zorder=5)
    
    plt.xlim(-2, 25)
    plt.ylim(-6, 6)
    plt.title(f'Distribuzione Monte Carlo: {N_TESTS} Target')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.show() # Attenzione: il codice si mette in pausa qui finché non chiudi la finestra!
    # ==========================================================


    print(f"--- AVVIO TEST DI COPERTURA: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):
        # Punto di partenza fisso
        current_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        x_history = [current_x.copy()]
        u_history = []
        box_history = []
        
        recovery_in_questo_test = 0
        esito = "Timeout" 

        # Variabili di stato di labirinto.py
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
        # SETUP VIDEO REAL-TIME
        # ==========================================
        plt.ion() # Enable real-time video mode
        fig_anim, ax_anim = plt.subplots(figsize=(12, 7))
        
        # Draw the fixed environment only once to avoid slowing down the video
        for poli in poligoni:
            ax_anim.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        ax_anim.scatter(target_base[0], target_base[1], color='red', marker='X', s=200, zorder=8)
        ax_anim.scatter(0.0, 0.0, color='blue', s=150, zorder=8)
        
        # Create the empty "actors" that we will move around the loop
        linea_traj, = ax_anim.plot([], [], color='black', linewidth=2, zorder=6)
        linea_pred, = ax_anim.plot([], [], color='orange', linewidth=2, zorder=5)
        punto_drone = ax_anim.scatter([], [], color='green', s=150, zorder=7)
        box_corrente = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=3, alpha=0.3, zorder=4)
        ax_anim.add_patch(box_corrente)
        
        ax_anim.set_xlim(-2, 25)
        ax_anim.set_ylim(-6, 6)
        ax_anim.grid(True, linestyle='--', alpha=0.5)
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

            # 1. LiDAR e Safe-Box
            hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=1.5)
            
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                
            # --- Direction calculation (final target) (Case 1) ---
            target_rel_x = x_ref_attuale[0] - current_x[0]
            target_rel_z = x_ref_attuale[1] - current_x[1]

            dx = target_rel_x
            dz = target_rel_z


            # CHOOSE BETWEEN DIFFERENT TARGETS FOR DIRECTION CALCULATION
            # !!! If you comment both you will use the final target (Case 1)!!!
            # !!! The selection is ininfluent for Cases (2,3,4), have W=0 !!!

            # # --- Direction calculation: use current velocity (Case 5) ---
            # # If velocity is relevant, follow velocity; if stopped, aim at the target.
            # dx = current_x[2] if abs(current_x[2]) > 0.1 else target_rel_x
            # dz = current_x[3] if abs(current_x[3]) > 0.1 else target_rel_z

            # --- Direction calculation: use predicted trajectory (Case 6) ---
            if x_sol_prev is not None and len(x_sol_prev) > 5:
                
                dx = x_sol_prev[5][0] - current_x[0]
                dz = x_sol_prev[5][1] - current_x[1]
                

                # If vectors are too small (e.g. perfect hovering), direct movement toward the target
                # if abs(dx) < 0.01 and abs(dz) < 0.01:#0.05
                #     dx, dz = target_rel_x, target_rel_z

            else:
                # Al primissimo passo (t=0)
                dx = target_rel_x
                dz = target_rel_z


            # WARM START SAFE BOX
            if box_abs_prev is not None:
                box_prev_rel = [
                    box_abs_prev[0] - current_x[0],
                    box_abs_prev[1] - current_x[0],
                    box_abs_prev[2] - current_x[1],
                    box_abs_prev[3] - current_x[1]
                ]
            else:
                box_prev_rel = None



            # CHOOSE BETWEEN WARM START METHOD (EXPANSION), DIRECTIONAL (EXPANSION) case 2-3-4 OR CLASSICAL METHOD case 1-5-6
            # # warm start
            # xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_warm_start(
            #     Q_rel, radii, dx, dz, drone_radius=0.1, box_prev=box_prev_rel, 
            #     expand_mode='directional',  # 'general', 'directional or 'score'
            #     W=50, rel=0.1
            # )

            # # directional
            # xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_directional(
            #      Q_rel, radii, dx, dz, drone_radius=0.1,
            # expand_mode='score',   # 'general', 'directional' or 'score'
            # W=50, rel=0.1)



##########################
# 
# resto con W=50 , rel=0.1 e senza elsif, provo best case con orizzonti diversi e raggio lidar = 1.5
#
# su 100 targets:
# con lidar 1.5
# case 6 NN con N=10: 
# case 6 NN con N=20: 
# case 6 NN con N=30: 

# con lidar 2.0
# case 6 NN con N=10: 
# case 6 NN con N=20: 
# case 6 NN con N=30:

# con lidar 3.0
# case 6 NN con N=10: 
# case 6 NN con N=20: 
# case 6 NN con N=30: 

# case 6 naive con N=10:
# case 6 naive con N=20: 
# case 6 naive con N=30: 

##############################################################################################################################################################################################


#*****************************************

# ######################################################################################################################################

            # classical
            xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_base(
                 Q_rel, radii, dx, dz, drone_radius=0.1, W=50, rel=0.1)
            
            
            box_abs = np.array([
                xMin_r + current_x[0], xMax_r + current_x[0], 
                zMin_r + current_x[1], zMax_r + current_x[1]
            ])
            box_history.append(box_abs.copy())
            


            # ==========================================
            # FORZATURA DEL BOX (Test)
            # Allarga artificialmente il safe-box per inglobare la traiettoria K
            # ==========================================
            # if x_sol_prev is not None:
            #     box_abs = force_trajectory_in_box(box_abs, x_sol_prev)
            # ==========================================



            # ==========================================
            # CHECK RECURSIVE FEASIBILITY
            # Check if the trajectory at step K (x_sol_prev) is inside the box just generated at step K+1 (box_abs)
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


            # ==========================================
            # DIAGNOSTIC BLOCK (MPC DEBUGGER)
            # ==========================================
            if t % 10 == 0: # Print every 10 steps
                print(f"\n--- DEBUG STEP {t} ---")
                print(f"1. Drone Position : X={current_x[0]:.2f}, Z={current_x[1]:.2f}")
                print(f"2. Green Box (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Z in [{box_abs[2]:.2f}, {box_abs[3]:.2f}]")
                print(f"3. Local Target  : X={x_ref_attuale[0]:.2f}, Z={x_ref_attuale[1]:.2f}")
                
                
                # Compute distance between drone and local target
                dist_to_local = np.linalg.norm(current_x[:2] - np.array([target_rel_x, target_rel_z])[:2])
                print(f"5. Distance to local target: {dist_to_local:.3f} metri")
            # ==========================================


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

                        # --- INIZIO CHECK AUTOPSIA CRASH ---
                        if x_sol_prev is not None:
                            punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3))
                            print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                            if punti_fuori > 0:
                                crashes_per_feasibility += 1
                        # --- FINE CHECK ---

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


            # # ==========================================
            # # VIDEO FRAME UPDATE
            # # ==========================================
            # # Aggiorna la linea della traiettoria
            # traj_attuale = np.array(x_history)
            # linea_traj.set_data(traj_attuale[:, 0], traj_attuale[:, 1])

            # if x_sol is not None:
            #     traj_pred_array = np.array(x_sol)
            #     linea_pred.set_data(traj_pred_array[:, 0], traj_pred_array[:, 1])
            
            # # Update the drone's position
            # punto_drone.set_offsets([[current_x[0], current_x[1]]])
            
            # # Update the coordinates and dimensions of the green safe-box
            # box_w = box_abs[1] - box_abs[0]
            # box_h = box_abs[3] - box_abs[2]
            # box_corrente.set_xy((box_abs[0], box_abs[2]))
            # box_corrente.set_width(box_w)
            # box_corrente.set_height(box_h)
            
            # # Change the color to red if the solver goes into Status 4 (Crash/Infeasibility)
            # if status in [3, 4]:
            #     box_corrente.set_edgecolor('red')
            #     box_corrente.set_facecolor('red')
            # else:
            #     box_corrente.set_edgecolor('lime')
            #     box_corrente.set_facecolor('lime')
            
            # ax_anim.set_title(f"Target {test_idx+1} | Step: {t} | Status MPC: {status}")
            
            # # Draw the frame on the screen
            # fig_anim.canvas.draw()
            # fig_anim.canvas.flush_events()
            
            # # SLOW MOTION EFFECT:
            # # 0.1 is fast. Set it to 0.5 to see it in slow motion.
            # plt.pause(0.1) 
            # # ==========================================


            # TWO WAYS TO MANAGE THE STALL WITH ARRIVAL AT THE RAISED OR THE ORIGINAL TARGET

            # # ==========================================
            # # STALL HANDLING
            # # ==========================================
            # if t > 0:
            #     spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            #     if spostamento < 0.01:
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
            # 1. ABSOLUTE SUCCESS CHECK (end test with Success)
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
                    
                    Z_MAX_SICURA = 4.8  # Margine di sicurezza sotto il soffitto
                    current_target[1] = min(current_target[1] + 0.20, Z_MAX_SICURA)
                    x_ref_attuale = current_target.copy()
                    contatore_stallo = 0

                # 3. LOCAL TARGET ARRIVAL CHECK
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

        # # ==========================================
        # # END OF VIDEO
        # # ==========================================
        # plt.ioff() # Disable realtime
        # # Leave the window open to analyze the last fatal/winning frame
        # plt.show() 
        # # ==========================================

    # ==========================================
    # STATISTICAL PLOT
    # ==========================================
    
    # # CORRECTION 3: individual plot for each target to analyze the green boxes
    # plt.figure(figsize=(10, 6))
    # for poli in poligoni:
    #     plt.gca().add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        
    # # Plot sampled green boxes (one every 8 steps to avoid clutter)
    # for box in box_history[::8]:
    #     box_w = box[1] - box[0]
    #     box_h = box[3] - box[2]
    #     plt.gca().add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, edgecolor='lime', facecolor='none', linewidth=0.8, alpha=0.4))
            
    # traj = np.array(x_history)
    # plt.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=2.5, label=f'Traiettoria Target {test_idx+1}')
    # plt.scatter(target_base[0], target_base[1], color='red', marker='X', s=150, zorder=5, label='Target')
    # plt.scatter(0.0, 0.0, color='blue', s=100, label='Start')
    # plt.xlim(-2, 25)
    # plt.ylim(-6, 6)
    # plt.title(f"Analisi di Volo - Target {test_idx+1} (Esito: {esito})")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend(loc='upper right', fontsize=10)
    # plt.show()

    



#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

#     labels = list(risultati.keys())
#     sizes = list(risultati.values())
#     colors = ['#4CAF50', '#FFC107', '#F44336']
#     explode = (0.1, 0, 0) 

#     ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
#     ax1.set_title('Tasso di Successo')

#     for poli in poligoni:
#         poly_patch = patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5)
#         ax2.add_patch(poly_patch)

#     for i, traj in enumerate(traiettorie_riuscita):
#         ax2.plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=f'Traj {i+1}')
    
#     ax2.scatter(target_base[0], target_base[1], color='red', marker='X', s=100, label='Target', zorder=5)
#     ax2.set_xlim(-1, 20)
#     ax2.set_ylim(-6, 6)
#     ax2.set_title('Top 10 Traiettorie di Successo')
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


    # print(f"\n--- RISULTATI FINALI ---")
    # print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")
    # print(f"Recovery totali innescate: {totale_recovery_attivate}")
    # print(f"Schianti dovuti a Feasibility Persa (traiettoria fuori dal box): {crashes_per_feasibility}/{risultati['Crashes']}")

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
            # 1. Disegna i box intermedi normali (uno ogni 10 o 5, escludendo l'ultimo)
            for box in bh[:-1:10]: 
                box_w = box[1] - box[0]
                box_h = box[3] - box[2]
                ax2.add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, 
                                                edgecolor='lime', facecolor='none', 
                                                linewidth=0.8, alpha=0.3))
            
            # 2. Disegna L'ULTIMISSIMO BOX in modo speciale (es. Magenta)
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
    #         z_ord = 4 # Lo teniamo più in alto per vederlo bene
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

