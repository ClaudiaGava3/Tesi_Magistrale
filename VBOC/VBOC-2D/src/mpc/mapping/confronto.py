import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from lidar import min_cube_select_base, get_lidar_hits_2d_qualsiasi, min_cube_select_directional, force_trajectory_in_box, min_cube_warm_start

# =====================================================================
# ... (Mantieni qui la tua funzione genera_ambiente_2d_test() intatta) ...
# =====================================================================
def genera_ambiente_2d_test():
    """Nuova mappa basata sullo schizzo con ostacoli blu e target verdi."""
    poligoni = [
        [[-2.0, -4.0], [25.0, -4.0], [25.0, -5.0], [-2.0, -5.0]], 
        [[-2.0,  5.0], [25.0,  5.0], [25.0,  6.0], [-2.0,  6.0]], 
        [[3.0, 1.0], [5.0, 3.0], [6.0, 1.0], [5.0, 0.0]],         
        [[7.0, -3.0], [9.0, -3.0], [9.0, -0.5], [7.0, -0.5]],     
        [[7.8, 3.9], [10.0, 3.9], [9.6, 0.8]],                     
        [[11.0, -0.4], [13.9, -0.6], [14.3, -1.7], [11.9, -2.8]], 
        [[12.0, 2.5], [12.6, 3.3], [13.6, 3.3], [14.1, 2.4], 
         [14.1, 1.2], [13.4, 0.9], [12.4, 1.0]],                  
        [[16.3, 4.1], [19.3, 4.2], [19.3, 2.5], [16.3, 2.5]],     
        [[15.0, -1.0], [18.0, -2.0], [19.0, 0.0], [16.0, 1.0]],   
    ]
    
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            segments.append([poli[i], poli[(i + 1) % n]])
    segments.append([[21.0, -4.0], [21.0, -2.0]])
            
    # Un solo target per l'analisi profonda
    targets = [
        np.array([11.2, 4.0, 0.0, 0.0, 0.0, 0.0])
        # np.array([11.0, 0.7, 0.0, 0.0, 0.0, 0.0])
        # np.array([7.6,  0.5, 0.0, 0.0, 0.0, 0.0])
        # np.array([10.2,-2.1, 0.0, 0.0, 0.0, 0.0])


    ]
    
    return poligoni, segments, targets

# ==========================================
# MOTORE DI SIMULAZIONE ISOLATO
# ==========================================
def esegui_simulazione(target_base, segmenti, params, metodo_box):
    """
    Esegue l'intera simulazione per un target specifico usando il metodo_box indicato.
    metodo_box può essere 'max' o 'espansione'.
    Restituisce la storia degli stati e dei box per il plot.
    """
    # 1. Inizializza un controller PULITO per evitare leak di memoria tra i test
    model = Model(params)
    controller = MpcController(model)
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)

    DT = params.dt
    SIM_TIME = 40.0
    N_SIM = int(SIM_TIME / DT)

    current_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_history = [current_x.copy()]
    u_history = []
    box_history = []
    
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

    for t in range(N_SIM):
        # --- 1. Recovery Timer ---
        if in_recovery:
            x_ref_attuale = target_recovery
            timer_recovery -= 1
            if timer_recovery <= 0:
                in_recovery = False
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
            if contatore_stallo == 0:
                x_ref_attuale = current_target.copy()

        # --- 2. LiDAR e Selezione Direzione ---
        hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=1.5)
        
        Q_rel = hits.copy()
        if len(hits) > 0:
            Q_rel[:, 0] -= current_x[0]
            Q_rel[:, 1] -= current_x[1]
            
        target_rel_x = x_ref_attuale[0] - current_x[0]
        target_rel_z = x_ref_attuale[1] - current_x[1]

        if x_sol_prev is not None and len(x_sol_prev) > 5:
            dx = x_sol_prev[5][0] - current_x[0]
            dz = x_sol_prev[5][1] - current_x[1]
            if abs(dx) < 0.05 and abs(dz) < 0.05:
                dx, dz = target_rel_x, target_rel_z
        else:
            dx = target_rel_x
            dz = target_rel_z

        # --- 3. CREAZIONE DEL BOX (BIVIO METODI) ---
        if metodo_box == 'max':
            xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_base(
                Q_rel, radii, dx, dz, drone_radius=0.1, W=50, rel=0.001
            )
        elif metodo_box == 'espansione':
            xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_directional(
                Q_rel, radii, dx, dz, drone_radius=0.1, expand_mode='score', W=50, rel=0.001
            )
        
        box_abs = np.array([
            xMin_r + current_x[0], xMax_r + current_x[0], 
            zMin_r + current_x[1], zMax_r + current_x[1]
        ])
        box_history.append(box_abs.copy())

        # --- 4. MPC Solve ---
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

        # --- 5. Gestione Infeasibility ---
        if status in [3, 4]:
            if alpha_curr is None: alpha_curr = 0.1
            
            # PLAN A: reset
            controller.ocp_solver.reset()
            controller.x_guess = np.tile(current_x, (controller.N, 1))
            controller.u_guess = np.full((controller.N, model.nu), u_hover)
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

            # PLAN B: storici
            if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                for i in range(len(u_history) - 1, -1, -1):
                    controller.ocp_solver.reset()
                    controller.x_guess = np.tile(current_x, (controller.N, 1))
                    controller.u_guess = u_history[i]
                    x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                    if status in [0, 2]: break

            # PLAN C: fallback centrale
            if (status in [3, 4]) and not in_recovery:
                passi_indietro = 10
                box_sicuro = box_history[-passi_indietro] if len(box_history) > passi_indietro else box_history[0]
                center_x = (box_sicuro[0] + box_sicuro[1]) / 2.0
                center_z = (box_sicuro[2] + box_sicuro[3]) / 2.0
                
                target_recovery = np.array([center_x, center_z, 0.0, 0.0, 0.0, 0.0])
                x_ref_attuale = target_recovery
                in_recovery = True
                timer_recovery = 40 
                
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
                    break
            
            elif status in [3, 4] and in_recovery:
                esito = "Crashes"
                break

        if u_sol is not None:
            u_history.append(u_sol.copy())

        current_x = x_sol[1]
        x_history.append(current_x.copy())
        x_sol_prev = x_sol.copy()

        # --- 6. Gestione Stallo e Successo ---
        dist_target_reale = np.linalg.norm(current_x[:2] - target_base[:2])
        if dist_target_reale < 0.3:
            esito = "Successes"
            break
                        
        if t > 0:
            spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            if spostamento < 0.01:
                contatore_stallo += 1
            else:
                contatore_stallo = 0 

            if contatore_stallo >= MAX_STALLO_ITER:
                Z_MAX_SICURA = 4.8
                current_target[1] = min(current_target[1] + 0.20, Z_MAX_SICURA)
                x_ref_attuale = current_target.copy()
                contatore_stallo = 0

            dist_al_locale = np.linalg.norm(current_x[:2] - current_target[:2])
            if dist_al_locale < 0.3 and not np.array_equal(current_target, target_base):
                current_target = target_base.copy()
                x_ref_attuale = current_target.copy()
                contatore_stallo = 0

    return np.array(x_history), box_history, esito


# ==========================================
# MAIN SCRIPT (CONFRONTO DOPPIO)
# ==========================================
def main_statistico():
    params = Parameters("sth") 
    params.act = 'gelu'
    params.build = True
    
    poligoni, segmenti, targets = genera_ambiente_2d_test()
    target_singolo = targets[0]  # Usiamo solo il primo target

    print("--- AVVIO TEST PARALLELO ---")
    
    # Eseguiamo i due test uno dopo l'altro
    print("1. Esecuzione Metodo MAX (min_cube_select_base)...")
    traj_max, box_hist_max, esito_max = esegui_simulazione(target_singolo, segmenti, params, metodo_box='max')
    
    print("2. Esecuzione Metodo ESPANSIONE (min_cube_select_directional)...")
    traj_exp, box_hist_exp, esito_exp = esegui_simulazione(target_singolo, segmenti, params, metodo_box='espansione')

    print(f"\n--- RISULTATI FINALI ---")
    print(f"Esito MAX: {esito_max}")
    print(f"Esito ESPANSIONE: {esito_exp}")

    # ==========================================
    # PLOT DI CONFRONTO
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle("Confronto Dinamico Closed-Loop: Max vs Espansione", fontsize=24)

    # Imposta la vista
    for ax in (ax1, ax2):
        for poli in poligoni:
            ax.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        ax.scatter(target_singolo[0], target_singolo[1], color='red', marker='X', s=150, zorder=5, label='Target')
        ax.scatter(0.0, 0.0, color='magenta', s=120, zorder=6, label='Start')
        ax.set_xlim(-2, 20)
        ax.set_ylim(-6, 6)
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- PLOT AX1 (CASO MAX - LIME) ---
    ax1.set_title(f"Metodo MAX (Esito: {esito_max})", fontsize=20)
    # Disegna 1 box ogni 10 step per non appesantire la vista
    for box in box_hist_max[::10]: 
        box_w = box[1] - box[0]
        box_h = box[3] - box[2]
        ax1.add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, edgecolor='lime', facecolor='none', linewidth=1.5, alpha=0.4))
    # Traiettoria
    ax1.plot(traj_max[:, 0], traj_max[:, 1], color='lime', linewidth=3.0, alpha=0.9, label='Traiettoria (Max)')
    # ax1.legend(loc='upper right')

    # --- PLOT AX2 (CASO ESPANSIONE - BLU) ---
    ax2.set_title(f"Metodo ESPANSIONE (Esito: {esito_exp})", fontsize=20)
    # Disegna 1 box ogni 10 step
    for box in box_hist_exp[::10]: 
        box_w = box[1] - box[0]
        box_h = box[3] - box[2]
        ax2.add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, edgecolor='blue', facecolor='none', linewidth=1.5, alpha=0.4))
    # Traiettoria
    ax2.plot(traj_exp[:, 0], traj_exp[:, 1], color='blue', linewidth=3.0, alpha=0.9, label='Traiettoria (Espansione)')
    # ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_statistico()