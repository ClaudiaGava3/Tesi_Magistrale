import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import matplotlib.animation as animation

from parser import Parameters
from src.MPC.mapping.mpc_abstract_obs import Model
from src.MPC.mapping.mpc_controller_obs import MpcController
from src.MPC.mapping.lidar import min_cube_select_base, get_lidar_hits_2d_qualsiasi, min_cube_select_directional, force_trajectory_in_box, min_cube_warm_start


# PER ATTIVARE IL TARGET IN VELOCITA' CAMBIARE I PESI NEL FILE ABSTRACT PER LA FUNZIONE DI COSTO


# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for all other text
})

def ruota_e_trasla(vertici_locali, cx, cz, angolo_deg):
    theta = np.radians(angolo_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    poligono_globale = []
    for x, z in vertici_locali:
        x_rot = x * cos_t - z * sin_t
        z_rot = x * sin_t + z * cos_t
        poligono_globale.append([x_rot + cx, z_rot + cz])
        
    return poligono_globale

import random

def genera_ambienti_random(num_tests=10):
    ambienti = []
    targets = []
    roof = 11.0

    # Forme base
    v_triangolo  = [[-2.0, -1.5], [2.0, -1.5], [0.0, 1.5]]                  
    v_rettangolo = [[-2.0, -0.75], [2.0, -0.75], [2.0, 0.75], [-2.0, 0.75]] 
    v_quadrato   = [[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]]     
    v_trapezio   = [[-2.0, -1.0], [2.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] 
    forme = [v_triangolo, v_rettangolo, v_quadrato, v_trapezio]

    for _ in range(num_tests):
        poligoni = []
        # Pavimento e Soffitto fissi
        poligoni.append([[-2.0, 0.0], [52.0, 0.0], [52.0, -1.0], [-2.0, -1.0]]) 
        poligoni.append([[-2.0, 11.0], [52.0, 11.0], [52.0, 12.0], [-2.0, 12.0]]) 

        num_ostacoli = 7
        x_start = 8.0
        x_end = 48.0
        step_x = (x_end - x_start) / num_ostacoli # Circa 5.7 metri a settore

        # Decidiamo a caso se lo slalom parte dall'alto o dal basso
        inizia_alto = random.choice([True, False])

        for i in range(num_ostacoli):
            forma = random.choice(forme)
            
            # 1. Piazzo l'ostacolo nel suo specifico settore (con un po' di margine casuale)
            cx = np.random.uniform(x_start + i * step_x + 0.5, x_start + (i + 1) * step_x - 0.5)
            
            # 2. Alternanza garantita dello slalom
            is_high = (i % 2 == 0) if inizia_alto else (i % 2 != 0)
            
            if is_high:
                angolo = random.choice([180, 135, 225])
            else:
                angolo = random.choice([0, 45, -45, 20, -20])

            # 3. Calcolo dell'ingombro ruotato per non bucare i bordi
            forma_ruotata_zero = ruota_e_trasla(forma, 0.0, 0.0, angolo)
            z_min_locale = min(v[1] for v in forma_ruotata_zero)
            z_max_locale = max(v[1] for v in forma_ruotata_zero)
            
            # 4. Assegnazione dell'altezza sicura
            margin = 0.0
            if is_high:
                limite_sup = 11.0 - margin - z_max_locale
                limite_inf = max(7.0, 0.0 + margin - z_min_locale) 
                cz = np.random.uniform(limite_inf, limite_sup) if limite_sup > limite_inf else limite_sup
            else:
                limite_inf = 0.0 + margin - z_min_locale
                limite_sup = min(4.0, 11.0 - margin - z_max_locale)
                cz = np.random.uniform(limite_inf, limite_sup) if limite_sup > limite_inf else limite_inf

            poligoni.append(ruota_e_trasla(forma, cx=cx, cz=cz, angolo_deg=angolo))

        segments = []
        for poli in poligoni:
            n = len(poli)
            for i in range(n):
                segments.append([poli[i], poli[(i + 1) % n]])
        
        # Muro fittizio finale per il Lidar
        segments.append([[55.0, 0.0], [55.0, 11.0]])

        ambienti.append((poligoni, segments))

        # Target Random di Velocità
        vx_rand = round(np.random.uniform(0.8, 1.2), 2)
        targets.append(np.array([0.0, 0.0, 0.0, vx_rand, 0.0, 0.0]))

    return ambienti, targets, roof


def dist_punto_segmento(px, pz, p1, p2):
    """Calcola la distanza minima tra un punto e un segmento."""
    x1, z1 = p1
    x2, z2 = p2
    l2 = (x2 - x1)**2 + (z2 - z1)**2
    if l2 == 0: 
        return np.hypot(px - x1, pz - z1)
    
    # Proiezione ortogonale del punto sul segmento (parametro t tra 0 e 1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (pz - z1) * (z2 - z1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_z = z1 + t * (z2 - z1)
    return np.hypot(px - proj_x, pz - proj_z)

def aggiungi_ostacoli_densi(ambienti, num_rombi=10):
    """
    Aggiunge ostacoli a forma di rombo all'ambiente.
    Garantisce che non si sovrappongano ai muri e che siano ben distanziati tra loro.
    """
    # Rombo più grande (raggio 0.6, larghezza/altezza totale 1.2 metri)
    v_rombo = [[0.0, 0.6], [0.6, 0.0], [0.0, -0.6], [-0.6, 0.0]] 
    nuovi_ambienti = []
    
    for poligoni, segmenti in ambienti:
        nuovi_poligoni = list(poligoni)
        nuovi_segmenti = list(segmenti)
        
        tentativi = 0
        rombi_aggiunti = 0
        centri_piazzati = [] # Memoria dei rombi già piazzati per distanziarli
        
        # Aumentiamo i tentativi massimi perché le condizioni ora sono più stringenti
        while rombi_aggiunti < num_rombi and tentativi < 500:
            # Genera centro casuale
            cx = np.random.uniform(7.0, 47.0)
            cz = np.random.uniform(2.0, 9.0)
            
            # 1. CHECK DISTANZA TRA ROMBI (Evita che siano "ammucchiati" o attaccati)
            troppo_vicini_tra_loro = False
            for (px, pz) in centri_piazzati:
                if np.hypot(cx - px, cz - pz) < 3.5:  # Distanza minima tra due rombi
                    troppo_vicini_tra_loro = True
                    break
            
            if troppo_vicini_tra_loro:
                tentativi += 1
                continue
                
            # 2. CHECK DISTANZA DAGLI OSTACOLI ORIGINALI
            troppo_vicino_ai_muri = False
            for seg in nuovi_segmenti:
                if seg[0][0] > 54.0: # Ignora il traguardo
                    continue
                
                dist = dist_punto_segmento(cx, cz, seg[0], seg[1])
                # 0.6m (raggio rombo) + 0.1m (drone) + 0.5m (margine di volo) = 1.2m
                if dist < 1.2: 
                    troppo_vicino_ai_muri = True
                    break
            
            # 3. PIAZZAMENTO
            if not troppo_vicino_ai_muri:
                rombo_traslato = [[x+cx, z+cz] for x, z in v_rombo]
                nuovi_poligoni.append(rombo_traslato)
                
                n = len(rombo_traslato)
                for j in range(n):
                    nuovi_segmenti.append([rombo_traslato[j], rombo_traslato[(j+1)%n]])
                    
                centri_piazzati.append((cx, cz)) # Salva il centro per i prossimi check
                rombi_aggiunti += 1
            
            tentativi += 1
            
        nuovi_ambienti.append((nuovi_poligoni, nuovi_segmenti))
        
    return nuovi_ambienti

def anteprima_ambiente(num_rombi=10):
    """
    Genera un singolo ambiente, aggiunge gli ostacoli densi e mostra un plot statico.
    Utile per calibrare visivamente la grandezza e il numero dei rombi.
    """
    # 1. Genera un solo ambiente base
    ambienti_base, targets, roof = genera_ambienti_random(1)
    
    # 2. Aggiunge i rombi
    ambienti_densi = aggiungi_ostacoli_densi(ambienti_base, num_rombi=num_rombi)
    
    # 3. Estrae i poligoni per il disegno
    poligoni, segmenti = ambienti_densi[0] 

    # 4. Configurazione del Plot
    fig, ax = plt.subplots(figsize=(14, 4))

    # Disegna tutti gli ostacoli (inclusi i nuovi rombi)
    for poli in poligoni:
        ax.add_patch(patches.Polygon(poli, closed=True, facecolor='silver', edgecolor='black', alpha=0.8, linewidth=1.2))
        
    # Elementi fissi (Traguardo e Start)
    # ax.plot([50.0, 50.0], [0.0, 11.0], color='red', linestyle='--', linewidth=2, label="Traguardo")
    # ax.scatter(1.0, 5.0, color='blue', s=150, label="Start Drone", zorder=5)

    # Dettagli estetici del grafico
    ax.set_xlim(-2, 52)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal') # Mantiene le proporzioni reali (non distorce i rombi)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Test Environment A")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    # ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

def main_statistico(N_TESTS):

    np.random.seed(44)
    random.seed(44)
    
    risultati = {"Successes": 0, "Timeout": 0, "Crashes": 0}
    traiettorie_riuscita = []
    all_box = []
    totale_recovery_attivate = 0
    crashes_per_feasibility = 0
    tutte_le_traiettorie = []
    registro_violazioni_globale = []

    # Pesi della funzione di costo (per la valutazione Closed-Loop)
    Q_COST = np.diag([0.0001, 0.0001, 20.0, 100.0, 1.0, 1.0])
    R_COST = np.diag([0.0001, 0.0001])

    params = Parameters("sth") 
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)

    DT = params.dt
    SIM_TIME = 50.0 # Extended time to cover all targets
    N_SIM = int(SIM_TIME / DT)
    
    
    # Generiamo 10 stanze uniche e 10 target random
    ambienti, targets, roof = genera_ambienti_random(N_TESTS)

    ambienti = aggiungi_ostacoli_densi(ambienti, num_rombi=0)

    print(f"--- AVVIO TEST DI COPERTURA: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):

        # Estrae i poligoni e i segmenti specifici per questo run
        poligoni, segmenti = ambienti[test_idx]

        # Punto di partenza fisso
        current_x = np.array([1.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        
        x_history = [current_x.copy()]
        u_history = []
        box_history = []
        
        recovery_in_questo_test = 0
        esito = "Successes" 

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
        # SETUP VIDEO REAL-TIME & SALVATAGGIO
        # ==========================================
        plt.ion() # Enable real-time video mode
        fig_anim, ax_anim = plt.subplots(figsize=(18, 5))
        
        # Draw the fixed environment only once
        for poli in poligoni:
            ax_anim.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        # ax_anim.scatter(target_base[0], target_base[1], color='red', marker='X', s=200, zorder=8)
        ax_anim.scatter(1.0, 5.0, color='blue', s=150, zorder=8)
        
        linea_traj, = ax_anim.plot([], [], color='black', linewidth=2, zorder=6)
        linea_pred, = ax_anim.plot([], [], color='orange', linewidth=2, zorder=5)
        punto_drone = ax_anim.scatter([], [], color='green', s=150, zorder=7)

        linea_angolo_drone, = ax_anim.plot([], [], color='red', linewidth=3, zorder=8)

        punti_lidar_scatter = ax_anim.scatter([], [], color='red', s=10, zorder=9, label="LiDAR Hits")

        box_corrente = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=3, alpha=0.3, zorder=4)
        ax_anim.add_patch(box_corrente)
        
        ax_anim.set_xlim(-2, 52)
        ax_anim.set_ylim(-1, 12)
        ax_anim.grid(True, linestyle='--', alpha=0.5)

        # AGGIUNGI QUESTA LISTA QUI: conterrà i fotogrammi del video
        frames_animazione = []
        # ==========================================

        iterazioni_violazione_box = []

        costo_totale_run = 0.0
        distanza_percorsa = 0.0
        distanza_x = 0.0

        for t in range(N_SIM):

            # 0. Recovery timer management
            if in_recovery:
                x_ref_attuale = target_recovery
                timer_recovery -= 1
                if timer_recovery <= 0:
                    in_recovery = False
                    
                    #Restore constraints
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
            hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=3.0)


            # ==========================================
            # --- CHECK CRASH FISICO (L'Arbitro) ---
            # ==========================================
            DRONE_RADIUS = 0.1
            if len(hits) > 0:
                # Calcoliamo la distanza euclidea REALE tra il drone e i punti colpiti (come in APF)
                distanze_ostacoli = np.linalg.norm(hits - current_x[0:2], axis=1)
                
                if np.min(distanze_ostacoli) <= DRONE_RADIUS:
                    esito = "Crashes"
                    costo_totale_run = 0.0
                    distanza_percorsa = 0.0
                    distanza_x = 0.0
                    print(f"💥 CRASH FISICO RILEVATO allo step {t}! Distanza dal muro: {np.min(distanze_ostacoli):.3f}m")
                    break
            # ==========================================

            
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                
            # --- Direction calculation (final target) (Case 1) ---
            target_rel_x = x_ref_attuale[3] # - current_x[3]
            target_rel_z = x_ref_attuale[4] # - current_x[4]

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
                
                dx = x_sol_prev[5][3] #- current_x[0]
                dz = x_sol_prev[5][4] #- current_x[1]
                

                # # If vectors are too small (e.g. perfect hovering), direct movement toward the target
                # if abs(dx) < 0.05 and abs(dz) < 0.05:#0.01
                #     dx, dz = target_rel_x, target_rel_z

            else:
                # Al primissimo passo (t=0)
                dx = target_rel_x
                dz = target_rel_z


      

            # ==========================================
            # WARM START SAFE BOX (MODIFICATO)
            # ==========================================
            if x_sol_prev is not None:
                # 1. Calcola il bounding box (min e max) dell'INTERA traiettoria predetta al passo precedente
                traj_xmin = np.min(x_sol_prev[:, 0])
                traj_xmax = np.max(x_sol_prev[:, 0])
                traj_zmin = np.min(x_sol_prev[:, 1])
                traj_zmax = np.max(x_sol_prev[:, 1])
                
                # 2. Converti questo Bounding Box in coordinate RELATIVE rispetto al drone attuale (current_x)
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

            if status not in [3, 4]:
                # # Estrae il costo totale (funzione obiettivo) calcolato dall'ottimizzatore C'è QUELLO ATTUALE PIù QUELLO PREDETTO, SE SOMMO TUTTO LO CONTO DUE VOLTE ALLA FINE
                # E POI C'è COSTO TERMINALE *20 MA APF NON HA UN COSTO TERMINALE
                # costo_totale_run += controller.ocp_solver.get_cost()


                # Calcolo del Closed-Loop Cost (Esattamente come in APF)
                err_x = current_x - x_ref_attuale
                # u_sol[0] è l'input effettivamente applicato al drone in questo istante
                err_u = u_sol[0] - np.array([u_hover, u_hover]) 
                
                costo_step = err_x.T @ Q_COST @ err_x + err_u.T @ R_COST @ err_u
                costo_totale_run += costo_step

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
            #     print(f"5. Distance to local target: {dist_to_local:.3f} metri")
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

                if status not in [3, 4]:
                # Estrae il costo totale (funzione obiettivo) calcolato dall'ottimizzatore
                    costo_totale_run += controller.ocp_solver.get_cost()

                # PLAN B: historical warm start
                if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                    for i in range(len(u_history) - 1, -1, -1):
                        past_u = u_history[i]
                        controller.ocp_solver.reset()
                        controller.x_guess = np.tile(current_x, (controller.N, 1))
                        controller.u_guess = past_u
                        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                        if status in [0, 2]:
                            costo_totale_run += controller.ocp_solver.get_cost()
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

                    if status not in [3, 4]:
                    # Estrae il costo totale (funzione obiettivo) calcolato dall'ottimizzatore
                        costo_totale_run += controller.ocp_solver.get_cost()
                    
                    if status in [3, 4]:
                        esito = "Crashes"
                        in_recovery = False
                        costo_totale_run = 0.0

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
                    costo_totale_run = 0.0

                    # --- CHECK POINTS OF PRED_TRAJ OUTSIDE ON THE SAFE BOX ---
                    if x_sol_prev is not None:
                        punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3))
                        print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                        if punti_fuori > 0:
                                crashes_per_feasibility += 1
                    break

            if u_sol is not None:
                u_history.append(u_sol.copy())

            step_dist = np.linalg.norm(x_sol[1, 0:2] - current_x[0:2])
            distanza_percorsa += step_dist
            distanza_x += abs(current_x[0]-x_sol[1,0])
            

            current_x = x_sol[1]
            x_history.append(current_x.copy())

            x_sol_prev = x_sol.copy()
            box_abs_prev = box_abs.copy()

            

            # ==========================================
            # VIDEO FRAME UPDATE
            # ==========================================
            if t % 5 == 0:  # Aggiunto per alleggerire il video e pareggiare l'APF
                # Aggiorna la linea della traiettoria
                traj_attuale = np.array(x_history)
                linea_traj.set_data(traj_attuale[:, 0], traj_attuale[:, 1])

                if x_sol is not None:
                    traj_pred_array = np.array(x_sol)
                    linea_pred.set_data(traj_pred_array[:, 0], traj_pred_array[:, 1])
                
                # Update the drone's position
                punto_drone.set_offsets([[current_x[0], current_x[1]]])

                lunghezza_corpo = 0.8
                theta = current_x[2] 
                dx = (lunghezza_corpo / 2.0) * np.cos(theta)
                dz = (lunghezza_corpo / 2.0) * -np.sin(theta)
                linea_angolo_drone.set_data([current_x[0] - dx, current_x[0] + dx], [current_x[1] - dz, current_x[1] + dz])
                
                # Update the coordinates and dimensions of the green safe-box
                box_w = box_abs[1] - box_abs[0]
                box_h = box_abs[3] - box_abs[2]
                box_corrente.set_xy((box_abs[0], box_abs[2]))
                box_corrente.set_width(box_w)
                box_corrente.set_height(box_h)

                if len(hits) > 0:
                    punti_lidar_scatter.set_offsets(hits)
                else:
                    punti_lidar_scatter.set_offsets(np.empty((0, 2)))

                
                # Change the color to red if the solver goes into Status 4 (Crash/Infeasibility)
                if status in [3, 4]:
                    box_corrente.set_edgecolor('red')
                    box_corrente.set_facecolor('red')
                else:
                    box_corrente.set_edgecolor('lime')
                    box_corrente.set_facecolor('lime')
                
                # ---> TITOLO AGGIORNATO COME RICHIESTO <---
                ax_anim.set_title(f"MPC-NN Target {test_idx +1} | Step: {t} | Dist: {distanza_percorsa:.1f}m | Vx: {current_x[3]:.2f} | Status MPC: {status}")
                
                # Draw the frame on the screen
                fig_anim.canvas.draw()
                fig_anim.canvas.flush_events()
                
                # (Rimosso plt.pause(0.1) per non rallentare l'esecuzione)

                # Converte il disegno corrente a schermo in un'immagine da salvare nel video
                image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
                frames_animazione.append(image)
            # ==========================================



            # # ==========================================
            # # STALL HANDLING AND RETURN TO BASE TARGET
            # # ==========================================
            # # 1. ABSOLUTE SUCCESS CHECK (end test with Success)
            # dist_target_reale = np.linalg.norm(current_x[:2] - target_base[:2])
            # if dist_target_reale < 0.3:
            #     esito = "Successes"
            #     break
                            
            # if t > 0:
            #     # 2. NORMAL STALL HANDLING
            #     spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            #     if spostamento < 0.01:
            #         contatore_stallo += 1
            #     else:
            #         contatore_stallo = 0 

            #     if contatore_stallo >= MAX_STALLO_ITER:
            #         print(f"\n⚠️ STALL DETECTED (Step {t})! Perturb the target upward.")
                    
            #         Z_MAX_SICURA = roof - 0.2  # Margine di sicurezza sotto il soffitto
            #         current_target[1] = min(current_target[1] + 0.20, Z_MAX_SICURA)
            #         x_ref_attuale = current_target.copy()
            #         contatore_stallo = 0

            #     # 3. LOCAL TARGET ARRIVAL CHECK
            #     dist_al_locale = np.linalg.norm(current_x[:2] - current_target[:2])
            #     if dist_al_locale < 0.3:
            #         if not np.array_equal(current_target, target_base):
            #             print(f"\n✅ Elevated target reached. Descend toward the final ground target.")
            #             current_target = target_base.copy()
            #             x_ref_attuale = current_target.copy()
            #             contatore_stallo = 0


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
        print(f"Dist. Totale: {distanza_percorsa:.2f} m | Dist. X: {distanza_x:.2f} m | Costo: {costo_totale_run:.2f}")
        
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
            "costo": costo_totale_run,
            "distanza": distanza_percorsa,
            "distanza_x": distanza_x,
            "iterazione": t,
            "num_violazioni": len(iterazioni_violazione_box),
            "correlazione": tipo_correlazione
        })
        # ----------------------------------------





        # ==========================================
        # END OF VIDEO
        # ==========================================
        # ==========================================
        # ESPORTAZIONE E SALVATAGGIO DEL VIDEO
        # ==========================================
        plt.ioff()
        plt.close(fig_anim) # Chiude la finestra video real-time per liberare memoria

        if len(frames_animazione) > 0:
            cartella_video = "video_MPC-NN.2"
            os.makedirs(cartella_video, exist_ok=True)
            
            print(f"Generazione del video per il Target {test_idx+1} in corso...")
            fig_movie = plt.figure(figsize=(12, 7))
            ax_movie = fig_movie.add_subplot(111)
            ax_movie.axis('off')
            
            # Mostra il primo frame
            im = ax_movie.imshow(frames_animazione[0])
            
            def update_frame(i):
                im.set_data(frames_animazione[i])
                return [im]
            
            # Crea l'animazione dalle immagini salvate
            ani = animation.FuncAnimation(fig_movie, update_frame, frames=len(frames_animazione), blit=True)
            
            # Configura il nome del file video
            nome_video = os.path.join(cartella_video, f"Video_MPC-NN_{test_idx+1:02d}_{esito}.mp4")
            
            # Salva come MP4 (richiede ffmpeg installato sul PC)
            # Se ffmpeg dà problemi, puoi cambiare l'estensione in '.gif' e usare il writer 'pillow'
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

        # # ==========================================
        # # PLOT STATISTICO INDIVIDUALE (Da mettere DENTRO il ciclo 'for test_idx', alla fine)
        # # Sostituisce la vecchia parte commentata "CORRECTION 3"
        # # ==========================================
        # fig_singolo, ax_singolo = plt.subplots(figsize=(18, 5))
        
        # # Disegna SOLO gli ostacoli di questa specifica stanza
        # for poli in poligoni:
        #     ax_singolo.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
            
        # # Disegna un po' di safe box (uno ogni 10 per non pasticciare il disegno)
        # if len(box_history) > 0:
        #     for box in box_history[::10]:
        #         box_w = box[1] - box[0]
        #         box_h = box[3] - box[2]
        #         ax_singolo.add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, edgecolor='lime', facecolor='none', linewidth=0.8, alpha=0.3))
                
        # # Traiettoria
        # traj = np.array(x_history)
        # ax_singolo.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=2.5, label=f'Traiettoria')
        # ax_singolo.plot([50.0, 50.0], [0.0, 11.0], color='red', linestyle='--', linewidth=2, label="Traguardo (Fine Simulazione)")
        # ax_singolo.scatter(1.0, 5.0, color='blue', s=120, label='Start', zorder=5)
        
        # ax_singolo.set_xlim(-2, 52)
        # ax_singolo.set_ylim(-1, 12)
        # ax_singolo.set_title(f"Analisi Test {test_idx+1} | Target Vx: {target_base[3]} m/s | Esito: {esito}")
        # ax_singolo.grid(True, linestyle='--', alpha=0.5)
        # ax_singolo.legend(loc='lower left', fontsize=12)
        
        # plt.tight_layout()
        # plt.show() # Mostra il grafico di questa stanza. Chiudilo per passare al test successivo!


# ==========================================
    # RESOCONTO GLOBALE FINALE (FUORI dal ciclo for, alla fine di main_statistico)
    # ==========================================
    print(f"\n--- RISULTATI FINALI ---")
    print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")
    
    print("\n" + "="*70)
    print("📊 RESOCONTO GLOBALE RECURSIVE FEASIBILITY (Traiettoria k in Box k+1)")
    print("="*70)
    for row in registro_violazioni_globale:
        print(f"Target {row['target']:02d} | Esito: {row['esito']:<9} | DistTot: {row['distanza']:<5.2f}m| DistX: {row['distanza_x']:<5.2f}m | Costo: {row['costo']:<8.2f} | Step Stop: {row['iterazione']:<4} | "
            f"Uscite dal box: {row['num_violazioni']:<3} | Correlazione: {row['correlazione']}")
    print("="*70 + "\n")

    # Mostra solo il grafico a torta riassuntivo alla fine
    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    labels = list(risultati.keys())
    sizes = list(risultati.values())
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0) 
    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax_pie.set_title('Tasso di Successo Globale')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    N_TESTS = 50
    
    main_statistico(N_TESTS)

    # anteprima_ambiente(num_rombi=12)
    