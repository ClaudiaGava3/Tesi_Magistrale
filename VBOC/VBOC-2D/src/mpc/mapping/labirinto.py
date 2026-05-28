import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Importo le tue librerie
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from test_lidar import get_lidar_hits_2d, get_lidar_hits_2d_qualsiasi, min_cube_select_2d

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches



def genera_caverna():
    """ Definisce gli ostacoli della caverna e i waypoints"""

    # --- DEFINIZIONE OSTACOLI ---
    ostacoli = []
    ostacoli.append([-2.0, 28.0, -2.0, -1.0])  # Pavimento
    ostacoli.append([-2.0, 28.0, 7.0, 8.0])    # Soffitto
    ostacoli.append([3.6, 4.5, -1.0, 3.6])     # Stalagmite 1
    ostacoli.append([6.0, 10.0, 4.2, 7.0])    # Stalattite 1
    ostacoli.append([5.0, 12.0, -1.0, 3.0])   # Stalagmite 2
    ostacoli.append([11.0, 15.0, 4.7, 7.0])   # Stalagmite 2
    ostacoli.append([13.0, 14.5, -1.0, 3.6])   # Stalagmite 2
    ostacoli.append([15.6, 18.0, 4.2, 7.0])   # Stalagmite 2
    ostacoli.append([15.0, 20.0, -1.0, 3.1])   # Stalagmite 2

    # --- DEFINIZIONE WAYPOINTS (Line of Sight) ---
    waypoints = [
        # np.array([5.2, 3.6, 0.0, 0.0, 0.0, 0.0]),   # WP1
        # np.array([10.0, 3.5, 0.0, 0.0, 0.0, 0.0]),  # WP2
        # np.array([15.0, 4.0, 0.0, 0.0, 0.0, 0.0]),  # WP3
        np.array([22.0, 3.0, 0.0, 0.0, 0.0, 0.0]),  # WP4
    ]
    return ostacoli, waypoints

# def genera_labirinto():
#     """ Labirinto 2D Complesso a scorrimento lungo """
#     ostacoli = []
    
#     # Bordi esterni (Floor e Ceiling) - Allungati fino a 45m
#     ostacoli.append([-2.0, 45.0, -2.0, 0.0])   # Pavimento
#     ostacoli.append([-2.0, 45.0, 10.0, 12.0])  # Soffitto

#     # Ostacolo 1: Stalattite (Forza il passaggio in basso)
#     ostacoli.append([3.0, 4.0, 4.0, 10.0])
    
#     # Ostacolo 2: Stalagmite (Forza il passaggio in alto)
#     ostacoli.append([7.0, 8.0, 0.0, 6.0])

#     # Ostacolo 3: Blocco fluttuante centrale (Scegliamo di passare sotto)
#     ostacoli.append([11.0, 14.0, 4.0, 6.0])

#     # Ostacolo 4: Strettoia orizzontale a imbuto
#     ostacoli.append([17.0, 19.0, 0.0, 3.0])    # Base strettoia
#     ostacoli.append([17.0, 19.0, 7.0, 10.0])   # Tetto strettoia

#     # Ostacolo 5: Muro altissimo (Costringe a radere il soffitto)
#     ostacoli.append([22.0, 23.0, 0.0, 8.0])

#     # Ostacolo 6: Muro bassissimo (Picchiata radente al pavimento)
#     ostacoli.append([26.0, 27.0, 2.0, 10.0])

#     # Ostacolo 7: Tunnel lungo conclusivo
#     ostacoli.append([30.0, 36.0, 0.0, 4.0])    # Pavimento rialzato
#     ostacoli.append([30.0, 36.0, 6.0, 10.0])   # Soffitto ribassato

#     # --- DEFINIZIONE WAYPOINTS (Path Planning Globale Simulato) ---
#     waypoints = [
#         np.array([1.5, 2.0, 0.0, 0.0, 0.0, 0.0]),   # WP1: Scendi per il primo varco
#         np.array([5.0, 2.0, 0.0, 0.0, 0.0, 0.0]),   # WP2: Oltre la prima stalattite
#         np.array([6.0, 8.0, 0.0, 0.0, 0.0, 0.0]),   # WP3: Impenna sopra la stalagmite
#         np.array([9.0, 8.0, 0.0, 0.0, 0.0, 0.0]),   # WP4: Oltre la stalagmite
#         np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # WP5: Scendi per passare sotto l'isola
#         np.array([15.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # WP6: Sotto l'isola fluttuante
#         np.array([16.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP7: Alzati e centrati per la strettoia
#         np.array([20.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP8: Fuori dalla strettoia
#         np.array([21.0, 9.0, 0.0, 0.0, 0.0, 0.0]),  # WP9: Arrampicata rasente al soffitto
#         np.array([24.0, 9.0, 0.0, 0.0, 0.0, 0.0]),  # WP10: Oltre il muro altissimo
#         np.array([25.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # WP11: Picchiata veloce rasente al pavimento
#         np.array([28.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # WP12: Oltre il muro bassissimo
#         np.array([29.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP13: Risali e centrati per il tunnel lungo
#         np.array([37.0, 5.0, 0.0, 0.0, 0.0, 0.0]),  # WP14: Uscita dal tunnel
#         np.array([42.0, 5.0, 0.0, 0.0, 0.0, 0.0])   # WP15: Target Finale
#     ]

#     return ostacoli, waypoints

def genera_labirinto():
    """ Labirinto 2D a condotti stretti e curve a 90 gradi (Pipe Maze) """
    ostacoli = []
    
    # Bordi esterni (Floor e Ceiling)
    ostacoli.append([-2.0, 22.0, -2.0, -0.5])   # Pavimento base
    ostacoli.append([-2.0, 22.0, 9.5, 11.0])    # Soffitto base

    # BLOCCO 1: Chiude sopra e sotto la partenza, lasciando solo il corridoio iniziale
    ostacoli.append([-2.0, 3.5, -0.5, 4.0])
    ostacoli.append([-2.0, 3.5, 6.0, 9.5])

    # POZZO 1 (X: 3.5 -> 5.0) - Il drone deve calarsi qui dentro

    # BLOCCO 2: Muro centrale che forza il drone a stare nel tunnel inferiore
    ostacoli.append([5.0, 9.0, 2.5, 9.5])
    # BLOCCO 3: Rialzo del pavimento nel tunnel inferiore
    ostacoli.append([5.0, 9.0, -0.5, 0.5])

    # POZZO 2 (X: 9.0 -> 10.5) - Il drone deve risalire qui dentro

    # BLOCCO 4: Muro centrale che forza il drone nel tunnel superiore
    ostacoli.append([10.5, 15.0, -0.5, 6.5])
    # BLOCCO 5: Abbassamento del soffitto nel tunnel superiore
    ostacoli.append([10.5, 15.0, 8.5, 9.5])

    # POZZO 3 (X: 15.0 -> 16.5) - Ultimo drop a 90 gradi

    # BLOCCO 6: Chiusura finale
    ostacoli.append([16.5, 20.0, 4.5, 9.5])
    ostacoli.append([16.5, 20.0, -0.5, 1.5])

    # --- DEFINIZIONE WAYPOINTS (Navigazione a 90 gradi) ---
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
    Ambiente con ostacoli generici (muri inclinati, rombi, triangoli).
    Restituisce una lista piatta di segmenti (muri) e i waypoints.
    """
    # Definiamo gli ostacoli come poligoni chiusi (lista di vertici [X, Z])
    poligoni = [
        # 1. Pavimento irregolare (leggera salita verso la fine)
        [[-2.0, -2.0], [25.0, -2.0], [25.0, 2.0], [-2.0, -0.5]],
        
        # 2. Soffitto obliquo (scende e poi risale)
        [[-2.0, 10.0], [8.0, 7.0], [25.0, 9.0], [25.0, 9.0], [-2.0, 12.0]],
        
        # 3. Ostacolo centrale: Un diamante/rombo fluttuante
        [[10.0, 3.5], [11.5, 5.0], [10.0, 6.5], [8.5, 5.0]],
        
        # 4. Una stalattite triangolare storta
        [[16.0, 4.5], [14.0, 3.0], [17.5, 4.0]]
    ]
    
    # Scomponiamo i poligoni in segmenti indipendenti per il LiDAR
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            # Collega il vertice corrente con il successivo (il modulo % chiude la figura)
            punto_A = poli[i]
            punto_B = poli[(i + 1) % n]
            segments.append([punto_A, punto_B])
            
    # Waypoints per navigare tra questi ostacoli irregolari
    waypoints = [
        # np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
        # np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # Passa sotto al rombo
        # np.array([16.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # Schiva la stalattite storta
        np.array([22.0, 5.0, 0.0, 0.0, 0.0, 0.0])   # Target finale
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
    SIM_TIME = 40.0 # Tempo aumentato per coprire tutti i target
    N_SIM = int(SIM_TIME / DT)

    
    target_idx = 0
    TOLLERANZA_WAYPOINT = 0.20

    # x labirinto
    x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    # x caverna
    #x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    ostacoli, waypoints = genera_ambiente_obliquo()
    current_x = x0.copy()

    x_history = [current_x]
    box_history = []
    u_history = []
    
    # Inizializzazione solver
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    # controllo stalli
    contatore_stallo = 0
    MAX_STALLO_ITER = 50

    # controllo recovery
    in_recovery = False
    timer_recovery = 0
    target_recovery = None

    ghost_waypoints = []       # Ricorda i target vecchi spostati per stallo
    mode_history = ['normal']  # Ricorda se in quell'istante era in recovery

    print(f"Inizio volo verso Waypoint {target_idx + 1}...")

    for t in range(N_SIM):
       # 0. Seleziona il target corrente
        if in_recovery:
            x_ref_attuale = target_recovery
            timer_recovery -= 1
            if timer_recovery <= 0:
                in_recovery = False
                # FINE EMERGENZA:
                # Ripristiniamo la Rete Neurale
                controller.ocp_solver.constraints_set(controller.N, "lh", np.zeros(4))
                
                lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                
                # Ripristino velocità
                lbx_e_curr[3:] = [-1.0, -1.0, -1.0]
                ubx_e_curr[3:] = [ 1.0,  1.0,  1.0]
                
                controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)

                print(f"\n🔄 FINE RECOVERY: Missione ripristinata verso WP {target_idx + 1}. Vincoli di sicurezza riattivati.")
        else:
            x_ref_attuale = waypoints[target_idx]

        # 1. LiDAR e Safe-Box
        hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], ostacoli, num_rays=360, max_range=2.0)
        # max range 1.5 per caverna senza recovery help, va bene 2.0 se metto la gestione dell'infeasiility; max range per labirinto 2.0 sennò vede troppo vicino
        Q_rel = hits.copy()
        if len(hits) > 0:
            Q_rel[:, 0] -= current_x[0]
            Q_rel[:, 1] -= current_x[1]
        
        # Passo la posizione relativa del target attuale per guidare l'espansione del box
        target_rel_x = x_ref_attuale[0] - current_x[0]
        target_rel_z = x_ref_attuale[1] - current_x[1]

        xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_2d(
            Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1
        )
        box_abs = np.array([xMin_r + current_x[0], xMax_r + current_x[0], zMin_r + current_x[1], zMax_r + current_x[1]])

        box_history.append(box_abs.copy())

        # ==========================================
        # BLOCCO DI DIAGNOSTICA (MPC DEBUGGER)
        # ==========================================
        if t % 10 == 0: # Stampo ogni 10 passi
            print(f"\n--- DEBUG PASSO {t} ---")
            print(f"1. Posizione Drone : X={current_x[0]:.2f}, Z={current_x[1]:.2f}")
            print(f"2. Box Verde (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Z in [{box_abs[2]:.2f}, {box_abs[3]:.2f}]")
            print(f"3. Target Locale  : X={x_ref_attuale[0]:.2f}, Z={x_ref_attuale[1]:.2f}")
            
            
            # Calcolo la distanza tra drone e target locale
            dist_to_local = np.linalg.norm(current_x[:2] - np.array([target_rel_x, target_rel_z])[:2])
            print(f"5. Distanza da percorrere nel box: {dist_to_local:.3f} metri")
        # ==========================================

        # 2. SOLVE MPC
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

        # ==========================================
        # GESTIONE INFEASIBILITY (Status 3 o 4)
        # ==========================================

        if (status in [3, 4]):
            # ==========================================
            # PIANO A: Reset della memoria (Hovering Guess)
            # ==========================================
            controller.ocp_solver.reset()
            controller.x_guess = np.tile(current_x, (controller.N, 1))
            controller.u_guess = np.full((controller.N, model.nu), u_hover)
            
            x_sol, u_sol, alpha_curr, status =  controller.solve_step(current_x, x_ref_attuale, box_abs)

            # margine_sicurezza = min_dist_to_wall - alpha_curr
            # pericolo = (margine_sicurezza < 0.15) and (status in [0, 2])

            # ==========================================
            # PIANO B (NUOVO): Historical Warm-Start
            # ==========================================
            if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                print(f"\n⚠️ PIANO A FALLITO. Avvio PIANO B (Ricerca ritroso nei controlli passati)...")
                
                # Vado all'indietro partendo dall'ultimo controllo salvato fino al primo
                for i in range(len(u_history) - 1, -1, -1):
                    past_u = u_history[i]
                    
                    controller.ocp_solver.reset()
                    controller.x_guess = np.tile(current_x, (controller.N, 1))
                    controller.u_guess = past_u
                    
                    x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                    
                    if status in [0, 2]:
                        passi_indietro = len(u_history) - i
                        print(f"✅ PIANO B Riuscito! Trovato warm-start valido {passi_indietro} passi fa.")
                        break

            # ==========================================
            # PIANO C: Ritiro al Centro
            # ==========================================
            
            if (status in [3, 4]) and not in_recovery:
                print(f"\n⚠️ PIANO B FALLITO. Avvio PIANO C (Ritiro al Centro con Rilassamento Alpha)...")
                
                passi_indietro = 10  # Numero di passi indietro da cui prendere il box sicuro
                if len(box_history) > passi_indietro:
                    box_sicuro = box_history[-passi_indietro]
                else:
                    box_sicuro = box_history[0]  # Se fallisce subito, torna alla partenza
                
                # Calcolo il centro del box passato
                center_x = (box_sicuro[0] + box_sicuro[1]) / 2.0
                center_z = (box_sicuro[2] + box_sicuro[3]) / 2.0
                
                target_recovery = np.array([center_x, center_z, 0.0, 0.0, 0.0, 0.0])
                x_ref_attuale = target_recovery
                
                in_recovery = True
                timer_recovery = 40 
                
                # ==========================================
                # RILASSAMENTO MATEMATICO DEL VINCOLO (Realistico)
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
                    print("❌ RECOVERY 2 FALLITA. Lo spazio fisico è del tutto inesistente. Chiusura.")
                    break
                else:
                    print("✅ RECOVERY 2 Riuscita! Il drone indietreggia verso il centro ignorando temporaneamente la rete.")
            
            elif status in [3, 4] and in_recovery:
                print("❌ IL DRONE È IN TRAPPOLA FISICA. Impossibile stabilizzare. Chiusura.")
                break
        # ==========================================

        if u_sol is not None:
            u_history.append(u_sol.copy())

        current_x = x_sol[1]
        x_history.append(current_x)

        mode_history.append('recovery' if in_recovery else 'normal')

        # ==========================================
        # GESTIONE DELLO STALLO (Local Minimum Escape)
        # ==========================================

        if t > 0:
            spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            
            if spostamento < 0.01:  # Se si è mosso di meno di 1 cm
                contatore_stallo += 1
            else:
                contatore_stallo = 0  # Resetto il contatore se si sblocca

            # Se è fermo da 50 iterazioni, spplico lo sblocco
            if contatore_stallo >= MAX_STALLO_ITER:
                print(f"\n⚠️ STALLO RILEVATO (Passo {t})! Il drone è intrappolato in un minimo locale.")
                print("   -> Perturbo il target locale verso l'alto")
                
                # salvo vecchio vincolo solo per plot
                ghost_waypoints.append(waypoints[target_idx].copy())

                # Alzo la Z del waypoint corrente
                # x labirinto
                #waypoints[target_idx][1] += 0.50 
                # x caverna
                waypoints[target_idx][1] += 0.20 
                
                # Aggiorniamo waypoint
                x_ref_attuale = waypoints[target_idx] 
                
                contatore_stallo = 0 
        # ==========================================

        # ==========================================
        # 3. LOGICA DI SWITCH DEL TARGET
        # ==========================================
        if not in_recovery:
            dist_al_target = np.linalg.norm(current_x[:2] - waypoints[target_idx][:2])

            if dist_al_target < TOLLERANZA_WAYPOINT:
                if target_idx < len(waypoints) - 1:
                    target_idx += 1
                    print(f"\n✅ Waypoint raggiunto! Passaggio al Target {target_idx + 1} a {waypoints[target_idx][:2]}")
                else:
                    print(f"\n🎯 MISSIONE COMPLETATA! Ultimo target raggiunto al passo {t}.")
                    break # Fine missione
        # ==========================================

    # ==========================================
    # PLOT FINALE DELLA TRAIETTORIA E DEGLI OSTACOLI
    # ==========================================

    x_h = np.array(x_history)
    plt.figure(figsize=(15, 6))
    
    # ==========================================
    # DISEGNO DEGLI OSTACOLI
    # ==========================================

    if len(ostacoli) > 0:
        # Capiamo se stiamo usando la nuova definizione a segmenti o la vecchia a rettangoli
        # Se il primo elemento è una lista di liste (es. [[x1, z1], [x2, z2]]), è un segmento
        if isinstance(ostacoli[0][0], (list, np.ndarray)):
            # Disegna i muri generici obliqui come linee spesse nere
            for seg in ostacoli:
                (x_A, z_A), (x_B, z_B) = seg
                plt.plot([x_A, x_B], [z_A, z_B], color='black', linewidth=4, zorder=2)
        else:
            # Disegna i vecchi ostacoli paralleli come rettangoli grigi pieni
            for obs in ostacoli:
                w = obs[1] - obs[0]
                h = obs[3] - obs[2]
                plt.gca().add_patch(patches.Rectangle((obs[0], obs[2]), w, h, color='dimgray', alpha=0.7, zorder=2))
    # ==========================================
    
    # --- DISEGNA I BOX VERDI ---
    # Usiamo uno step (es. i[::5]) per non disegnare un box ogni singolo istante (diventerebbe tutto verde scuro)
    # Se vuoi vederli TUTTI, togli il [::5] dal ciclo for.
    for i, box in enumerate(box_history[::5]):
        box_w = box[1] - box[0]
        box_h = box[3] - box[2]
        # Aggiungiamo l'etichetta solo al primo box per la legenda
        label = 'Safe-Box (AABB)' if i == 0 else ""
        plt.gca().add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, 
                                              edgecolor='lime', facecolor='none', 
                                              linewidth=1.0, alpha=0.3, label=label))
    
    # # Disegna la traiettoria
    # plt.plot(x_h[:, 0], x_h[:, 1], color='cyan', linewidth=2.5, label='Volo Multi-Target')
    
    # # Disegna tutti i Waypoints
    # for i, wp in enumerate(waypoints):
    #     color = 'red' if i == target_idx else 'orange'
    #     plt.scatter(wp[0], wp[1], color=color, marker='X', s=150, zorder=6, label=f'WP {i+1}' if i==0 else "")

    # ==========================================
    # DISEGNO DELLA TRAIETTORIA (Bicolore)
    # ==========================================
    for i in range(1, len(x_h)):
        # Se in quell'istante era in recovery, usiamo il magenta (o arancione), altrimenti cyan
        colore_tratto = 'magenta' if mode_history[i] == 'recovery' else 'cyan'
        plt.plot(x_h[i-1:i+1, 0], x_h[i-1:i+1, 1], color=colore_tratto, linewidth=2.5)
        
    # Creiamo due linee invisibili solo per farle comparire belle pulite nella legenda
    plt.plot([], [], color='cyan', linewidth=2.5, label='Navigazione Standard')
    plt.plot([], [], color='magenta', linewidth=2.5, label='Manovra di Recovery')

    # ==========================================
    # DISEGNO DEI TARGET (Attuali e Ghost)
    # ==========================================
    # 1. Disegna i vecchi target perturbati (sfocati e più piccoli)
    for g_wp in ghost_waypoints:
        plt.scatter(g_wp[0], g_wp[1], color='red', marker='X', s=80, alpha=0.25)
    
    if ghost_waypoints: # Aggiunge alla legenda solo se ci sono stati stalli
        plt.scatter([], [], color='red', marker='X', s=80, alpha=0.25, label='Target perturbati (Stallo)')

    # 2. Disegna i Waypoints reali attuali
    for i, wp in enumerate(waypoints):
        colore_wp = 'red' if i == target_idx else 'orange'
        testo_label = f'Target Finale' if i == target_idx else ""
        plt.scatter(wp[0], wp[1], color=colore_wp, marker='X', s=150, zorder=6, label=testo_label)
    
    plt.scatter(x0[0], x0[1], color='lime', s=100, label='Start', zorder=6)
    plt.title('Autonomous Navigation')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    
    # Mostra la legenda fuori dal grafico o in un angolo
    #plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.show()

    

if __name__ == '__main__':
    main()