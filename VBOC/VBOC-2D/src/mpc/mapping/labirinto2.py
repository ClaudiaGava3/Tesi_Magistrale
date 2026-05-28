import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Importo le tue librerie
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from test_lidar import get_lidar_hits_2d, min_cube_select_2d

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
    ostacoli.append([6.0, 10.0, 3.8, 7.0])    # Stalattite 1
    ostacoli.append([5.0, 12.0, -1.0, 3.0])   # Stalagmite 2
    ostacoli.append([11.0, 15.0, 4.7, 7.0])   # Stalagmite 2
    ostacoli.append([13.0, 14.5, -1.0, 3.6])   # Stalagmite 2
    ostacoli.append([15.6, 18.0, 4.0, 7.0])   # Stalagmite 2
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
        np.array([4.25, 5.0, 0.0, 0.0, 0.0, 0.0]),   # WP1
        np.array([7.25, 1.5, 0.0, 0.0, 0.0, 0.0]),   # WP2
        np.array([10.0, 2.0, 0.0, 0.0, 0.0, 0.0]),  # WP3
        # np.array([12.0, 7.2, 0.0, 0.0, 0.0, 0.0]),  # WP3
        # np.array([9.75, 7.5, 0.0, 0.0, 0.0, 0.0]),  # WP4
        np.array([15.75, 5.5, 0.0, 0.0, 0.0, 0.0]),  # WP5
        np.array([15.75, 3.0, 0.0, 0.0, 0.0, 0.0]),  # WP6
        np.array([25.5, 3.0, 0.0, 0.0, 0.0, 0.0])    # WP7
    ]

    return ostacoli, waypoints

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
    #x0 = np.array([-5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    # x caverna
    x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    ostacoli, waypoints = genera_caverna()
    current_x = x0.copy()

    x_history = [current_x]
    box_history = []
    
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
                # FINE EMERGENZA: Ripristiniamo il vincolo rigido della Rete Neurale (Distanza - Alpha >= 0.0)
                controller.ocp_solver.constraints_set(controller.N, "lh", np.zeros(4))
                print(f"\n🔄 FINE RECOVERY: Missione ripristinata verso WP {target_idx + 1}. Vincoli di sicurezza riattivati.")
        else:
            x_ref_attuale = waypoints[target_idx]


        # nella caverna con percorso guidato e con tanti ostacoli meglio tenere il raggio del lidar stretto (ex 1.5 e box default 1.0). se invece faccio guida autonoma funziona sia con raggio stretto e va solo in stallo, sia con raggio più grande ma mi serve recoverì per alcuni punti; nel labirinto meglio avere raggio più grande per la vista del lidar soprattutto poi per guida autonoma
        # 1. LiDAR e Safe-Box
        hits, radii = get_lidar_hits_2d(current_x[0], current_x[1], ostacoli, num_rays=360, max_range=1.5)
        Q_rel = hits.copy()
        if len(hits) > 0:
            Q_rel[:, 0] -= current_x[0]
            Q_rel[:, 1] -= current_x[1]
        
        # Passiamo la posizione relativa del target attuale per guidare l'espansione del box
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
        if t % 10 == 0: # Stampiamo ogni 10 passi per non intasare il terminale
            print(f"\n--- DEBUG PASSO {t} ---")
            print(f"1. Posizione Drone : X={current_x[0]:.2f}, Z={current_x[1]:.2f}")
            print(f"2. Box Verde (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Z in [{box_abs[2]:.2f}, {box_abs[3]:.2f}]")
            print(f"3. Target Locale  : X={x_ref_attuale[0]:.2f}, Z={x_ref_attuale[1]:.2f}")
            
            
            # Calcoliamo la distanza tra drone e target locale
            dist_to_local = np.linalg.norm(current_x[:2] - np.array([target_rel_x, target_rel_z])[:2])
            print(f"5. Distanza da percorrere nel box: {dist_to_local:.3f} metri")
        # ==========================================

        # 2. SOLVE MPC
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

        # ==========================================
        # GESTIONE INFEASIBILITY (Status 3 o 4)
        # ==========================================

        if status in [3, 4]:
            # RECOVERY 1: Reset della memoria
            controller.ocp_solver.reset()
            controller.x_guess = np.tile(current_x, (controller.N, 1))
            controller.u_guess = np.full((controller.N, model.nu), u_hover)
            
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
            
            if status in [3, 4] and not in_recovery:
                print(f"\n⚠️ RECOVERY 1 FALLITA. Avvio RECOVERY 2 (Ritiro al Centro con Rilassamento Alpha)...")
                
                # Calcola il centro ESATTO dello spazio libero REALE (nessun trucco sui muri)
                center_x = (box_abs[0] + box_abs[1]) / 2.0
                center_z = (box_abs[2] + box_abs[3]) / 2.0
                target_recovery = np.array([center_x, center_z, 0.0, 0.0, 0.0, 0.0])
                x_ref_attuale = target_recovery
                
                in_recovery = True
                timer_recovery = 20 
                
                # ==========================================
                # RILASSAMENTO MATEMATICO DEL VINCOLO (Realistico)
                # Diciamo ad Acados: "Ti autorizzo a violare l'Alpha di sicurezza, 
                # purche' non sbatti contro i muri fisici".
                # Impostiamo il lower bound del vincolo terminale a -0.05 invece di 0.0
                # ==========================================
                controller.ocp_solver.constraints_set(controller.N, "lh", np.full(4, -0.05))
                
                # Ritentiamo la solve usando i MURI REALI
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
        
        current_x = x_sol[1]
        x_history.append(current_x)

        mode_history.append('recovery' if in_recovery else 'normal')

        # ==========================================
        # GESTIONE DELLO STALLO (Local Minimum Escape)
        # ==========================================
        # Verifichiamo lo spostamento rispetto al passo precedente


        if t > 0:
            spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            
            if spostamento < 0.01:  # Se si è mosso di meno di 1 cm
                contatore_stallo += 1
            else:
                contatore_stallo = 0  # Resetta il contatore se si sblocca

            # Se è fermo da 50 iterazioni, agiamo
            if contatore_stallo >= MAX_STALLO_ITER:
                print(f"\n⚠️ STALLO RILEVATO (Passo {t})! Il drone è intrappolato in un minimo locale.")
                print("   -> Perturbo il target locale verso l'alto di 20 cm...")
                
                # --- SALVA IL VECCHIO TARGET PRIMA DI SPOSTARLO ---
                ghost_waypoints.append(waypoints[target_idx].copy())

                # Alziamo la Z del waypoint corrente di 20 cm
                waypoints[target_idx][1] += 0.20 
                
                # Aggiorniamo subito la variabile usata nel ciclo per il prossimo passo
                x_ref_attuale = waypoints[target_idx] 
                
                # Resettiamo il contatore per dargli tempo di muoversi
                contatore_stallo = 0 
        # ==========================================

        # ==========================================
        # 3. LOGICA DI SWITCH DEL TARGET
        # ==========================================
        # Il controllo di arrivo si fa SOLO se non siamo in emergenza, 
        # e si calcola la distanza rispetto al Waypoint VERO, non a quello temporaneo.
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

    # --- PLOT FINALE ---
    x_h = np.array(x_history)
    plt.figure(figsize=(15, 6))
    
    # Disegna gli ostacoli
    for obs in ostacoli:
        plt.gca().add_patch(patches.Rectangle((obs[0], obs[2]), obs[1]-obs[0], obs[3]-obs[2], color='dimgray', alpha=0.7))
    
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
    plt.title('Guided Navigation')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    
    # Mostra la legenda fuori dal grafico o in un angolo
    #plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()