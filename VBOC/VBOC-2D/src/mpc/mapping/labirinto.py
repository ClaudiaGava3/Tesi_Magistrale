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

def genera_caverna():
    """ Definisce gli ostacoli della caverna """
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
    return ostacoli

def main():
    print("--- Avvio Navigazione Multi-Target (Waypoints) ---")
    params = Parameters('sth')
    params.act = 'gelu'
    params.build = True 

    model = Model(params)
    controller = MpcController(model)

    DT = params.dt
    SIM_TIME = 20.0 # Tempo aumentato per coprire tutti i target
    N_SIM = int(SIM_TIME / DT)

    # --- DEFINIZIONE WAYPOINTS (Line of Sight) ---
    # Creiamo una scia di molliche di pane sicura
    waypoints = [
        np.array([5.2, 3.6, 0.0, 0.0, 0.0, 0.0]),   # WP1: Sali dritto per superare la stalagmite
        np.array([10.0, 3.5, 0.0, 0.0, 0.0, 0.0]),   # WP2: Ora vai a destra, sopra la stalagmite
         np.array([15.0, 4.0, 0.0, 0.0, 0.0, 0.0]),  # WP3: Scendi per evitare la stalattite
        np.array([22.0, 3.0, 0.0, 0.0, 0.0, 0.0]),  # WP5: Risali per la seconda stalagmite
    ]
    target_idx = 0
    TOLLERANZA_WAYPOINT = 0.20

    x0 = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    ostacoli = genera_caverna()
    current_x = x0.copy()

    x_history = [current_x]
    box_history = []
    
    # Inizializzazione solver
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)

    print(f"Inizio volo verso Waypoint {target_idx + 1}...")

    for t in range(N_SIM):
        # 0. Seleziona il target corrente
        x_ref_attuale = waypoints[target_idx]

        # 1. LiDAR e Safe-Box
        hits, radii = get_lidar_hits_2d(current_x[0], current_x[1], ostacoli, num_rays=360, max_range=1.0)
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

        if status in [3, 4]:
            print(f"\n⚠️ Muro Teletrasportato (Box Flip)! Il solver è andato in panico. Avvio Recovery Mode...")
            
            # Resettiamo la "memoria" del solver (Warm Start)
            controller.ocp_solver.reset()
            # Gli diciamo di ipotizzare di stare fermo dov'è
            controller.x_guess = np.tile(current_x, (controller.N, 1))
            controller.u_guess = np.full((controller.N, model.nu), u_hover)
            
            # Ritentiamo a risolvere con la mente locale "pulita"
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
            
            if status in [3, 4]:
                print(f"❌ Recovery Fallita. Il drone è fisicamente in trappola. Chiusura.")
                break
            else:
                print(f"✅ Recovery Riuscita! Il drone ha ricalcolato la traiettoria nel nuovo Box.")
        
        current_x = x_sol[1]
        x_history.append(current_x)

        # 3. LOGICA DI SWITCH DEL TARGET
        # Calcoliamo la distanza euclidea tra drone e waypoint corrente (solo X e Z)
        dist_al_target = np.linalg.norm(current_x[:2] - x_ref_attuale[:2])

        if dist_al_target < TOLLERANZA_WAYPOINT:
            if target_idx < len(waypoints) - 1:
                target_idx += 1
                print(f"\n✅ Waypoint raggiunto! Passaggio al Target {target_idx + 1} a {waypoints[target_idx][:2]}")
            else:
                print(f"\n🎯 MISSIONE COMPLETATA! Ultimo target raggiunto al passo {t}.")
                break # Fine missione

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
    
    # Disegna la traiettoria
    plt.plot(x_h[:, 0], x_h[:, 1], color='cyan', linewidth=2.5, label='Volo Multi-Target')
    
    # Disegna tutti i Waypoints
    for i, wp in enumerate(waypoints):
        color = 'red' if i == target_idx else 'orange'
        plt.scatter(wp[0], wp[1], color=color, marker='X', s=150, zorder=6, label=f'WP {i+1}' if i==0 else "")
    
    plt.scatter(x0[0], x0[1], color='lime', s=100, label='Start', zorder=6)
    plt.title('Navigazione Semi-Autonoma: Traiettoria e Safe-Boxes')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    
    # Mostra la legenda fuori dal grafico o in un angolo
    #plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()