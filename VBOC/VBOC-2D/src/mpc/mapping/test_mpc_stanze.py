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
from test_lidar import get_lidar_hits_2d, min_cube_select_2d

# def simulate_lidar_2d(drone_x, drone_y, obstacles, num_rays=36, max_range=5.0):
#     """Simula il LIDAR a 360 gradi e restituisce la distanza dall'ostacolo più vicino."""
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
    """Restituisce gli ostacoli, il target e il titolo per la stanza selezionata."""
    if id_stanza == 1:
        # STANZA 1: Corridoio (Sopra e sotto liberi)
        obstacles = [
            [1.5, 4.5, 1.0, 3.0],    # Soffitto
            [1.5, 4.5, -5.0, -3.5]   # Pavimento
        ]
        x_ref = np.array([5.0, -1.5, 0.0, 0.0, 0.0, 0.0])
        titolo = "Hallway"
        
    elif id_stanza == 2:
        # STANZA 2: Muro Frontale Centrale (Vicolo Cieco)
        obstacles = [
            [2.0, 3.0, -3.0, 2.0]    # Grande muro in mezzo
        ]
        x_ref = np.array([4.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        titolo = "Front Wall"
        
        
    elif id_stanza == 3:
        # STANZA 3: Muri Sfalsati (Slalom)
        obstacles = [
            [1.0, 2.0, -1.0, 3.0],   # Muro che scende
            [3.0, 4.0, -5.0, -1.5]   # Muro che sale
        ]
        x_ref = np.array([5.0, -2.5, 0.0, 0.0, 0.0, 0.0])
        titolo = "Staggered Walls"
        
    else:
        obstacles = []
        x_ref = np.array([5.0, -2.0, 0.0, 0.0, 0.0, 0.0])
        titolo = "Empty Room"
        
    return obstacles, x_ref, titolo

def plot_mappa_2d(x_history, box_history, obstacles, x_ref, x0, titolo):
    """Disegna la mappa 2D completa con traiettoria e ostacoli."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Disegna gli ostacoli
    for obs in obstacles:
        x_min, x_max, z_min, z_max = obs
        width = x_max - x_min
        height = z_max - z_min
        rect = patches.Rectangle((x_min, z_min), width, height, linewidth=1, 
                                 edgecolor='black', facecolor='gray', alpha=0.6)
        ax.add_patch(rect)
        
    # 2. Disegna la traiettoria
    x_hist = np.array(x_history)
    ax.plot(x_hist[:, 0], x_hist[:, 1], color='blue', label='Traiettoria', 
            linewidth=1.5, marker='o', markersize=4)
    
    # 3. Disegna il box asimmetrico calcolato da Max (ogni 10 step per non appesantire)
    for t in range(0, len(box_history), 10):
        b_xmin, b_xmax, b_zmin, b_zmax = box_history[t]
        width = b_xmax - b_xmin
        height = b_zmax - b_zmin
        drone_box = patches.Rectangle((b_xmin, b_zmin), width, height, 
                                      linewidth=1, edgecolor='green', facecolor='lime', alpha=0.1)
        ax.add_patch(drone_box)
        
    # Start e Target
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
    print("--- Inizializzazione Campagna Test Stanze ---")
    
    params = Parameters('sth')
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)

    DT = params.dt
    SIM_TIME = 5.5
    N_SIM = int(SIM_TIME / DT)

    # LOOP SULLE 3 STANZE
    for STANZA_ID in [1, 2, 3]:
        obstacles, x_ref, titolo_stanza = get_ambiente(STANZA_ID)
        
        print(f"\n=============================================")
        print(f"TEST: {titolo_stanza}")
        print(f"=============================================")

        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        current_x = x0.copy()

        x_history = [x0]
        box_history = []  # Ora salviamo l'intero box, non solo alpha

        u_hover = (model.mass * 9.81) / (2.0 * model.cf)
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x0, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        for t in range(N_SIM):
            
            # 1. Simula LIDAR e crea i raggi tangenti
            hits, radii = get_lidar_hits_2d(current_x[0], current_x[1], obstacles, num_rays=360)
            
            # 2. Algoritmo di Max
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]

            # -- NUOVO: Calcola la posizione relativa del target --
            target_rel_x = x_ref[0] - current_x[0]
            target_rel_z = x_ref[1] - current_x[1]

            # Passa i parametri relativi alla funzione
            xMin_rel, xMax_rel, zMin_rel, zMax_rel, _ = min_cube_select_2d(
                Q_rel, 
                radii, 
                target_rel_x, 
                target_rel_z, 
                drone_radius=0.1
            )
            
            # 3. Trasforma il box in coordinate assolute
            box_abs = np.array([
                xMin_rel + current_x[0],
                xMax_rel + current_x[0],
                zMin_rel + current_x[1],
                zMax_rel + current_x[1]
            ])

            

            # 4. MPC calcola usando i 4 lati assoluti della stanza
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, box_abs)
            
            # # Calcoliamo la distanza minima ai bordi per fare i log a terminale
            # dist_x_min = current_x[0] - box_abs[0]
            # dist_x_max = box_abs[1] - current_x[0]
            # dist_z_min = current_x[1] - box_abs[2]
            # dist_z_max = box_abs[3] - current_x[1]
            # min_dist_to_wall = min(dist_x_min, dist_x_max, dist_z_min, dist_z_max)

            if status == 4:
                print(f" --> L'MPC ha dichiarato la traiettoria INFATTIBILE al passo {t}.")
                # print(f" --> Alpha richiesto: {alpha_curr:.2f}m | Distanza dal muro più vicino: {min_dist_to_wall:.2f}m")
                break
            # else:
                # print(f" --> (Status {status}, alpha_pred {alpha_curr:.2f}, min_dist_muro {min_dist_to_wall:.2f}).")

            current_x = x_sol[1] 
            x_history.append(current_x)
            box_history.append(box_abs)

        # Plotta la stanza prima di passare alla successiva
        plot_mappa_2d(x_history, box_history, obstacles, x_ref, x0, titolo_stanza)

if __name__ == '__main__':
    main()