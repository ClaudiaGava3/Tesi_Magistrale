import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch
from learning import NeuralNetwork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, combinations

# Importo librerie acados/modelli
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from test_lidar import get_lidar_hits_3d, min_cube_select_3d


def get_ambiente(id_stanza):
    """Restituisce gli ostacoli 3D, il target a 12 stati e il titolo per la stanza selezionata."""
    
    # Inizializza il target con 12 zeri: [x,y,z, phi,theta,psi, vx,vy,vz, wx,wy,wz]
    x_ref = np.zeros(12)
    
    if id_stanza == 1:
        # STANZA 1: Corridoio (Sopra e sotto ostacoli, libero in mezzo)
        # Formato ostacolo: [x_min, x_max, y_min, y_max, z_min, z_max]
        obstacles = [
            [1.5, 4.5, -2.0, 2.0, 1.0, 3.0],    # Soffitto (largo 4m su Y)
            [1.5, 4.5, -2.0, 2.0, -5.0, -3.5]   # Pavimento (largo 4m su Y)
        ]
        x_ref[0] = 2.5   # Target X
        x_ref[1] = 0.0   # Target Y
        x_ref[2] = -1.5  # Target Z
        titolo = "Stanza 1: Corridoio Libero 3D"
        
    elif id_stanza == 2:
        # STANZA 2: Muro Frontale Centrale (Vicolo Cieco)
        obstacles = [
            [2.0, 3.0, -2.0, 2.0, -3.0, 2.0]    # Grande muro al centro (largo 4m su Y)
        ]
        x_ref[0] = 4.0
        x_ref[1] = 0.0
        x_ref[2] = -1.0
        titolo = "Stanza 2: Muro Frontale 3D (Traiettoria Bloccata)"
        
    elif id_stanza == 3:
        # STANZA 3: Muri Sfalsati (Slalom)
        obstacles = [
            [1.0, 2.0, -2.0, 2.0, -1.0, 3.0],   # Muro che scende dall'alto
            [3.0, 4.0, -2.0, 2.0, -5.0, -2.5]   # Muro che sale dal basso
        ]
        x_ref[0] = 4.5
        x_ref[1] = 0.0
        x_ref[2] = -1.5
        titolo = "Stanza 3: Pareti Sfalsate 3D"
        
    else:
        obstacles = []
        x_ref[0] = 5.0
        x_ref[1] = 0.0
        x_ref[2] = -2.0
        titolo = "Stanza Vuota 3D"
        
    return obstacles, x_ref, titolo

def plot_mappa_3d(x_history, box_history, obstacles, x_ref, x0, titolo):
    """Disegna la mappa 3D completa con due viste: 3/4 e frontale (X-Z)."""
    # Allarghiamo la figura per farci stare due grafici comodi
    fig = plt.figure(figsize=(15, 7)) 
    
    # Primo grafico: Vista 3D classica (3/4)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(titolo + "\n(Vista 3/4)")
    
    # Secondo grafico: Vista Frontale (Piano X-Z)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(titolo + "\n(Vista Frontale X-Z)")
    # MAGIA: Mettiamo l'elevazione a 0 e ruotiamo di -90 gradi per guardare lungo l'asse Y
    ax2.view_init(elev=0, azim=-90) 

    # Applichiamo il disegno a ENTRAMBI i grafici
    for ax in [ax1, ax2]:
        # 1. Disegna gli ostacoli (AABB)
        for obs in obstacles:
            x_min, x_max, y_min, y_max, z_min, z_max = obs
            # Crea i vertici del parallelepipedo
            v = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                          [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]])
            faces = [[v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]], [v[0],v[1],v[5],v[4]], 
                     [v[2],v[3],v[7],v[6]], [v[1],v[2],v[6],v[5]], [v[4],v[7],v[3],v[0]]]
            ax.add_collection3d(Poly3DCollection(faces, facecolors='gray', linewidths=1, edgecolors='black', alpha=0.3))
            
        # 2. Disegna la traiettoria
        x_hist = np.array(x_history)
        ax.plot(x_hist[:, 0], x_hist[:, 1], x_hist[:, 2], color='blue', label='Traiettoria', linewidth=2.5, marker='o', markersize=3)

        # 3. Disegna il box asimmetrico calcolato da Max
        for t in range(0, len(box_history), 10):
            b_xmin, b_xmax, b_ymin, b_ymax, b_zmin, b_zmax = box_history[t]
            
            # Disegnamo solo i contorni (wireframe) del box
            r = [b_xmin, b_xmax]
            p = [b_ymin, b_ymax]
            q = [b_zmin, b_zmax]
            for s, e in combinations(np.array(list(product(r, p, q))), 2):
                dist = np.sum(np.abs(np.array(s) - np.array(e)))
                if np.isclose(dist, r[1]-r[0]) or np.isclose(dist, p[1]-p[0]) or np.isclose(dist, q[1]-q[0]):
                    ax.plot3D(*zip(s, e), color="lime", alpha=0.2)
            
        # Start e Target
        ax.scatter(x0[0], x0[1], x0[2], color='green', s=100, label='Start')
        ax.scatter(x_ref[0], x_ref[1], x_ref[2], color='red', s=100, marker='X', label='Target')

        # Limiti degli assi
        ax.set_xlim([-1, 6])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-6, 4])
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # Mettiamo la legenda solo nel primo per non intasare la vista
        if ax == ax1:
            ax.legend(loc='upper right')

    plt.tight_layout()
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
    for STANZA_ID in [3, 2, 3]:
        obstacles, x_ref, titolo_stanza = get_ambiente(STANZA_ID)
        
        print(f"\n=============================================")
        print(f"TEST: {titolo_stanza}")
        print(f"=============================================")

        x0 = np.zeros(12)
        current_x = x0.copy()

        x_history = [x0]
        box_history = []  # Ora salviamo l'intero box, non solo alpha

        u_hover = (model.mass * 9.81) / (4.0 * model.cf)
        controller.ocp_solver.reset()
        controller.x_guess = np.tile(x0, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        for t in range(N_SIM):
            
            # 1. Simula LIDAR e crea i raggi tangenti
            hits, radii = get_lidar_hits_3d(current_x[0], current_x[1],  current_x[2], obstacles, num_rays=1000)
            
            # 2. Algoritmo di Max
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                Q_rel[:, 2] -= current_x[2]

            # -- NUOVO: Calcola la posizione relativa del target --
            target_rel_x = x_ref[0] - current_x[0]
            target_rel_y = x_ref[1] - current_x[1]
            target_rel_z = x_ref[2] - current_x[2]

            # Passa i parametri relativi alla funzione
            xMin_rel, xMax_rel, yMin_rel, yMax_rel,zMin_rel, zMax_rel, _ = min_cube_select_3d(
                Q_rel, 
                radii, 
                target_rel_x,
                target_rel_y, 
                target_rel_z, 
                drone_radius=0.1
            )
            
            # 3. Trasforma il box in coordinate assolute
            box_abs = np.array([
                xMin_rel + current_x[0],
                xMax_rel + current_x[0],
                yMin_rel + current_x[1],
                yMax_rel + current_x[1],
                zMin_rel + current_x[2],
                zMax_rel + current_x[2]
            ])

            

            # 4. MPC calcola usando i 4 lati assoluti della stanza
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, box_abs)
            
            # # Calcoliamo la distanza minima ai bordi per fare i log a terminale
            # dist_x_min = current_x[0] - box_abs[0]
            # dist_x_max = box_abs[1] - current_x[0]
            # dist_z_min = current_x[1] - box_abs[2]
            # dist_z_max = box_abs[3] - current_x[1]
            # min_dist_to_wall = min(dist_x_min, dist_x_max, dist_z_min, dist_z_max)

            if status in [3, 4]:
                # print(f"\n⚠️ Muro Teletrasportato (Box Flip)! Il solver è andato in panico. Avvio Recovery Mode...")
                
                # # Resettiamo la "memoria" del solver (Warm Start)
                controller.ocp_solver.reset()
                # Gli diciamo di ipotizzare di stare fermo dov'è
                controller.x_guess = np.tile(current_x, (controller.N, 1))
                controller.u_guess = np.full((controller.N, model.nu), u_hover)
                
                # Ritentiamo a risolvere con la mente locale "pulita"
                x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref, box_abs)
                
                if status in [3, 4]:
                    print(f"❌ Recovery Fallita. Il drone è fisicamente in trappola. Chiusura.")
                    break


            current_x = x_sol[1] 
            x_history.append(current_x)
            box_history.append(box_abs)

        # Plotta la stanza prima di passare alla successiva
        plot_mappa_3d(x_history, box_history, obstacles, x_ref, x0, titolo_stanza)

if __name__ == '__main__':
    main()