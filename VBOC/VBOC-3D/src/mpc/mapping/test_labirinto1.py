import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from itertools import product, combinations
import time

# Importo le tue librerie
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from MPC.mapping.lidar import get_lidar_hits_3d, min_cube_select_3d, get_lidar_hits_3d_qualsiasi

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 20,     
    'axes.labelsize': 16,     
    'xtick.labelsize': 10,    
    'ytick.labelsize': 10,    
    'legend.fontsize': 14,    
    'font.size': 14           
})

def genera_caverna_3d():
    """ 
    Definisce gli ostacoli della caverna 3D.
    Formato: [x_min, x_max, y_min, y_max, z_min, z_max]
    Il corridoio volabile va da Y = -2.0 a Y = +2.0
    """
    ostacoli = []
    # 1. Soffitto e Pavimento
    ostacoli.append([-2.0, 28.0, -2.0, 2.0, -2.0, -1.0])  # Pavimento
    ostacoli.append([-2.0, 28.0, -2.0, 2.0, 7.0, 8.0])    # Soffitto
    
    # 2. Pareti Laterali (Chiudono il tunnel)
    ostacoli.append([-2.0, 28.0, 2.0, 3.0, -2.0, 8.0])    # Muro Destro (Fondo)
    ostacoli.append([-2.0, 28.0, -3.0, -2.0, -2.0, 8.0])  # Muro Sinistro (Fronte)
    
    # 3. Ostacoli interni (Stalattiti e Stalagmiti)
    ostacoli.append([3.6, 4.5, -2.0, 2.0, -1.0, 3.6])     # Stalagmite 1
    ostacoli.append([6.0, 10.0, -2.0, 2.0, 4.2, 7.0])     # Stalattite 1
    ostacoli.append([5.0, 12.0, -2.0, 2.0, -1.0, 3.0])    # Stalagmite 2
    ostacoli.append([11.0, 15.0, -2.0, 2.0, 4.7, 7.0])    # Stalattite 2
    ostacoli.append([13.0, 14.5, -2.0, 2.0, -1.0, 3.6])   # Stalagmite 3
    ostacoli.append([15.6, 18.0, -2.0, 2.0, 4.2, 7.0])    # Stalattite 3
    ostacoli.append([15.0, 20.0, -2.0, 2.0, -1.0, 3.1])   # Stalagmite 4

    # --- DEFINIZIONE WAYPOINTS 12D ---
    waypoints = [
        # np.array([5.2, 0.0, 4.0, 0,0,0, 0,0,0, 0,0,0]),  # WP1
        # np.array([10.0, 0.0, 3.6, 0,0,0, 0,0,0, 0,0,0]), # WP2
        # np.array([15.0, 0.0, 4.0, 0,0,0, 0,0,0, 0,0,0]), # WP3
        np.array([22.0, 0.0, 4.0, 0,0,0, 0,0,0, 0,0,0])  # WP4
    ]
    
    return ostacoli, waypoints


# def genera_stanza_irregolare_3d():
#     """ 
#     Stanza con pareti laterali oblique e ostacoli centrali irregolari (solidi).
#     Modificata per un test più fluido con N=16.
#     """
#     ostacoli = []

#     # ==========================================
#     # 1. PARETI LATERALI OBLIQUE (Confini del corridoio)
#     # ==========================================
#     # Addolcito l'imbuto per evitare restringimenti estremi
#     ostacoli.extend([
#         [[0.0, 4.0, -1.0], [12.0, 2.5, -1.0], [12.0, 2.5, 5.0], [0.0, 4.0, 5.0]], 
#         [[12.0, 2.5, -1.0], [25.0, 4.0, -1.0], [25.0, 4.0, 5.0], [12.0, 2.5, 5.0]],
        
#         [[0.0, -4.0, -1.0], [12.0, -2.5, -1.0], [12.0, -2.5, 5.0], [0.0, -4.0, 5.0]],
#         [[12.0, -2.5, -1.0], [25.0, -4.0, -1.0], [25.0, -4.0, 5.0], [12.0, -2.5, 5.0]]
#     ])

#     # ==========================================
#     # 2. POLIEDRO ASIMMETRICO (Sostituisce il muro tagliente)
#     # ==========================================
#     # Sfaccettato per invitare lo scivolamento laterale
#     A = [4.0, -0.5, -1.0]
#     B = [6.0, 1.5, -1.0]
#     C = [5.0, 3.0, -1.0]
#     Top1 = [4.5, 0.5, 2.0]
#     Top2 = [5.5, 2.0, 3.0]
#     ostacoli.extend([
#         [A, B, C],
#         [A, B, Top2, Top1],
#         [B, C, Top2],
#         [C, A, Top1, Top2] 
#     ])

#     # ==========================================
#     # 3. CRISTALLO FLUTTUANTE (Tetraedro Irregolare, 4 triangoli)
#     # ==========================================
#     # Sollevato per garantire un passaggio pulito sotto
#     P1 = [12.0, -2.0, 3.5]
#     P2 = [14.0,  0.0, 4.0]
#     P3 = [12.5,  1.0, 3.5]
#     P4 = [13.0, -0.5, 6.0] 
#     ostacoli.extend([
#         [P1, P2, P3],
#         [P1, P2, P4],
#         [P2, P3, P4],
#         [P3, P1, P4]
#     ])

#     # ==========================================
#     # 4. RAMPA/CUNEO LATERALE (3 Quads, 2 Triangoli)
#     # ==========================================
#     # Rampa posizionata per guidare dolcemente verso il bersaglio
#     R1 = [18.0, -3.5, -1.0]
#     R2 = [21.0, -3.5, -1.0]
#     R3 = [21.0, -0.5, -1.0]
#     R4 = [18.0, -0.5, -1.0]
#     R_Top1 = [21.0, -3.5, 2.0]
#     R_Top2 = [21.0, -0.5, 2.0]
#     ostacoli.extend([
#         [R1, R2, R3, R4],             
#         [R2, R3, R_Top2, R_Top1],     
#         [R1, R4, R_Top2, R_Top1],     
#         [R1, R2, R_Top1],             
#         [R4, R3, R_Top2]              
#     ])

#     # Target finale
#     waypoints = [
#         np.array([24.0, 0.0, 1.5,  0,0,0,  0,0,0,  0,0,0], dtype=np.float64)
#     ]
#     return ostacoli, waypoints

def genera_stanza_irregolare_3d():
    """ 
    Stanza Challenge: 
    - Pareti oblique.
    - Piramide a terra.
    - Stalattite (tetraedro rovesciato) per forzare il tuffo.
    - Rampa finale con il target nascosto esattamente dietro.
    """
    ostacoli = []

    # ==========================================
    # 1. PARETI LATERALI OBLIQUE
    # ==========================================
    ostacoli.extend([
        [[0.0, 4.0, -1.0], [12.0, 2.5, -1.0], [12.0, 2.5, 5.0], [0.0, 4.0, 5.0]], 
        [[12.0, 2.5, -1.0], [25.0, 4.0, -1.0], [25.0, 4.0, 5.0], [12.0, 2.5, 5.0]],
        
        [[0.0, -4.0, -1.0], [12.0, -2.5, -1.0], [12.0, -2.5, 5.0], [0.0, -4.0, 5.0]],
        [[12.0, -2.5, -1.0], [25.0, -4.0, -1.0], [25.0, -4.0, 5.0], [12.0, -2.5, 5.0]]
    ])

    # ==========================================
    # 2. POLIEDRO A TERRA (Invito a salire o deviare)
    # ==========================================
    A = [4.0, -0.5, -1.0]
    B = [6.0, 1.5, -1.0]
    C = [5.0, 3.0, -1.0]
    Top1 = [4.5, 0.5, 2.0]
    Top2 = [5.5, 2.0, 3.0]
    ostacoli.extend([
        [A, B, C],
        [A, B, Top2, Top1],
        [B, C, Top2],
        [C, A, Top1, Top2] 
    ])

    # ==========================================
    # 3. LA STALATTITE (Tetraedro rovesciato)
    # ==========================================
    # La base è attaccata in alto, la punta scende pericolosamente fino a Z=1.0
    P1 = [12.0, -2.0, 4.5] # Base alta
    P2 = [14.0,  0.0, 4.5] # Base alta
    P3 = [12.5,  1.5, 4.5] # Base alta
    P4 = [13.0, -0.5, 1.0] # Punta verso il basso!
    ostacoli.extend([
        [P1, P2, P3],
        [P1, P2, P4],
        [P2, P3, P4],
        [P3, P1, P4]
    ])

    # ==========================================
    # 4. RAMPA LATERALE (Nasconde il target)
    # ==========================================
    # Occupa la parte destra (da Y = -3.5 a Y = -0.5)
    R1 = [18.0, -3.5, -1.0]
    R2 = [21.0, -3.5, -1.0]
    R3 = [21.0, -0.5, -1.0]
    R4 = [18.0, -0.5, -1.0]
    R_Top1 = [21.0, -3.5, 2.5] # Altezza rampa aumentata per nascondere meglio
    R_Top2 = [21.0, -0.5, 2.5]
    ostacoli.extend([
        [R1, R2, R3, R4],             
        [R2, R3, R_Top2, R_Top1],     
        [R1, R4, R_Top2, R_Top1],     
        [R1, R2, R_Top1],             
        [R4, R3, R_Top2]              
    ])

    # ==========================================
    # TARGET NASCOSTO
    # ==========================================
    # Piazzato a X=22.5 (subito dietro la rampa che finisce a X=21)
    # Piazzato a Y=-2.0 (esattamente nella fascia bloccata dalla rampa)
    # Piazzato a Z=-0.5 (rasoterra, costringe a tuffarsi dopo averla superata)
    waypoints = [
        np.array([22.5, -2.0, -0.5,  0,0,0,  0,0,0,  0,0,0], dtype=np.float64)
    ]
    
    return ostacoli, waypoints

def genera_stanza_mista_3d():
    """ 
    Riproduzione fedele dello schizzo: 
    Ostacoli obliqui, prismi triangolari e pareti non allineate agli assi.
    """
    ostacoli = []
    
    # 1. MURO OBLIQUO INIZIALE (A SINISTRA)
    # Taglia in diagonale il piano X-Y, costringendo il drone a deviare a destra
    ostacoli.append([[3.0, 4.0, -2.0], [7.0, -1.0, -2.0], [7.0, -1.0, 6.0], [3.0, 4.0, 6.0]])
    
    # 2. PRISMA TRIANGOLARE (LA RAMPA AL CENTRO)
    # Questo testa la capacità del nuovo LiDAR di leggere facce a 3 e 4 vertici
    # - Base rettangolare a terra
    ostacoli.append([[9.0, -2.5, -2.0], [13.0, -2.5, -2.0], [13.0, 2.0, -2.0], [9.0, 2.0, -2.0]])
    # - Scivolo obliquo (Rampa)
    ostacoli.append([[9.0, -2.5, -2.0], [13.0, -2.5, 3.0], [13.0, 2.0, 3.0], [9.0, 2.0, -2.0]])
    # - Lato Sinistro (TRIANGOLO!)
    ostacoli.append([[9.0, -2.5, -2.0], [13.0, -2.5, -2.0], [13.0, -2.5, 3.0]])
    # - Lato Destro (TRIANGOLO!)
    ostacoli.append([[9.0, 2.0, -2.0], [13.0, 2.0, -2.0], [13.0, 2.0, 3.0]])
    # - Parete verticale sul retro
    ostacoli.append([[13.0, -2.5, -2.0], [13.0, 2.0, -2.0], [13.0, 2.0, 3.0], [13.0, -2.5, 3.0]])

    # 3. MENSOLA SOSPESA E INCLINATA (A DESTRA)
    # Pende dall'alto verso il basso e in diagonale. Il drone deve passarci SOTTO o a LATO.
    # - Faccia frontale inclinata
    ostacoli.append([[14.0, -4.0, 2.0], [17.0, 4.0, 1.0], [17.0, 4.0, 6.0], [14.0, -4.0, 8.0]])
    # - Tetto inclinato
    ostacoli.append([[14.0, -4.0, 8.0], [17.0, 4.0, 6.0], [18.0, 4.0, 6.0], [15.0, -4.0, 8.0]])

    # Target posizionato in basso e leggermente a destra come nello schizzo
    waypoints = [
        np.array([22.0, -1.5, 0.0,  0,0,0,  0,0,0,  0,0,0], dtype=np.float64)
    ]
    return ostacoli, waypoints

def crea_animazione_3d(x_history, box_history, ostacoli, waypoints):
    """Genera un'animazione video del volo nel tunnel 3D"""
    print("\n🎬 Generazione del Video 3D in corso... (Potrebbe richiedere un minuto)")
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Riduciamo il numero di frame per l'animazione (1 frame ogni 3 step per fluidità)
    step_frame = 3
    x_h = np.array(x_history)[::step_frame]
    b_h = box_history[::step_frame]
    
    # Elementi grafici che si aggiorneranno
    linea_scia, = ax.plot([], [], [], color='cyan', linewidth=2.5, label='Trajectory')
    punto_drone, = ax.plot([], [], [], marker='o', color='blue', markersize=8, label='Drone')
    
    # Lista per le linee del Box Verde wireframe
    box_lines = [ax.plot([], [], [], color="lime", alpha=0.6, linewidth=1.5)[0] for _ in range(12)]

    #--- Setup Ambiente Statico per ostacoli paralleli---
    for i, obs in enumerate(ostacoli):
        if i == 3: continue # Muro "di vetro" per la telecamera
        x_min, x_max, y_min, y_max, z_min, z_max = obs
        v = np.array([[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
                      [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]])
        faces = [[v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]], [v[0],v[1],v[5],v[4]], 
                 [v[2],v[3],v[7],v[6]], [v[1],v[2],v[6],v[5]], [v[4],v[7],v[3],v[0]]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors='gray', linewidths=0.5, edgecolors='black', alpha=0.4))

    # # --- Setup Ambiente Statico per ostacoli qualsiasi---
    # # Disegniamo gli ostacoli (che ora sono singole facce poligonali)
    # for faccia in ostacoli:
    #     # "faccia" è già la lista dei vertici [X,Y,Z], la passiamo direttamente!
    #     poly = Poly3DCollection([faccia], facecolors='gray', linewidths=1.0, edgecolors='black', alpha=0.4)
    #     ax.add_collection3d(poly)

    # Disegna i Waypoints
    for i, wp in enumerate(waypoints):
        ax.scatter(wp[0], wp[1], wp[2], color='red', marker='X', s=100, zorder=6)

    # Settaggi vista
    ax.set_xlim([-1, 24])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-2, 8])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title("Autonomous Flight in the 3D Tunnel")
    ax.view_init(elev=10, azim=-60)
    # ax.view_init(elev=0, azim=0)
    ax.legend(loc='upper right')

    # Funzione di aggiornamento frame
    def update(num):
        # 1. Aggiorna Scia e Drone
        linea_scia.set_data(x_h[:num, 0], x_h[:num, 1])
        linea_scia.set_3d_properties(x_h[:num, 2])
        punto_drone.set_data([x_h[num, 0]], [x_h[num, 1]])
        punto_drone.set_3d_properties([x_h[num, 2]])
        
        # 2. Aggiorna Box Verde Wireframe
        if num < len(b_h):
            bx_min, bx_max, by_min, by_max, bz_min, bz_max = b_h[num]
            r, p, q = [bx_min, bx_max], [by_min, by_max], [bz_min, bz_max]
            
            line_idx = 0
            for s, e in combinations(np.array(list(product(r, p, q))), 2):
                dist = np.sum(np.abs(np.array(s)-np.array(e)))
                if np.isclose(dist, bx_max-bx_min) or np.isclose(dist, by_max-by_min) or np.isclose(dist, bz_max-bz_min):
                    if line_idx < 12:
                        box_lines[line_idx].set_data([s[0], e[0]], [s[1], e[1]])
                        box_lines[line_idx].set_3d_properties([s[2], e[2]])
                        line_idx += 1
                        
        return [linea_scia, punto_drone] + box_lines

    ani = animation.FuncAnimation(fig, update, frames=len(x_h), interval=50, blit=True)
    
    # Salvataggio in GIF (Non richiede installazioni esterne come ffmpeg)
    output_file = "labirinto_3d.gif"
    ani.save(output_file, writer='pillow', fps=20)
    print(f"✅ Video salvato con successo: {output_file}")
    plt.show()


def main():
    print("--- Avvio Navigazione Multi-Target 3D ---")
    params = Parameters('sth')
    params.act = 'gelu'
    params.build = True 

    model = Model(params)
    controller = MpcController(model)


    DT = params.dt
    SIM_TIME = 100.0 
    N_SIM = int(SIM_TIME / DT)

    
    target_idx = 0
    TOLLERANZA_WAYPOINT = 0.30 # Aumentata leggermente in 3D

    # x labirinto
    x0 = np.array([1.0, 0.0, 4.5, 0,0,0, 0,0,0, 0,0,0]) # Partenza a Z=5.0
    # x ostacoli obliqui
    #x0 = np.array([1.0, 2.0, 2.0, 0,0,0, 0,0,0, 0,0,0]) # Partenza a Z=5.0
    
    ostacoli, waypoints = genera_caverna_3d()
    #ostacoli, waypoints = genera_stanza_irregolare_3d()
    current_x = x0.copy()

    x_history = [current_x]
    box_history = []
    u_history = []

    # Inizializzazione solver
    u_hover = (model.mass * 9.81) / (4.0 * model.cf) # diviso 4.0 per il 3D
    controller.ocp_solver.reset()
    controller.x_guess = np.tile(x0, (controller.N, 1))
    controller.u_guess = np.full((controller.N, model.nu), u_hover)



    
    # Controllo stalli
    contatore_stallo = 0
    MAX_STALLO_ITER = 50

    in_recovery = False
    timer_recovery = 0
    target_recovery = np.zeros(12)
    
    ghost_waypoints = []       # Ricorda i target vecchi spostati per stallo
    mode_history = ['normal']  # Ricorda se in quell'istante era in recovery

    print(f"Inizio volo verso Waypoint {target_idx + 1}...")

    for t in range(N_SIM):

        if in_recovery:
            x_ref_attuale = target_recovery
            timer_recovery -= 1
            
            if timer_recovery <= 0:
                in_recovery = False

                # controller.resetHorizon(params.N)
                
                # # 2. Ridimensioniamo le memorie per tornare alla normalità
                # controller.x_guess = np.tile(current_x, (controller.N, 1))
                # controller.u_guess = np.full((controller.N, model.nu), u_hover)

                # FINE EMERGENZA:
                # Ripristiniamo la Rete Neurale
                controller.ocp_solver.constraints_set(controller.N, "lh", np.zeros(6))
                
                lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")




                
                # Ripristino velocità
                lbx_e_curr[6:] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                ubx_e_curr[6:] = [ 1.0,  1.0,  1.0, 1.0,  1.0,  1.0]
                
                controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)


                print(f"\n🔄 FINE RECOVERY: Missione ripristinata verso WP {target_idx + 1}. Vincoli di sicurezza riattivati.")
        else:
            x_ref_attuale = waypoints[target_idx]

        # 1. LiDAR 3D
        hits, radii = get_lidar_hits_3d(current_x[0], current_x[1], current_x[2], ostacoli, num_rays=1000, max_range=1.5)
        #hits, radii = get_lidar_hits_3d_qualsiasi(current_x[:3], ostacoli, num_rays=1000, max_range=1.5)
        Q_rel = hits.copy()
        if len(hits) > 0:
            Q_rel[:, 0] -= current_x[0]
            Q_rel[:, 1] -= current_x[1]
            Q_rel[:, 2] -= current_x[2]
        
        target_rel_x = x_ref_attuale[0] - current_x[0]
        target_rel_y = x_ref_attuale[1] - current_x[1]
        target_rel_z = x_ref_attuale[2] - current_x[2]

        xMin_r, xMax_r, yMin_r, yMax_r, zMin_r, zMax_r, _ = min_cube_select_3d(
            Q_rel, radii, target_rel_x, target_rel_y, target_rel_z, drone_radius=0.1, lidar_ray=1.5
        )
        box_abs = np.array([
            xMin_r + current_x[0], xMax_r + current_x[0], 
            yMin_r + current_x[1], yMax_r + current_x[1], 
            zMin_r + current_x[2], zMax_r + current_x[2]
        ])

        box_history.append(box_abs.copy())


        # ==========================================
        # BLOCCO DI DIAGNOSTICA (MPC DEBUGGER)
        # ==========================================
        if t % 10 == 0: # Stampo ogni 10 passi
            print(f"\n--- DEBUG PASSO {t} ---")
            print(f"1. Posizione Drone : X={current_x[0]:.2f}, Y={current_x[1]:.2f}, Z={current_x[2]:.2f}")
            print(f"2. Box Verde (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Y in [{box_abs[2]:.2f}, {box_abs[3]:.2f}] | Z in [{box_abs[4]:.2f}, {box_abs[5]:.2f}]")
            print(f"3. Target Locale  : X={x_ref_attuale[0]:.2f}, Y={x_ref_attuale[1]:.2f}, Z={x_ref_attuale[2]:.2f}")
            

            
            # Calcolo la distanza tra drone e target locale
            dist_to_local = np.linalg.norm(current_x[:3] - np.array([target_rel_x, target_rel_y, target_rel_z])[:3])
            print(f"5. Distanza da percorrere nel box: {dist_to_local:.3f} metri")
        # ==========================================

        # 2. SOLVE MPC
        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

        if status in [3, 4]:
             # ==========================================
            # PIANO A: Reset della memoria (Hovering Guess)
            # ==========================================
            controller.ocp_solver.reset()
            controller.x_guess = np.tile(current_x, (controller.N, 1))
            controller.u_guess = np.full((controller.N, model.nu), u_hover)


            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
            
                        
            # ==========================================
            # PIANO B: Historical Warm-Start 3D
            # ==========================================
            if status in [3, 4] and not in_recovery and len(u_history) > 0:
                print(f"\n⚠️ PIANO A FALLITO. Avvio PIANO B (Ricerca a ritroso nei controlli passati 3D)...")
                
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
            # PIANO C: Ritiro al Centro con Rilassamento Alpha
            # ==========================================
            if status in [3, 4] and not in_recovery:
                print(f"\n⚠️ PIANO B FALLITO. Avvio PIANO C (Ritiro al Centro con Rilassamento Alpha)...")

                # controller.resetHorizon(params.N_naive)

                # # 2. Ridimensioniamo le memorie per non far crashare il solve_step
                # controller.x_guess = np.tile(current_x, (controller.N, 1))
                # controller.u_guess = np.full((controller.N, model.nu), u_hover)
                
                passi_indietro = 5 # Numero di passi indietro da cui prendere il box sicuro
                if len(box_history) > passi_indietro:
                    box_sicuro = box_history[-passi_indietro]
                else:
                    box_sicuro = box_history[0]  # Se fallisce subito, torna alla partenza

                # Calcoliamo il centro geometrico del box tridimensionale
                center_x = (box_sicuro[0] + box_sicuro[1]) / 2.0
                center_y = (box_sicuro[2] + box_sicuro[3]) / 2.0
                center_z = (box_sicuro[4] + box_sicuro[5]) / 2.0
                

                # Inizializziamo il target di ripiego a 12 stati
                target_recovery = np.zeros(12)
                target_recovery[0] = center_x
                target_recovery[1] = center_y
                target_recovery[2] = center_z
                x_ref_attuale = target_recovery
                
                in_recovery = True
                timer_recovery = 80 
                
                # ==========================================
                # RILASSAMENTO MATEMATICO DEL VINCOLO (Realistico)
                # ==========================================
                controller.ocp_solver.constraints_set(controller.N, "lh", np.full(6, -alpha_curr))
               
                lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                
                lbx_e_curr[6:] = [-10.0, -10.0, -10.0,-10.0, -10.0, -10.0]
                ubx_e_curr[6:] = [ 10.0,  10.0,  10.0, 10.0,  10.0,  10.0]
               
                controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                
                x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                
                if status in [3, 4]:
                    print("❌ PIANO C FALLITO. Lo spazio fisico è del tutto inesistente. Chiusura.")
                    break
                else:
                    print("✅ PIANO C Riuscito! Il drone indietreggia verso il centro ignorando temporaneamente la rete.")

        if x_sol is None or status in [3, 4]:
            print("\\n❌ FALLIMENTO CRITICO: Il solutore è collassato (probabile uscita dal Safe-Box). Interruzione sicura.")
            break

        if u_sol is not None:
            u_history.append(u_sol.copy())
        
        current_x = x_sol[1]
        x_history.append(current_x)

        mode_history.append('recovery' if in_recovery else 'normal')


        # ==========================================
        # GESTIONE DELLO STALLO (Local Minimum Escape 3D)
        # ==========================================
        if t > 0 and not in_recovery:
            # Calcoliamo lo spostamento nello spazio 3D (X, Y, Z) rispetto al passo prima
            spostamento = np.linalg.norm(current_x[:3] - x_history[-2][:3])
            
            if spostamento < 0.01:  # Meno di 1 cm
                contatore_stallo += 1
            else:
                contatore_stallo = 0  
                
            if contatore_stallo >= MAX_STALLO_ITER:
                print(f"\n⚠️ STALLO RILEVATO (Passo {t})! Il drone è intrappolato in un minimo locale 3D.")
                print("   -> Perturbo il target locale verso l'alto di 20 cm...")
                
                ghost_waypoints.append(waypoints[target_idx].copy())
                
                # ATTENZIONE: Nel 3D la quota Z è all'indice 2!
                waypoints[target_idx][2] += 0.20 
                x_ref_attuale = waypoints[target_idx] 
                
                contatore_stallo = 0 
        # ==========================================

        # ==========================================
        # 3. LOGICA DI SWITCH DEL TARGET (3D: X, Y, Z)
        # ==========================================
        # CALCOLIAMO LA DISTANZA DAL WAYPOINT REALE (non da x_ref_attuale che potrebbe essere il centro del box!)
        dist_al_target_reale = np.linalg.norm(current_x[:3] - waypoints[target_idx][:3])

        if dist_al_target_reale < TOLLERANZA_WAYPOINT:
            if target_idx < len(waypoints) - 1:
                target_idx += 1
                print(f"\n✅ Waypoint raggiunto! Passaggio al Target {target_idx + 1}")
            else:
                print(f"\n🎯 MISSIONE COMPLETATA! Ultimo target raggiunto al passo {t}.")
                break 
        # ==========================================

    # --- AVVIA ANIMAZIONE VIDEO ---
    crea_animazione_3d(x_history, box_history, ostacoli, waypoints)

if __name__ == '__main__':
    main()