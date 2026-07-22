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


# sui primi 6 targets
# con N=10 vede troppi errori e entra in loop e non avanza, anche con time recovery troppo alti ex 60
# con N=15, raggiolidar=1.5, -alpha_curr e +-10 x rilassamento vincoli, passi indietro 3, W=0.5, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =20 ==> successi =3, timout =1, fallimenti =2
# aumento passi indietro e riduco recovery time, stringo rilassamento velocità
# con N=15, raggiolidar=1.5, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=0.5, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =3, timout =1, fallimenti =2
# riduco raggio lidar
# con N=15, raggiolidar=1.0, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=0.5, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =4, timout =0, fallimenti =2
# aumento peso allungamento box W
# con N=15, raggiolidar=1.0, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =4, timout =0, fallimenti =2
# non cambia niente, lo lascio alto e aumento range velocità (con W alto fallisce nei punti al centro)
# con N=15, raggiolidar=1.0, -alpha_curr e +-10 x rilassamento vincoli, passi indietro 10, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =4, timout =0, fallimenti =2
# non cambia niente, rimetto range velocità stringente e aumento passi indietro
# con N=15, raggiolidar=1.0, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 20, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =4, timout =0, fallimenti =2
# non cambia niente, rimetto passi indietro a 10 e allargo time recovery
# con N=15, raggiolidar=1.0, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =60 ==> successi =4, timout =0, fallimenti =2
# non cambia proprio niente, rimetto time recovery come prima e allargo orizzonte
# con N=20, raggiolidar=1.0, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =6, timout =0, fallimenti =0
# sta andando meglio, provo a riallargare raggio lidar
# con N=20, raggiolidar=1.5, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =6, timout =0, fallimenti =0
# va anche così, provo con tutti i target

# su tutti i targets
# con N=20, raggiolidar=1.5, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=15.0, itermax=50, terget update=+20, spostamento<0.01, sim time=40, time recovery =40 ==> successi =9, timout =4, fallimenti =0

# RISULTATI PER STATISTICA (13 targets)
# N=10 e raggio 1.5 con NN: successi = 8, timeout = 2, schianti = 3 (recovery totali 120)
# N=15 e raggio 1.5 con NN: successi = 7, timeout = 3, schianti = 3 (recovery totali 59)
# N=20 e raggio 1.5 con NN: successi = 9, timeout = 4, schianti = 0 (recovery totali 73)
# N=20 e raggio 1.0 con NN: successi = 10, timeout = 3, schianti = 0 (recovery totali 12)
# N=20 e raggio 2.0 con NN: successi = 5, timeout = 2, schianti = 6 (recovery totali 34)
# N=20 e raggio 2.5 con NN: successi = 3, timeout = 3, schianti = 7 (recovery totali 68)
# N=20 e raggio 3.0 con NN: successi = 3, timeout = 4, schianti = 6 (recovery totali 74)
# N=30 e raggio 1.5 con NN: successi = 6, timeout = 7, schianti = 0 (recovery totali 119)



# N=10 e raggio 1.5 senza NN: successi = 1, timeout = 0, schianti = 12 (recovery totali 16)
# N=15 e raggio 1.5 senza NN: successi = 5 (3), timeout = 0, schianti = 8 (10) (recovery totali 9 )
# N=20 e raggio 1.5 senza NN: successi = 7 (4), timeout = 2, schianti = 4 (7) (recovery totali 6)
# N=20 e raggio 1.0 senza NN: successi = 8 (5), timeout = 4, schianti = 1 (4) (recovery totali 1 )
# N=20 e raggio 2.0 senza NN: successi = 6 (2), timeout = 0, schianti = 7 (11) (recovery totali 12)
# N=20 e raggio 2.5 senza NN: successi = 5 (2), timeout = 0, schianti = 8 (11) (recovery  totali 12)
# N=20 e raggio 3.0 senza NN: successi = 5 (2), timeout = 0, schianti = 8 (11) (recovery totali 13)
# N=30 e raggio 1.5 senza NN: successi = 4, timeout = 9, schianti = 0 (recovery totali 0) #non passa più attraverso con orizzonti più lunghi



# RISULTATI CON PUSH BONUS e MIN CUBE SEL 2D W=15 (box deciso con target) =case1
# N=20 e raggio 1.5 con NN: successi = 9, timeout = 4, schianti = 0 (recovery totali 73)
# RISULTATI CON PUSH BONUS e MIN CUBE SEL 2D W=50 (box deciso con x_current=velocità attuali)= case5
# N=20 e raggio 1.5 con NN: successi = 5, timeout = 1, schianti = 7 (recovery totali 22)
# RISULTATI CON PUSH BONUS e MIN CUBE SEL DIRECTIONAL W=0 -> solo area con expando directional e expand generico (box deciso con x_sol_precedente[5][0] - current_x[0])=case 3 4
# N=20 e raggio 1.5 con NN: successi = 3, timeout = 0, schianti = 10 (recovery totali 63)
# RISULTATI CON PUSH BONUS e MIN CUBE SEL 2D W=50 (box deciso con x_sol_precedente[5][0] - current_x[0])=case6
# N=20 e raggio 1.5 con NN: successi = 11, timeout = 2, schianti = 0 (recovery totali 30)
# RISULTATI CON PUSH NOT BLOCKED FACES e MIN CUBE SEL DIRECTIONAL W=0 -> solo area (box deciso con x_sol_precedente[5][0] - current_x[0])=case2
# N=20 e raggio 1.5 con NN: successi = 1, timeout = 0, schianti = 12 (recovery totali 49)
# RISULTATI CON PUSH NOT BLOCKED FACES e MIN CUBE SEL DIRECTIONAL W=0 -> solo area (box deciso con x_sol_precedente[2][0] - current_x[0])
# N=20 e raggio 1.5 con NN: successi = 1, timeout = 0, schianti = 12 (recovery totali 15)
# RISULTATI CON PUSH NOT BLOCKED FACES e MIN CUBE SEL DIRECTIONAL W=0 -> solo area (box deciso con x_sol_precedente[20][0] - current_x[0])
# N=20 e raggio 1.5 con NN: successi = , timeout = , schianti = (recovery totali ) #da 100% successi ma attraversa gli ostacoli in realtàs



# RISULTATI CON PUSH BONUS e MIN CUBE SEL 2D W=50 (box deciso con x_sol_precedente[5][0] - current_x[0])
# N=20 e raggio 1.5 con NN: successi = 11, timeout = 2, schianti = 0 (recovery totali 30)
# RISULTATI CON PUSH BONUS e MIN CUBE SEL 2D W=50 (box deciso con x_sol_precedente[5][0] - current_x[0])
# N=20 e raggio 3.0 con NN: successi = 5, timeout = 3, schianti = 5 (recovery totali 124)

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
        # np.array([4.8, -2.0, 0.0, 0.0, 0.0, 0.0]),
        # np.array([6.2,  3.8, 0.0, 0.0, 0.0, 0.0]),
        # np.array([7.6,  0.5, 0.0, 0.0, 0.0, 0.0]),
        # np.array([10.2,-2.1, 0.0, 0.0, 0.0, 0.0]),
        # np.array([11.0, 0.7, 0.0, 0.0, 0.0, 0.0]),
        # np.array([11.2, 4.0, 0.0, 0.0, 0.0, 0.0]),
        # np.array([15.1, 3.8, 0.0, 0.0, 0.0, 0.0]),
        # np.array([14.8, 0.8, 0.0, 0.0, 0.0, 0.0]),
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
        # SETUP VIDEO REAL-TIME & SALVATAGGIO
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

        # AGGIUNGI QUESTA LISTA QUI: conterrà i fotogrammi del video
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
                

                # # If vectors are too small (e.g. perfect hovering), direct movement toward the target
                # if abs(dx) < 0.05 and abs(dz) < 0.05:#0.01
                #     dx, dz = target_rel_x, target_rel_z

            else:
                # Al primissimo passo (t=0)
                dx = target_rel_x
                dz = target_rel_z


        

            # # WARM START SAFE BOX
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
            # WARM START SAFE BOX (MODIFICATO)
            # ==========================================
            if x_sol_prev is not None:
                # 1. Calcola il bounding box (min e max) dell'INTERA traiettoria predetta al passo precedente
                traj_xmin = np.min(x_sol_prev[:, 0])
                traj_xmax = np.max(x_sol_prev[:, 0])
                traj_zmin = np.min(x_sol_prev[:, 1])
                traj_zmax = np.max(x_sol_prev[:, 1])
                
                # 2. Converti questo Bounding Box in coordinate RELATIVE rispetto al drone attuale (current_x)
                box_prev_rel = [
                    traj_xmin - current_x[0]-0.15,
                    traj_xmax - current_x[0]+0.15,
                    traj_zmin - current_x[1]-0.15,
                    traj_zmax - current_x[1]+0.15
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


#             ####################################################################################################################################################################################################################################################################################################################################
# # provare con 0.01-0.05 nei due parametri anche espansione score e direzionale del mode diractional poi vedere se case 6 e ws restano buoni, W=50
# # con 0.01 in entrambi, il case 3 (general) da 6 succ 7 fall, il case 3 (score) da 6 succ 6 fall 1 time, il case 4 (directional) da 4 succ 6 time 3 fall???

# # togliendo check vett piccoli e usando 0.1 nel lidar con W=50, il case 3 (general) da 6 succ 7 fall, il case 3 (score) da 6 succ 1 time 6 fall, il case 4 (directional) da 3 succ 7 time e 3 fall 
# # case 6 da 11 succ e 2 time, case ws score da 3 succ 10 time, case ws con directional da 2 succ 11 time
# 
# con W=0: score da 6 succ e 7 fall, general da 5 succ 8 fall
# con W=100: score da 7 succ 1 time 5 fall, general da 6 succ e 7 fall *******
# con W200: score da 6 succ 1 time 6 fall

##########################
# CASO MIGLIORE SAREBBE SCORE CON REL=0.1, NO ELSIF, W=100 CON 7SUCC 1TIME 5FALL, IN QUESTA MODALITà HO RISULTATI GENERAL 6SUCC 7FALL MA NON DIRECTIONAL (DA AGGIUNGERE EVENTUALMENTE); SE MANTENGO W=50 CASE GENERAL è UGUALE E SCORE DA 6SUCC 1TIME E 6FALL, DIRECTIONAL DA 3SUCC 7TIME 3FALL
##########################
# 
# resto con W=50 , rel=0.1 e senza elsif, provo best case con orizzonti diversi e raggio lidar = 1.5
#
# su 13 targets:
# con lidar 1.5
# case 6 NN con N=10: 4  succ 6 time 3 fall
# case 6 NN con N=15: 5  succ 2 time 6 fall
# case 6 NN con N=20: 11 succ 2 time 0 fall (anche con rel=0.01)
# case 6 NN con N=25: 10 succ 3 time 0 fall
# case 6 NN con N=30: 7  succ 6 time 0 fall (DA FINIRE, mancano 4 video)

# con lidar 2.0
# case 6 NN con N=10: 4  succ 5 time 4 fall
# case 6 NN con N=20: 11 succ 2 time 0 fall
# case 6 NN con N=30: 6  succ 5 time 2 fall

# con lidar 3.0
# case 6 NN con N=10: 4 succ 7 time 2 fall
# case 6 NN con N=15: 4 succ 1 time 8 fall
# case 6 NN con N=20: 5 succ 3 time 5 fall*** (5,1,7 con vett piccoli)
# case 6 NN con N=25: 3 succ 4 time 6 fall
# case 6 NN con N=30: 3 succ 9 time 1 fall

# con lidar 4.5
# case 6 NN con N=10: 3 succ 7 time 3 fall, 3,4,2 to add + 0,3,1???, 2,6,4 prima
# case 6 NN con N=15: 3 succ 9 time 1 fall
# case 6 NN con N=20: 6 succ 4 time 3 fall
# case 6 NN con N=25: 6 succ 1 time 6 fall
# case 6 NN con N=30: 6 succ 4 time 3 fall

# **** ---> BEST CASE: N=20 LD=1.5
# NAIVE: case 6 naive con N=20 e lidar 1.5: 3 succ 10 fall 0 tim
# NN 100 test: case 6 NN con N=20 e lidar 1.5: DA FARE ***

##############################################################################################################################################################################################
# ws modificato (DA PROVARE)  (aggiunta tolleranza E AGGIUNTA LOGICA PER ESPENSIONE X Z!!!)


# espansione directional DA RIFARE PER LOGICA X Z (non sembra cambi niente con raggio lidar 1.5 per ora ho provato con N10 e è uguale)
#LD 1.5
#N10: 12 succ 1 time 0 fall RIFATTO, new da 12,1,0
#N15:                              , new da 8,0,0
#N20: 9  succ 4 time 0 fall RIFATTO, new da 11,2,0
#N30: 8  succ 5 time 0 fall RIFATTO, new da 7,6,0

#LD 3.0
#N10: 11 succ 1 time 1 fall  RIFATTO, new da 13,0,0
#N20: 9 succ 0 time 4 fall  RIFATTO, new da 11,2,0
#N30: 9 succ 4 time 0 fall RIFATTO, new da 8,4,1 (7 fallisce ma è colpa dell'angolo, facciamo finta di niente! ---> 9,4,0)
# SOSTITUIRE VIDEO DEL 7
# HO PROVATO A AUMENTARE E ABBASSARE RISOLUZIONE PER EVITARE GLITCH, NON CAMBIA UN CAZZO!

#LD 4.5
#N10: 7 succ 4 time 2 fall rifatto, new da 10, 3, 0
#N15: 1 succ 0 time 12 fall rifatto ????
#N20: 5 succ 0 time 8 fall rifatto, new da 10,2,1 
#N25: 
#N30: 7 succ 4 time 2 fall RIFATTO, new da 7,4,2

#*****************************************

# ######################################################################################################################################

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


            # ==========================================
            # VIDEO FRAME UPDATE
            # ==========================================
            # Aggiorna la linea della traiettoria
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
            
            ax_anim.set_title(f"Target {test_idx +9} | Step: {t} | Status MPC: {status}")
            
            # Draw the frame on the screen
            fig_anim.canvas.draw()
            fig_anim.canvas.flush_events()
            
            plt.pause(0.1) 

            # AGGIUNGI QUESTE RIGHE QUI PER CATTURARE IL FRAME:
            # Converte il disegno corrente a schermo in un'immagine da salvare nel video
            image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
            frames_animazione.append(image)
            # ==========================================
            # ==========================================


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
                    
                    Z_MAX_SICURA = roof - 0.2  # Margine di sicurezza sotto il soffitto
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





        # ==========================================
        # END OF VIDEO
        # ==========================================
        # ==========================================
        # ESPORTAZIONE E SALVATAGGIO DEL VIDEO
        # ==========================================
        plt.ioff()
        plt.close(fig_anim) # Chiude la finestra video real-time per liberare memoria

        if len(frames_animazione) > 0:
            cartella_video = "video_traiettorie"
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
            nome_video = os.path.join(cartella_video, f"Video_Target_N15_{test_idx+9:02d}_{esito}.mp4")
            
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

