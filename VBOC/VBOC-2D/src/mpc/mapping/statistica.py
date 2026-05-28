import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController
from test_lidar import min_cube_select_2d, get_lidar_hits_2d_qualsiasi 


# con N=15, raggiolidar=1.5, -alpha_curr e +-5 x rilassamento vincoli, passi indietro 10, W=0.5, itermax=50, terget update=+20, spostamento<0.01 ==> successi =23, timout =0, fallimenti =2

def genera_ambiente_2d_test():
    """Nuova mappa basata sullo schizzo con ostacoli blu e target verdi."""
    poligoni = [
        # Basi fisse (Pavimento, Soffitto, Muretto finale)
        [[-2.0, -4.0], [25.0, -4.0], [25.0, -5.0], [-2.0, -5.0]], 
        [[-2.0,  5.0], [25.0,  5.0], [25.0,  6.0], [-2.0,  6.0]], 
        
        # Ostacoli Grigi (mantenuti)
        [[3.0, 1.0], [5.0, 3.0], [6.0, 1.0], [5.0, 0.0]],         # Rombo SX
        [[7.0, -3.0], [9.0, -3.0], [9.0, -0.5], [7.0, -0.5]],     # Quadrato Basso
        
        # NUOVI OSTACOLI BLU
        [[7.8, 3.9], [10.0, 4.1], [9.6, 0.8]],                     # Triangolo alto SX
        [[11.0, -0.4], [13.9, -0.6], [14.3, -1.7], [11.9, -2.8]], # Rettangolo obliquo basso
        [[12.0, 2.5], [12.6, 3.3], [13.6, 3.3], [14.1, 2.4], 
         [14.1, 1.2], [13.4, 0.9], [12.4, 1.0]],                  # Esagono centrale
        [[16.3, 4.1], [19.3, 4.2], [19.3, 2.5], [16.3, 2.5]],     # Rettangolo alto DX
        [[15.0, -1.0], [18.0, -2.0], [19.0, 0.0], [16.0, 1.0]],   # Rettangolo basso dx
    ]
    
    # Muretto finale a DX
    segments = []
    for poli in poligoni:
        n = len(poli)
        for i in range(n):
            segments.append([poli[i], poli[(i + 1) % n]])
    segments.append([[21.0, -4.0], [21.0, -2.0]])
            
    # I PUNTINI VERDI (Target Testati estratti dall'immagine)
    targets = [
        np.array([4.8, -2.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([6.2,  3.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([7.6,  0.5, 0.0, 0.0, 0.0, 0.0]),
        np.array([10.2,-2.1, 0.0, 0.0, 0.0, 0.0]),
        np.array([11.0, 0.7, 0.0, 0.0, 0.0, 0.0]),
        np.array([11.2, 4.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([15.1, 3.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([14.8, 0.8, 0.0, 0.0, 0.0, 0.0]),
        np.array([15.5,-2.6, 0.0, 0.0, 0.0, 0.0]),
        np.array([18.2, 1.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([19.6,-2.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([20.1, 3.4, 0.0, 0.0, 0.0, 0.0]),
        np.array([22.5, 0.4, 0.0, 0.0, 0.0, 0.0])
    ]
    
    
    # Lista dei 5 target (le X rosse nello schizzo)
    # targets = [
    #     np.array([7.0,  2.0, 0.0, 0.0, 0.0, 0.0]),  # Dietro il rombo
    #     np.array([10.0, -2.5, 0.0, 0.0, 0.0, 0.0]), # Sotto al centro
    #     np.array([14.0,  1.0, 0.0, 0.0, 0.0, 0.0]), # A dx del triangolo
    #     np.array([21.0,  2.5, 0.0, 0.0, 0.0, 0.0]), # In alto a dx
    #     np.array([22.0, -1.5, 0.0, 0.0, 0.0, 0.0])  # Oltre il muretto
    # ]

    # # Fascia 1: Subito dopo il primo rombo (X = 7.5)
    # for z in np.linspace(-2.5, 4.0, 5):
    #     targets.append(np.array([7.5, z, 0.0, 0.0, 0.0, 0.0]))
        
    # # Fascia 2: Centro mappa (X = 14.0)
    # for z in np.linspace(-2.5, 4.5, 5):
    #     targets.append(np.array([14.0, z, 0.0, 0.0, 0.0, 0.0]))
        
    # # Fascia 3: Tra la stalattite e il muro finale (X = 19.5)
    # for z in np.linspace(-1.5, 4.0, 5):
    #     targets.append(np.array([19.5, z, 0.0, 0.0, 0.0, 0.0]))
        
    # # Fascia 4: Oltre il muro finale (X = 23.5)
    # for z in np.linspace(-3.0, 4.0, 5):
    #     targets.append(np.array([23.5, z, 0.0, 0.0, 0.0, 0.0]))
    return poligoni, segments, targets

def main_statistico():
    
    risultati = {"Successi": 0, "Timeout": 0, "Schianti": 0}
    traiettorie_riuscita = []
    totale_recovery_attivate = 0

    params = Parameters("sth") 
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)

    DT = params.dt
    SIM_TIME = 40.0 # Tempo aumentato per coprire tutti i target
    N_SIM = int(SIM_TIME / DT)
    
    poligoni, segmenti, targets = genera_ambiente_2d_test()
    N_TESTS = len(targets) # Ora fa solo 5 test, uno per target!

    print(f"--- AVVIO TEST DI COPERTURA: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):
        # Punto di partenza fisso (il pallino blu nello schizzo)
        current_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        x_history = [current_x.copy()]
        u_history = []
        box_history = []
        
        recovery_in_questo_test = 0
        esito = "Timeout" 

        # Variabili di stato di labirinto.py
        contatore_stallo = 0
        MAX_STALLO_ITER = 50
        in_recovery = False

        timer_recovery = 0
        target_recovery = None
        current_target = target_base.copy()
        x_ref_attuale = current_target.copy()

        controller.ocp_solver.reset()
        controller.x_guess = np.tile(current_x, (controller.N, 1))
        controller.u_guess = np.full((controller.N, model.nu), u_hover)

        

        for t in range(N_SIM):
            # 0. Gestione Timer Recovery
            if in_recovery:
                x_ref_attuale = target_recovery
                timer_recovery -= 1
                if timer_recovery <= 0:
                    in_recovery = False
                    
                    # Ripristino Vincoli (Logica originale labirinto.py)
                    for i in range(1, controller.N + 1):
                        controller.ocp_solver.constraints_set(i, "lh", np.zeros(4))
                    
                    lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                    ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                    lbx_e_curr[3:] = [-1.0, -1.0, -1.0]
                    ubx_e_curr[3:] = [ 1.0,  1.0,  1.0]
                    controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                    controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                    
                    x_ref_attuale = target_base.copy()
            else:
                # Ripristina target base se è stato perturbato dallo stallo in precedenza
                if contatore_stallo == 0:
                    x_ref_attuale = current_target.copy()

            # 1. LiDAR e Safe-Box
            hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=1.0)
            
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                
            target_rel_x = x_ref_attuale[0] - current_x[0]
            target_rel_z = x_ref_attuale[1] - current_x[1]

            xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_2d(
                Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1
            )
            
            box_abs = np.array([
                xMin_r + current_x[0], xMax_r + current_x[0], 
                zMin_r + current_x[1], zMax_r + current_x[1]
            ])
            box_history.append(box_abs.copy())
            
            # 2. MPC Solve
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

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


            # ==========================================
            # GESTIONE INFEASIBILITY (Esatta logica di labirinto.py)
            # ==========================================
            if status in [3, 4]:
                recovery_in_questo_test += 1
                totale_recovery_attivate += 1
                if alpha_curr is None: alpha_curr = 0.1
                
                # PIANO A: Reset memoria
                controller.ocp_solver.reset()
                controller.x_guess = np.tile(current_x, (controller.N, 1))
                controller.u_guess = np.full((controller.N, model.nu), u_hover)
                x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

                # PIANO B: Historical Warm-Start
                if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                    for i in range(len(u_history) - 1, -1, -1):
                        past_u = u_history[i]
                        controller.ocp_solver.reset()
                        controller.x_guess = np.tile(current_x, (controller.N, 1))
                        controller.u_guess = past_u
                        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                        if status in [0, 2]:
                            break

                # PIANO C: Ritiro al Centro
                if (status in [3, 4]) and not in_recovery:
                    print("Piano C avviato")
                    passi_indietro = 5
                    if len(box_history) > passi_indietro:
                        box_sicuro = box_history[-passi_indietro]
                    else:
                        box_sicuro = box_history[0]
                    
                    center_x = (box_sicuro[0] + box_sicuro[1]) / 2.0
                    center_z = (box_sicuro[2] + box_sicuro[3]) / 2.0
                    
                    target_recovery = np.array([center_x, center_z, 0.0, 0.0, 0.0, 0.0])
                    x_ref_attuale = target_recovery
                    
                    in_recovery = True
                    timer_recovery = 60 
                    
                    # Rilassamento Modificato come richiesto
                    controller.ocp_solver.constraints_set(controller.N, "lh", np.full(4, -alpha_curr))
                   
                    lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                    ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                    lbx_e_curr[3:] = [-10.0, -10.0, -10.0]
                    ubx_e_curr[3:] = [ 10.0,  10.0,  10.0]
                    controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                    controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                    
                    x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                    
                    if status in [3, 4]:
                        esito = "Schianti"
                        break
                
                elif status in [3, 4] and in_recovery:
                    esito = "Schianti"
                    break

            if u_sol is not None:
                u_history.append(u_sol.copy())

            current_x = x_sol[1]
            x_history.append(current_x.copy())

            # # ==========================================
            # # GESTIONE DELLO STALLO
            # # ==========================================
            # if t > 0:
            #     spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            #     if spostamento < 0.01:
            #         contatore_stallo += 1
            #     else:
            #         contatore_stallo = 0 

            #     if contatore_stallo >= MAX_STALLO_ITER:
            #         print(f"\n⚠️ STALLO RILEVATO (Passo {t})! Il drone è intrappolato in un minimo locale.")
            #         print("   -> Perturbo il target locale verso l'alto")
                    
            #         current_target[1] += 0.20 
            #         x_ref_attuale = current_target.copy()
            #         contatore_stallo = 0 
            
            # ==========================================
            # GESTIONE DELLO STALLO E RITORNO AL TARGET BASE
            # ==========================================
            
            # 1. CONTROLLO VITTORIA ASSOLUTA (Chiude il test con Successo)
            dist_target_reale = np.linalg.norm(current_x[:2] - target_base[:2])
            if dist_target_reale < 0.3:
                esito = "Successi"
                break
                            
            if t > 0:
                # 2. GESTIONE DELLO STALLO NORMALE
                spostamento = np.linalg.norm(current_x[:2] - x_history[-2][:2])
                if spostamento < 0.01:
                    contatore_stallo += 1
                else:
                    contatore_stallo = 0 

                if contatore_stallo >= MAX_STALLO_ITER:
                    print(f"\n⚠️ STALLO RILEVATO (Passo {t})! Perturbo il target verso l'alto.")
                    # Alza il target di 0.25, ma NON OLTRE il soffitto (Z=4.5)
                    current_target[1] = min(current_target[1] + 0.25, 4.5)
                    x_ref_attuale = current_target.copy()
                    contatore_stallo = 0

                # 3. CONTROLLO ARRIVO AL TARGET LOCALE (Sopra la meta)
                dist_al_locale = np.linalg.norm(current_x[:2] - current_target[:2])
                if dist_al_locale < 0.3:
                    if not np.array_equal(current_target, target_base):
                        print(f"\n✅ Target rialzato raggiunto. Scendo verso il Target Finale a terra.")
                        current_target = target_base.copy()
                        x_ref_attuale = current_target.copy()
                        contatore_stallo = 0


                 # 2. CONTROLLO ARRIVO AL TARGET LOCALE (Ghost)
                dist_al_locale = np.linalg.norm(current_x[:2] - current_target[:2])
                if dist_al_locale < 0.3:
                    # Il drone ha raggiunto la quota alta per scavalcare l'ostacolo.
                    # Ora resetta il target a quello originale per forzare la discesa.
                    if not np.array_equal(current_target, target_base):
                        print(f"\n✅ Target di stallo superato. Ripunto al Target Finale a terra.")
                        current_target = target_base.copy()
                        x_ref_attuale = current_target.copy()
                    contatore_stallo = 0

            # Controllo Vittoria
            dist_target = np.linalg.norm(current_x[:2] - target_base[:2])
            if dist_target < 0.3:
                esito = "Successi"
                break

        risultati[esito] += 1
        if esito == "Successi" :
            traiettorie_riuscita.append(np.array(x_history))
            
        print(f"Test {test_idx+1}/{N_TESTS} -> Esito: {esito} (Recovery usate: {recovery_in_questo_test})")

        print(f"\n--- RISULTATI FINALI ---")
        print(f"Successi: {risultati['Successi']} | Timeout: {risultati['Timeout']} | Schianti: {risultati['Schianti']}")

    # ==========================================
    # PLOT STATISTICI
    # ==========================================
    
    # # CORREZIONE 3: Plot individuale per ogni singolo target per analizzare i box verdi
    # plt.figure(figsize=(10, 6))
    # for poli in poligoni:
    #     plt.gca().add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        
    # # Plot dei box verdi campionati (uno ogni 8 passi per non oscurare il grafico)
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
    # PLOT STATISTICO FINALE (REACHABILITY MAP)
    # ==========================================
    print(f"\n--- RISULTATI FINALI ---")
    print(f"Successi: {risultati['Successi']} | Timeout: {risultati['Timeout']} | Schianti: {risultati['Schianti']}")
    print(f"Recovery totali innescate: {totale_recovery_attivate}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 1. GRAFICO A TORTA
    labels = list(risultati.keys())
    sizes = list(risultati.values())
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0) 

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax1.set_title('Tasso di Successo (20 Target)')

    # 2. REACHABILITY MAP (Mappa di Copertura)
    for poli in poligoni:
        ax2.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))

    # Tracciamo TUTTE le traiettorie con alpha=0.7 per far vedere l'addensamento dei percorsi
    for i, traj in enumerate(traiettorie_riuscita):
        ax2.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=1.5, alpha=0.7)
    
    # Tracciamo TUTTI i target testati
    for tb in targets:
        ax2.scatter(tb[0], tb[1], color='red', marker='X', s=100, zorder=5)
        
    ax2.scatter(0.0, 0.0, color='blue', s=120, zorder=6, label='Start')
    
    # Linee invisibili per una legenda pulita
    ax2.plot([], [], color='cyan', linewidth=1.5, label='Traiettorie di Volo')
    ax2.scatter([], [], color='red', marker='X', s=100, label='Target Testati')

    ax2.set_xlim(-2, 25)
    ax2.set_ylim(-6, 6)
    ax2.set_title('Reachability Map (Copertura Spaziale)')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_statistico()

