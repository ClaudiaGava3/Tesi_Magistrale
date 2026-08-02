import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os

# Importiamo le funzioni dell'ambiente dal tuo script
from parser import Parameters
from statistica_2 import genera_ambienti_random
from lidar import get_lidar_hits_2d_qualsiasi

# ==========================================
# 1. PARAMETRI FISICI E DI TUNING APF
# ==========================================
MASS = 0.75          # kg
G = 9.81             # m/s^2
CF = 1.5e-3          # Coefficiente di spinta
DT = 0.02            # Time step
U_HOVER = (MASS * G) / (2.0 * CF)

# Pesi della funzione di costo (Replicati da MPC)
Q_COST = np.diag([0.0001, 0.0001, 20.0, 100.0, 1.0, 1.0])
R_COST = np.diag([0.0001, 0.0001])

# ---> PARAMETRI DA TARARE PER L'APF <---
K_ATT_BASE = 1.5     # Moltiplicatore base per la forza attrattiva costante 
K_BRAKE = 3.0        # Quanto frena forte se supera la velocità target
K_REP = 0.5          # Guadagno repulsivo per ogni singolo punto LiDAR
RHO_0 = 1.5          # Distanza limite in cui il punto LiDAR inizia a respingere
DRONE_RADIUS = 0.1  # Raggio del drone per check schianto

def compute_apf_forces(current_x, x_ref, hits):
    pos_drone = current_x[0:2]
    v_curr_x = current_x[3]
    v_curr_z = current_x[4]  # Serve per lo smorzatore in Z
    v_target_x = x_ref[3]
    
    # 1. FORZA ATTRATTIVA (Metodo del Prof)
    # Spinge se v_curr_x < v_target, frena se v_curr_x > v_target.
    # Quando v_curr_x == v_target, la forza è ZERO e il drone prosegue per INERZIA.
    f_att_x = K_ATT_BASE * (v_target_x - v_curr_x)
    
    # Freno in Z: se sbatte contro un ostacolo, l'APF lo rimette dritto a Vz=0
    f_att_z = 2.0 * (0.0 - v_curr_z)
        
    F_att = np.array([f_att_x, f_att_z])
    
    # 2. FORZA REPULSIVA (Somma dei contributi di OGNI punto LiDAR)
    F_rep = np.array([0.0, 0.0])
    
    if len(hits) > 0:
        for hit in hits:
            dist_vec = pos_drone - hit 
            rho = np.linalg.norm(dist_vec)
            
            if 0.01 < rho <= RHO_0:
                grad_rho = dist_vec / rho 
                mag = K_REP * (1.0 / rho - 1.0 / RHO_0) * (1.0 / (rho**2))
                F_rep += mag * grad_rho
                
            elif rho <= 0.01: 
                grad_rho = dist_vec / 0.01
                mag = K_REP * (1.0 / 0.01 - 1.0 / RHO_0) * (1.0 / (0.01**2))
                F_rep += mag * grad_rho
                
    return F_att, F_rep

def calcola_costo_step(current_x, u_sol, x_ref):
    err_x = current_x - x_ref
    err_u = u_sol - np.array([U_HOVER, U_HOVER])
    return err_x.T @ Q_COST @ err_x + err_u.T @ R_COST @ err_u

def main_apf(N_TESTS):
    np.random.seed(44)
    random.seed(44)

    risultati = {"Successes": 0, "Timeout": 0, "Crashes": 0}
    registro_globale = []
    
    ambienti, targets, roof = genera_ambienti_random(N_TESTS)

    print(f"--- AVVIO TEST APF: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):
        poligoni, segmenti = ambienti[test_idx]
        current_x = np.array([1.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        x_ref_attuale = target_base.copy()
        
        x_history = [current_x.copy()]
        costo_totale_run = 0.0
        distanza_percorsa = 0.0
        distanza_x = 0.0
        
        esito = "Successes" # Inizializzato come successo (sopravvivenza)
        
        plt.ion()
        fig_anim, ax_anim = plt.subplots(figsize=(18, 5))
        for poli in poligoni:
            ax_anim.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
        ax_anim.scatter(1.0, 5.0, color='blue', s=150, zorder=8)
        
        linea_traj, = ax_anim.plot([], [], color='black', linewidth=2, zorder=6)
        punto_drone = ax_anim.scatter([], [], color='purple', s=150, zorder=7, label="APF Drone")
        punti_lidar_scatter = ax_anim.scatter([], [], color='red', s=10, zorder=9, label="LiDAR Hits")
        
        ax_anim.set_xlim(-2, 52)
        ax_anim.set_ylim(-1, 12)
        ax_anim.grid(True, linestyle='--', alpha=0.5)
        # ax_anim.legend(loc='lower left')
        
        frames_animazione = []
        SIM_TIME = 50.0
        N_SIM = int(SIM_TIME / DT)

        for t in range(N_SIM):
            # 1. Rilevamento Lidar
            hits, _ = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=3.0)
            
            # --- CHECK CRASH FISICO ---
            if len(hits) > 0:
                distanze_ostacoli = np.linalg.norm(hits - current_x[0:2], axis=1)
                if np.min(distanze_ostacoli) <= DRONE_RADIUS:
                    esito = "Crashes"
                    costo_totale_run = 0.0 
                    break
            
            # 2. APF: Calcolo Forze
            F_att, F_rep = compute_apf_forces(current_x, x_ref_attuale, hits)
            F_tot = F_att + F_rep
            
            # 3. Dinamica
            a_x = F_tot[0] / MASS
            a_z = F_tot[1] / MASS
            
            T_des_x = MASS * a_x
            T_des_z = MASS * (a_z + G)
            T_tot = np.sqrt(T_des_x**2 + T_des_z**2)
            
            new_theta = np.arctan2(T_des_x, T_des_z)
            new_wy = (new_theta - current_x[2]) / DT
            
            u_mot = T_tot / (2.0 * CF)
            u_sol = np.array([u_mot, u_mot])
            
            new_vx = current_x[3] + a_x * DT
            new_vz = current_x[4] + a_z * DT
            new_x = current_x[0] + current_x[3] * DT
            new_z = current_x[1] + current_x[4] * DT
            
            # Calcolo distanza percorsa in questo step
            step_dist = np.linalg.norm(np.array([new_x, new_z]) - current_x[0:2])
            
            distanza_percorsa += step_dist
            distanza_x += abs(current_x[0] - new_x)

            current_x = np.array([new_x, new_z, new_theta, new_vx, new_vz, new_wy])
            x_history.append(current_x.copy())
            
            # 4. Accumulo Costo
            costo_totale_run += calcola_costo_step(current_x, u_sol, x_ref_attuale)

            # VIDEO UPDATE
            if t % 5 == 0: 
                traj_attuale = np.array(x_history)
                linea_traj.set_data(traj_attuale[:, 0], traj_attuale[:, 1])
                punto_drone.set_offsets([[current_x[0], current_x[1]]])
                
                if len(hits) > 0:
                    punti_lidar_scatter.set_offsets(hits)
                else:
                    punti_lidar_scatter.set_offsets(np.empty((0, 2)))
                
                ax_anim.set_title(f"APF Target {test_idx +1} | Step: {t} | Dist: {distanza_percorsa:.1f}m | Vx: {current_x[3]:.2f}")
                
                fig_anim.canvas.draw()
                fig_anim.canvas.flush_events()
                
                image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
                frames_animazione.append(image)

        # FINE DEL SINGOLO TEST
        print(f"\n🎯 REPORT APF TARGET {test_idx+1}/{N_TESTS}")
        print(f"Esito finale: {esito} (Iterazione di stop: {t})")
        print(f"Costo accumulato: {costo_totale_run:.2f} | Distanza Percorsa: {distanza_percorsa:.2f} m")
        
        registro_globale.append({
            "target": test_idx + 1,
            "esito": esito,
            "costo": costo_totale_run,
            "distanza": distanza_percorsa,
            "distanza_x ": distanza_x,
            "iterazione": t
        })
        risultati[esito] += 1
        
        plt.ioff()
        plt.close(fig_anim)
        
        # SALVATAGGIO VIDEO
        if len(frames_animazione) > 0:
            cartella_video = "video_apf"
            os.makedirs(cartella_video, exist_ok=True)
            print(f"Generazione del video APF per il Target {test_idx+1} in corso...")
            fig_movie = plt.figure(figsize=(12, 7))
            ax_movie = fig_movie.add_subplot(111)
            ax_movie.axis('off')
            im = ax_movie.imshow(frames_animazione[0])
            
            def update_frame(i):
                im.set_data(frames_animazione[i])
                return [im]
            
            ani = animation.FuncAnimation(fig_movie, update_frame, frames=len(frames_animazione), blit=True)
            nome_video = os.path.join(cartella_video, f"Video_APF_{test_idx+1:02d}_{esito}.mp4")
            try:
                ani.save(nome_video, writer='ffmpeg', fps=10)
            except:
                ani.save(nome_video.replace(".mp4", ".gif"), writer='pillow', fps=10)
            plt.close(fig_movie)

    # ==========================================
    # RESOCONTO GLOBALE FINALE
    # ==========================================
    print("\n" + "="*80)
    print("📊 RESOCONTO GLOBALE APF")
    print("="*80)
    for row in registro_globale:
        print(f"Target {row['target']:02d} | Esito: {row['esito']:<9} | Distanza: {row['distanza']:<6.2f}m | Costo: {row['costo']:<10.2f} | Step Stop: {row['iterazione']:<4}")
    print("="*80 + "\n")
    print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")


    # ==========================================
    # PLOT STATISTICO INDIVIDUALE CON PROFILO VELOCITÀ
    # ==========================================
    fig_singolo, (ax_traj, ax_vel) = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # --- PLOT TRAIETTORIA (Sopra) ---
    for poli in poligoni:
        ax_traj.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
                
    traj = np.array(x_history)
    ax_traj.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=2.5, label='Traiettoria APF')
    ax_traj.plot([50.0, 50.0], [0.0, 11.0], color='red', linestyle='--', linewidth=2, label="Traguardo")
    ax_traj.scatter(1.0, 5.0, color='blue', s=120, label='Start', zorder=5)
        
    ax_traj.set_xlim(-2, 52)
    ax_traj.set_ylim(-1, 12)
    ax_traj.set_title(f"Analisi Test {test_idx+1} | Esito: {esito}")
    ax_traj.grid(True, linestyle='--', alpha=0.5)
    # ax_traj.legend(loc='lower left', fontsize=12)
        
    # --- PLOT VELOCITÀ Vx (Sotto) ---
    steps = np.arange(len(traj))
    vx_storico = traj[:, 3]
        
    ax_vel.plot(steps, vx_storico, color='blue', linewidth=2, label='Vx Corrente')
    ax_vel.axhline(y=target_base[3], color='red', linestyle='--', linewidth=2, label=f'Target Vx ({target_base[3]:.2f} m/s)')
        
    # Aggiungiamo un po' di margine dinamico all'asse Y per inquadrare bene eventuali picchi
    y_min = min(0.0, np.min(vx_storico)) - 0.2
    y_max = max(target_base[3], np.max(vx_storico)) + 0.5
        
    ax_vel.set_xlim(0, len(steps))
    ax_vel.set_ylim(y_min, y_max)
    ax_vel.set_title("Profilo di Velocità (Vx)")
    ax_vel.set_xlabel("Step (k)")
    ax_vel.set_ylabel("Velocità [m/s]")
    ax_vel.grid(True, linestyle='--', alpha=0.5)
    ax_vel.legend(loc='lower right', fontsize=12)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N_TESTS = 10
    main_apf(N_TESTS)