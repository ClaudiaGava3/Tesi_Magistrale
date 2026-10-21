
# ==========================================================================================================================================================
"""Run a planar APF-guided drone simulation with feedback MPC dynamics.

Artificial potential fields combine target-seeking and LiDAR-based repulsive
forces to generate a short-horizon motion reference. A feedback MPC then
tracks that reference and supplies the predicted drone states and motor inputs
used to advance the simulation and accumulate tracking and control cost.
"""
# ==========================================================================================================================================================

from math import tau
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
from matplotlib.patches import Ellipse


from parser import Parameters
from src.MPC.mapping.MPC_test_target_vel import genera_ambienti_random
from src.MPC.mapping.MPC_test_target_vel import aggiungi_ostacoli_densi
from src.MPC.mapping.lidar import get_lidar_hits_2d_qualsiasi
from APF_mpc_abstract import Model
from APF_mpc_controller import MpcController

# ==========================================
# 1. PHYSICAL PARAMETERS AND APF TUNING
# ==========================================

MASS = 0.75          # kg
G = 9.81             # m/s^2
CF = 1.5e-3          # Thrust coefficent
DT = 0.02            # Time step
U_HOVER = (MASS * G) / (2.0 * CF)

I_YY = 0.00735         # Inertia along x-axis (kg*m^2)
L_ARM = 0.10         # Length of the arm (meters)

# Weights for the simulation's accumulated state-tracking and control cost.
Q_COST = np.diag([0.0001, 0.0001, 20.0, 100.0, 1.0, 1.0])
R_COST = np.diag([0.0001, 0.0001])

# ---> PARAMETERS TO TUNE FOR THE APF <---
K_ATT_BASE = 2.5     # Gain on the target-versus-current horizontal velocity error
K_REP = 0.05          # Repulsive gain for each individual LiDAR point
K_DAMP = 5.0
RHO_0 =  3.0         # Limit distance at which the LiDAR point starts repelling
DRONE_RADIUS = 0.1  # Radius of the drone for collision checking

def compute_apf_forces(current_x, x_ref, hits):
    pos_drone = current_x[0:2]
    v_curr_x = current_x[3]
    v_curr_z = current_x[4]  # Serve per lo smorzatore in Z
    v_target_x = x_ref[3]
    
    # 1. ATTRACTIVE FORCE
    # The force accelerates toward the target velocity, or brakes when above it.
    # At equal velocities it is zero, so there is no APF horizontal acceleration.
    f_att_x = K_ATT_BASE * (v_target_x - v_curr_x)
    
    # Always damp vertical velocity toward zero, independently of obstacle hits.
    f_att_z = K_DAMP * (0.0 - v_curr_z)
        
    F_att = np.array([f_att_x, f_att_z])
    
    # 2. REPULSIVE FORCE (Sum of contributions from EACH LiDAR point)
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
    np.random.seed(18)
    random.seed(18)

    print("--- MPC INITIALIZATION ---")
    params = Parameters("sth")
    params.build = True
    model = Model(params)
    mpc = MpcController(model)

    risultati = {"Successes": 0, "Timeout": 0, "Crashes": 0}
    registro_globale = []
    
    ambienti, targets, roof = genera_ambienti_random(N_TESTS)

    ambienti = aggiungi_ostacoli_densi(ambienti, num_rombi=12)

    print(f"--- AVVIO TEST APF: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):
        poligoni, segmenti = ambienti[test_idx]
        current_x = np.array([1.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        x_ref_attuale = target_base.copy()
        
        x_history = [current_x.copy()]
        costo_totale_run = 0.0
        distanza_percorsa = 0.0
        distanza_x = 0.0
        
        esito = "Successes" # Initialized as success (survival)

        # ==========================================
        # SETUP VIDEO REAL-TIME & SALVATAGGIO (DOPPIA CAMERA)
        # ==========================================
        plt.ion() 
        fig_anim = plt.figure(figsize=(10, 10)) # Figura quadrata per la camera circolare
                
        # --- Asse Principale (Zoom sul drone) ---
        ax_main = fig_anim.add_axes([0.1, 0.1, 0.8, 0.8])
        ax_main.set_aspect('equal')
                
        # Maschera per rendere l'apertura circolare (raggio 3m)
        circ_mask = patches.Circle((0.5, 0.5), 0.495, transform=ax_main.transAxes, fill=False, edgecolor='black', linewidth=3, linestyle='--', zorder=15)        
        ax_main.add_patch(circ_mask)
        ax_main.set_clip_path(circ_mask)
        for spine in ax_main.spines.values():
            spine.set_visible(False) # Nasconde il quadrato esterno
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.grid(True, linestyle='--', alpha=0.5)
        
        # --- Asse PiP (Vista Globale, in basso a destra) - INGRANDITO ---
        # [left, bottom, width, height] - ho quasi raddoppiato l'area
        ax_pip = fig_anim.add_axes([0.45, 0.02, 0.52, 0.32]) 
        ax_pip.set_aspect('equal')
        ax_pip.set_xlim(-2, 52)
        ax_pip.set_ylim(-1, 12)
        ax_pip.set_xticks([])
        ax_pip.set_yticks([])
        ax_pip.set_title("Global View", fontsize=16)
                
        # Disegna l'ambiente su ENTRAMBI gli assi
        for poli in poligoni:
            ax_main.add_patch(patches.Polygon(poli, closed=True, facecolor='silver', edgecolor='black', alpha=0.8, linewidth=1.2))
            ax_pip.add_patch(patches.Polygon(poli, closed=True, facecolor='silver', edgecolor='black', alpha=0.8, linewidth=1.2))
                    
        ax_pip.scatter(1.0, 5.0, color='blue', s=20, zorder=8) # Start point solo globale
                
        # --- Elementi Camera Main ---
        drone_main = Ellipse((0, 0), width=0.30, height=0.10, color='blue', zorder=7)
        ax_main.add_patch(drone_main)
        linea_angolo_main, = ax_main.plot([], [], color='red', linewidth=1.5, zorder=8)
        punti_lidar_main = ax_main.scatter([], [], color='red', s=15, zorder=9)
        # box_corrente_main = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=3, alpha=0.3, zorder=4)
        # ax_main.add_patch(box_corrente_main)
                
        # --- Elementi Camera PiP ---
        drone_pip = Ellipse((0, 0), width=0.30, height=0.10, color='green', zorder=7)
        ax_pip.add_patch(drone_pip)
        linea_traj_pip, = ax_pip.plot([], [], color='cyan', linewidth=1.5, zorder=6) # Traiettoria visibile solo qui
        # box_corrente_pip = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=1.5, alpha=0.6, zorder=4)
        # ax_pip.add_patch(box_corrente_pip)
        # Frecce attrattive (ne basta 1, per il drone)
        freccia_att_main = ax_main.quiver(0, 0, 0, 0, color='lime', alpha=0.6, scale_units='xy', angles='xy', scale=1, width=0.015, zorder=10)
        freccia_att_pip = ax_pip.quiver(0, 0, 0, 0, color='lime', alpha=0.6, scale_units='xy', angles='xy', scale=1, width=0.010, zorder=10)
        
        # Frecce repulsive (Prenotiamo array da 360 elementi pari ai raggi LiDAR)
        freccia_rep_main = ax_main.quiver(np.zeros(360), np.zeros(360), np.zeros(360), np.zeros(360), color='red', alpha=0.6, scale_units='xy', angles='xy', scale=1, width=0.012, zorder=10)
        freccia_rep_pip = ax_pip.quiver(np.zeros(360), np.zeros(360), np.zeros(360), np.zeros(360), color='red', alpha=0.6, scale_units='xy', angles='xy', scale=1, width=0.008, zorder=10)
        
        frames_animazione = []
        # ==========================================
        
        SIM_TIME = 50.0
        N_SIM = int(SIM_TIME / DT)

        for t in range(N_SIM):
            # 1. Lidar detection
            hits, _ = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=3.0)
            
            # --- PHYSICAL CRASH CHECK ---
            if len(hits) > 0:
                distanze_ostacoli = np.linalg.norm(hits - current_x[0:2], axis=1)
                if np.min(distanze_ostacoli) <= DRONE_RADIUS:
                    esito = "Crashes"
                    costo_totale_run = 0.0 
                    break
            
            # 2. APF: Force calculation
            F_att, F_rep = compute_apf_forces(current_x, x_ref_attuale, hits)
            F_tot = F_att + F_rep
            
            # 3. Convert the resultant APF force into a desired acceleration.
            a_x = F_tot[0] / MASS
            a_z = F_tot[1] / MASS
            
            # --- Extrapolate an APF-based reference over the MPC horizon ---
            dt_target = DT * mpc.N
            
            x_target_fittizio = current_x[0] + current_x[3] * dt_target + 0.5 * a_x * (dt_target**2)
            z_target_fittizio = current_x[1] + current_x[4] * dt_target + 0.5 * a_z * (dt_target**2)
            vx_target_fittizio = current_x[3] + a_x * dt_target
            vz_target_fittizio = current_x[4] + a_z * dt_target

            
            x_ref_mpc = np.array([x_target_fittizio, z_target_fittizio, 0.0, vx_target_fittizio, vz_target_fittizio, 0.0])

            # --- MPC CALL ---
            x_sol, u_sol_mpc, status = mpc.solve_step(current_x, x_ref_mpc)

            if status not in [0, 2]:
                print(f"⚠️ MPC FALLITO allo step {t} (Status: {status})")
                esito = "Crashes"
                costo_totale_run = 0.0
                break

            # --- Advance the simulation using the first MPC-predicted step ---
            nuovo_stato = x_sol[1]
            u_sol = u_sol_mpc[0]

            # Calculation of distance traveled in this step 
            step_dist = np.linalg.norm(np.array([nuovo_stato[0], nuovo_stato[1]]) - current_x[0:2])
            distanza_percorsa += step_dist
            distanza_x += abs(current_x[0] - nuovo_stato[0])

            # Update current_x
            current_x = nuovo_stato
            x_history.append(current_x.copy())
            
            # 4. Cost accumulation
            costo_totale_run += calcola_costo_step(current_x, u_sol, x_ref_attuale)

            # ==========================================
            # VIDEO FRAME UPDATE (DOPPIA CAMERA)
            # ==========================================
            if t % 5 == 0:  

                # --- Update Main (Zoom Circolare) ---
                ax_main.set_xlim(current_x[0] - 3.0, current_x[0] + 3.0)
                ax_main.set_ylim(current_x[1] - 3.0, current_x[1] + 3.0)
                
                drone_main.set_center((current_x[0], current_x[1]))
                # CORREZIONE: Aggiunto il segno meno per allineare l'ellisse alla linea rossa
                drone_main.set_angle(np.degrees(-current_x[2])) 
                
                lunghezza_ali = 0.30 
                theta = current_x[2] 
                dx_ala = (lunghezza_ali / 2.0) * np.cos(theta)
                dz_ala = (lunghezza_ali / 2.0) * -np.sin(theta)
                linea_angolo_main.set_data([current_x[0] - dx_ala, current_x[0] + dx_ala], [current_x[1] - dz_ala, current_x[1] + dz_ala])
                
                if len(hits) > 0:
                    punti_lidar_main.set_offsets(hits)
                else:
                    punti_lidar_main.set_offsets(np.empty((0, 2)))

               # Fattori di scala visivi (modificali a piacimento per allungare/accorciare le frecce)
                scala_f_att = 0.2
                scala_f_rep = 5.0 
                
                # 1. Forza Attrattiva (Parte dal drone)
                freccia_att_main.set_offsets([[current_x[0], current_x[1]]])
                freccia_att_main.set_UVC(F_att[0] * scala_f_att, F_att[1] * scala_f_att)
                
                freccia_att_pip.set_offsets([[current_x[0], current_x[1]]])
                freccia_att_pip.set_UVC(F_att[0] * scala_f_att, F_att[1] * scala_f_att)

                # 2. Forze Repulsive (Array fissi da 360 per non far crashare Matplotlib)
                X_rep, Y_rep = np.zeros(360), np.zeros(360)
                U_rep, V_rep = np.zeros(360), np.zeros(360)
                
                if len(hits) > 0:
                    idx = 0
                    for hit in hits:
                        dist_vec = current_x[0:2] - hit
                        rho = np.linalg.norm(dist_vec)
                        if 0.01 < rho <= RHO_0:
                            grad_rho = dist_vec / rho
                            mag = K_REP * (1.0 / rho - 1.0 / RHO_0) * (1.0 / (rho**2))
                            
                            X_rep[idx] = hit[0]
                            Y_rep[idx] = hit[1]
                            U_rep[idx] = mag * grad_rho[0] * scala_f_rep
                            V_rep[idx] = mag * grad_rho[1] * scala_f_rep
                            idx += 1
                
                # Applica i vettori
                freccia_rep_main.set_offsets(np.c_[X_rep, Y_rep])
                freccia_rep_main.set_UVC(U_rep, V_rep)
                
                freccia_rep_pip.set_offsets(np.c_[X_rep, Y_rep])
                freccia_rep_pip.set_UVC(U_rep, V_rep)
                
                # # 2. Forze Repulsive (Partono dai singoli ostacoli e puntano verso il drone)
                # X_rep, Y_rep, U_rep, V_rep = [], [], [], []
                # if len(hits) > 0:
                #     for hit in hits:
                #         dist_vec = current_x[0:2] - hit
                #         rho = np.linalg.norm(dist_vec)
                #         if 0.01 < rho <= RHO_0:
                #             grad_rho = dist_vec / rho
                #             mag = K_REP * (1.0 / rho - 1.0 / RHO_0) * (1.0 / (rho**2))
                            
                #             X_rep.append(hit[0])
                #             Y_rep.append(hit[1])
                #             U_rep.append(mag * grad_rho[0] * scala_f_rep)
                #             V_rep.append(mag * grad_rho[1] * scala_f_rep)
                
                # # Gestione di sicurezza: se non ci sono ostacoli, nasconde le frecce a zero
                # if len(X_rep) == 0:
                #     X_rep, Y_rep, U_rep, V_rep = [0], [0], [0], [0]
                
                freccia_rep_main.set_offsets(np.c_[X_rep, Y_rep])
                freccia_rep_main.set_UVC(np.array(U_rep), np.array(V_rep))
                
                freccia_rep_pip.set_offsets(np.c_[X_rep, Y_rep])
                freccia_rep_pip.set_UVC(np.array(U_rep), np.array(V_rep))

                # box_w = box_abs[1] - box_abs[0]
                # box_h = box_abs[3] - box_abs[2]
                # box_corrente_main.set_xy((box_abs[0], box_abs[2]))
                # box_corrente_main.set_width(box_w)
                # box_corrente_main.set_height(box_h)

                # if status in [3, 4]:
                #     box_corrente_main.set_edgecolor('red')
                #     box_corrente_main.set_facecolor('red')
                # else:
                #     box_corrente_main.set_edgecolor('lime')
                #     box_corrente_main.set_facecolor('lime')

                # --- Update PiP (Vista Globale) ---
                drone_pip.set_center((current_x[0], current_x[1]))
                # CORREZIONE: Aggiunto il segno meno anche qui
                drone_pip.set_angle(np.degrees(-current_x[2]))
                
                traj_attuale = np.array(x_history)
                linea_traj_pip.set_data(traj_attuale[:, 0], traj_attuale[:, 1])
                
                # box_corrente_pip.set_xy((box_abs[0], box_abs[2]))
                # box_corrente_pip.set_width(box_w)
                # box_corrente_pip.set_height(box_h)
                
                # if status in [3, 4]:
                #     box_corrente_pip.set_edgecolor('red')
                #     box_corrente_pip.set_facecolor('red')
                # else:
                #     box_corrente_pip.set_edgecolor('lime')
                #     box_corrente_pip.set_facecolor('lime')

                fig_anim.suptitle(f"Target {test_idx +1} | Step: {t} | Dist: {distanza_percorsa:.1f}m | Vx: {current_x[3]:.2f} | Status: {status}", fontsize=20)
                
                # Draw the frame on the screen
                fig_anim.canvas.draw()
                fig_anim.canvas.flush_events()
                
                image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
                frames_animazione.append(image)
            # ==========================================

        # END OF SINGLE TEST
        print(f"\n🎯 REPORT APF TARGET {test_idx+1}/{N_TESTS}")
        print(f"Esito finale: {esito} (Iterazione di stop: {t})")
        print(f"Costo accumulato: {costo_totale_run:.2f} | Dist. Totale: {distanza_percorsa:.2f} m | Dist. X: {distanza_x:.2f} m")
        
        registro_globale.append({
            "target": test_idx + 1,
            "esito": esito,
            "costo": costo_totale_run,
            "distanza": distanza_percorsa,
            "distanza_x": distanza_x,
            "iterazione": t
        })
        risultati[esito] += 1
        
        plt.ioff()
        plt.close(fig_anim)
        
        # SAVE VIDEO
        if len(frames_animazione) > 0:
            cartella_video = "video_article"
            os.makedirs(cartella_video, exist_ok=True)
            print(f"Generazione del video APF per il Target {test_idx+1} in corso...")

            video_dpi = 220
            video_fps = 15
            fig_movie = plt.figure(figsize=(16, 9), dpi=video_dpi)
            ax_movie = fig_movie.add_subplot(111)
            ax_movie.axis('off')
            im = ax_movie.imshow(frames_animazione[0])

            def update_frame(i):
                im.set_data(frames_animazione[i])
                return [im]

            ani = animation.FuncAnimation(
                fig_movie,
                update_frame,
                frames=len(frames_animazione),
                blit=True,
                interval=1000 // video_fps
            )

            nome_video = os.path.join(cartella_video, f"Video_APF_{test_idx+51:02d}_{esito}.mp4")
            try:
                ani.save(
                    nome_video,
                    writer='pillow',
                    fps=video_fps,
                    dpi=video_dpi,
                    codec='libx264',
                    bitrate='8000k'
                )
                print(f"🎬 Video APF salvato con successo: {nome_video}")
            except Exception:
                nome_video_gif = nome_video.replace(".mp4", ".gif")
                ani.save(nome_video_gif, writer='pillow', fps=video_fps, dpi=video_dpi)
                print(f"🎬 Video APF (GIF) salvato con successo: {nome_video_gif}")
            plt.close(fig_movie)

    # ==========================================
    # FINAL GLOBAL REPORT
    # ==========================================
    print("\n" + "="*80)
    print("📊 RESOCONTO GLOBALE APF")
    print("="*80)
    for row in registro_globale:
        print(f"Target {row['target']:02d} | Esito: {row['esito']:<9} | Distanza: {row['distanza']:<6.2f}m | DistanzaX: {row['distanza_x']:<6.2f}m | Costo: {row['costo']:<10.2f} | Step Stop: {row['iterazione']:<4}")
    print("="*80 + "\n")
    print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")


    # ==========================================
    # INDIVIDUAL STATISTICAL PLOT WITH VELOCITY PROFILE
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
        
    # --- PLOT VELOCITY Vx ---
    steps = np.arange(len(traj))
    vx_storico = traj[:, 3]
        
    ax_vel.plot(steps, vx_storico, color='blue', linewidth=2, label='Vx Corrente')
    ax_vel.axhline(y=target_base[3], color='red', linestyle='--', linewidth=2, label=f'Target Vx ({target_base[3]:.2f} m/s)')
        
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
    N_TESTS = 3
    main_apf(N_TESTS)