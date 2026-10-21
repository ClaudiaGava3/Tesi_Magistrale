
# ==========================================================================================================================================================
"""Run a statistical velocity-target MPC campaign in randomized slalom maps.

Each trial generates a polygonal obstacle course, attempts to add 12 dense
diamonds, and tracks a randomly selected horizontal target velocity. The entry
point runs three trials by default. LiDAR mapping, safe-box selection, stall
handling, and solver recovery are evaluated over the campaign. The configuration
and result notes below are from earlier experiments. Dynamic-obstacle
initialization and movement-with-bounce routines are implemented, but the
current ``ambiente_dinamico = False`` setting keeps obstacles stationary; set
it to ``True`` to enable their motion. The controller reports the NN margin but
does not apply it to normal terminal clearances; recovery uses it to relax the
mapped-box terminal bound.

WARNING: set ``Q_cost`` in ``mpc_abstract_obs.py`` to prioritize velocity
tracking before running this test. Its state order is
``[x, z, theta, vx, vz, wy]``; increase the velocity weights for this case.
The local ``Q_COST`` in this script is only used to calculate reported
closed-loop cost and does not configure the MPC objective.
"""
# ==========================================================================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

from parser import Parameters
from src.MPC.mapping.mpc_abstract_obs import Model
from src.MPC.mapping.mpc_controller_obs import MpcController
from src.MPC.mapping.lidar import min_cube_select_base, get_lidar_hits_2d_qualsiasi, min_cube_select_directional, force_trajectory_in_box, min_cube_warm_start


# To prioritize velocity tracking, change Q_cost in mpc_abstract_obs.py;
# this script does not change the MPC objective weights automatically.


# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 22,    # X-axis tick label size
    'ytick.labelsize': 22,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for all other text
})

def ruota_e_trasla(vertici_locali, cx, cz, angolo_deg):
    theta = np.radians(angolo_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    poligono_globale = []
    for x, z in vertici_locali:
        x_rot = x * cos_t - z * sin_t
        z_rot = x * sin_t + z * cos_t
        poligono_globale.append([x_rot + cx, z_rot + cz])
        
    return poligono_globale

import random

def inizializza_velocita_ostacoli(poligoni):
    """
    Assign random velocities and compute each polygon's center and bounding
    radius for boundary and obstacle-collision bounce checks.
    """
    velocita = []
    centri_raggi = []  # Used to detect proximity between moving obstacles.
    
    for poli in poligoni:
        # Use the polygon's X extent to identify fixed boundary structures.
        x_min = min(p[0] for p in poli)
        x_max = max(p[0] for p in poli)
        
        # Keep very wide floor/ceiling polygons and the far-end marker fixed.
        if (x_max - x_min) > 20.0 or x_min > 53.0:
            velocita.append([0.0, 0.0])  # Stationary obstacle.
            centri_raggi.append({'cx': 0, 'cz': 0, 'r': 0})
        else:
            # Assign each ordinary obstacle a random velocity in [-0.5, 0.5] m/s.
            vx = np.random.uniform(-0.5, 0.5)
            vz = np.random.uniform(-0.5, 0.5)
            velocita.append([vx, vz])
            
            # Approximate the polygon by a bounding circle for bounce checks.
            cx = sum(p[0] for p in poli) / len(poli)
            cz = sum(p[1] for p in poli) / len(poli)
            r = max(np.hypot(p[0] - cx, p[1] - cz) for p in poli)
            centri_raggi.append({'cx': cx, 'cz': cz, 'r': r})
            
    return velocita, centri_raggi

def muovi_ostacoli_con_rimbalzi(poligoni, velocita, centri_raggi, dt, roof=11.0):
    """
    Translate polygon vertices and reverse velocities at boundaries or overlaps.
    """
    nuovi_poligoni = []
    nuovi_segmenti = []
    
    for i, poli in enumerate(poligoni):
        vx, vz = velocita[i]
        
        # If the obstacle is stationary, copy it unchanged.
        if vx == 0.0 and vz == 0.0:
            nuovi_poligoni.append(poli)
        else:
            # 1. Translate the obstacle by its current velocity.
            poli_spostato = [[x + vx * dt, z + vz * dt] for x, z in poli]
            nuovi_poligoni.append(poli_spostato)
            
            # Update the moving polygon's bounding-circle center.
            centri_raggi[i]['cx'] += vx * dt
            centri_raggi[i]['cz'] += vz * dt
            
            cx = centri_raggi[i]['cx']
            cz = centri_raggi[i]['cz']
            r = centri_raggi[i]['r']
            
            # 2. Reverse velocity when the obstacle reaches a room boundary.
            # Vertical bounce at the floor or ceiling.
            if cz - r < 0.0:
                velocita[i][1] *= -1  # Reverse vertical velocity.
            elif cz + r > roof:
                velocita[i][1] *= -1  # Reverse vertical velocity.
                
            # Horizontal bounce between the configured X limits.
            if cx - r < 5.0:
                velocita[i][0] *= -1  # Reverse horizontal velocity.
            elif cx + r > 50.0:
                velocita[i][0] *= -1  # Reverse horizontal velocity.
                
            # 3. Reverse velocity when moving obstacle bounding circles overlap.
            for j in range(len(poligoni)):
                if i != j and velocita[j] != [0.0, 0.0]:  # Check moving obstacles only.
                    cx_altro = centri_raggi[j]['cx']
                    cz_altro = centri_raggi[j]['cz']
                    r_altro = centri_raggi[j]['r']
                    
                    distanza = np.hypot(cx - cx_altro, cz - cz_altro)
                    dist_minima_sicurezza = r + r_altro
                    
                    if distanza < dist_minima_sicurezza:
                        # Approximate an elastic collision by reversing both velocity components.
                        velocita[i][0] *= -1
                        velocita[i][1] *= -1
        
        # 4. Rebuild wall segments for the LiDAR detector.
        n = len(nuovi_poligoni[-1])
        for v in range(n):
            nuovi_segmenti.append([nuovi_poligoni[-1][v], nuovi_poligoni[-1][(v + 1) % n]])
            
    return nuovi_poligoni, nuovi_segmenti

def genera_ambienti_random(num_tests=10):
    ambienti = []
    targets = []
    roof = 11.0

    # Base polygon shapes.
    v_triangolo  = [[-2.0, -1.5], [2.0, -1.5], [0.0, 1.5]]                  
    v_rettangolo = [[-2.0, -0.75], [2.0, -0.75], [2.0, 0.75], [-2.0, 0.75]] 
    v_quadrato   = [[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]]     
    v_trapezio   = [[-2.0, -1.0], [2.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] 
    forme = [v_triangolo, v_rettangolo, v_quadrato, v_trapezio]

    for _ in range(num_tests):
        poligoni = []
        # Fixed floor and ceiling.
        poligoni.append([[-2.0, 0.0], [52.0, 0.0], [52.0, -1.0], [-2.0, -1.0]]) 
        poligoni.append([[-2.0, 11.0], [52.0, 11.0], [52.0, 12.0], [-2.0, 12.0]]) 

        num_ostacoli = 7
        x_start = 8.0
        x_end = 48.0
        step_x = (x_end - x_start) / num_ostacoli  # About 5.7 m per sector.

        # Randomly choose whether the slalom starts with an upper or lower obstacle.
        inizia_alto = random.choice([True, False])

        for i in range(num_ostacoli):
            forma = random.choice(forme)
            
            # 1. Place the obstacle within its longitudinal sector with random margin.
            cx = np.random.uniform(x_start + i * step_x + 0.5, x_start + (i + 1) * step_x - 0.5)
            
            # 2. Alternate upper and lower obstacle placement.
            is_high = (i % 2 == 0) if inizia_alto else (i % 2 != 0)
            
            if is_high:
                angolo = random.choice([180, 135, 225])
            else:
                angolo = random.choice([0, 45, -45, 20, -20])

            # 3. Compute the rotated vertical extent to keep the obstacle inside the room.
            forma_ruotata_zero = ruota_e_trasla(forma, 0.0, 0.0, angolo)
            z_min_locale = min(v[1] for v in forma_ruotata_zero)
            z_max_locale = max(v[1] for v in forma_ruotata_zero)
            
            # 4. Sample a vertical position within the allowed band.
            margin = 0.0
            if is_high:
                limite_sup = 11.0 - margin - z_max_locale
                limite_inf = max(6.0, 0.0 + margin - z_min_locale) 
                cz = np.random.uniform(limite_inf, limite_sup) if limite_sup > limite_inf else limite_sup
            else:
                limite_inf = 0.0 + margin - z_min_locale
                limite_sup = min(4.0, 11.0 - margin - z_max_locale)
                cz = np.random.uniform(limite_inf, limite_sup) if limite_sup > limite_inf else limite_inf

            poligoni.append(ruota_e_trasla(forma, cx=cx, cz=cz, angolo_deg=angolo))

        segments = []
        for poli in poligoni:
            n = len(poli)
            for i in range(n):
                segments.append([poli[i], poli[(i + 1) % n]])
        
        # Add a far-end segment so LiDAR rays detect the end of the course.
        segments.append([[55.0, 0.0], [55.0, 11.0]])

        ambienti.append((poligoni, segments))

        # Sample a horizontal target velocity for this environment.
        vx_rand = round(np.random.uniform(0.8, 1.2), 2)
        targets.append(np.array([0.0, 0.0, 0.0, vx_rand, 0.0, 0.0]))

    return ambienti, targets, roof


def dist_punto_segmento(px, pz, p1, p2):
    """Calcola la distanza minima tra un punto e un segmento."""
    x1, z1 = p1
    x2, z2 = p2
    l2 = (x2 - x1)**2 + (z2 - z1)**2
    if l2 == 0: 
        return np.hypot(px - x1, pz - z1)
    
    # Proiezione ortogonale del punto sul segmento (parametro t tra 0 e 1)
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (pz - z1) * (z2 - z1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_z = z1 + t * (z2 - z1)
    return np.hypot(px - proj_x, pz - proj_z)

def aggiungi_ostacoli_densi(ambienti, num_rombi=10):
    """Add spaced diamond obstacles while keeping clearance from existing walls."""
    # Diamond half-diagonal is 0.6 m; total width and height are 1.2 m.
    v_rombo = [[0.0, 0.6], [0.6, 0.0], [0.0, -0.6], [-0.6, 0.0]] 
    nuovi_ambienti = []
    
    for poligoni, segmenti in ambienti:
        nuovi_poligoni = list(poligoni)
        nuovi_segmenti = list(segmenti)
        
        tentativi = 0
        rombi_aggiunti = 0
        centri_piazzati = []  # Centers of previously placed diamonds.
        
        # Allow many placement attempts because the clearance checks are restrictive.
        while rombi_aggiunti < num_rombi and tentativi < 500:
            # Sample a candidate center.
            cx = np.random.uniform(7.0, 47.0)
            cz = np.random.uniform(2.0, 9.0)
            
            # 1. Enforce minimum spacing between diamond centers.
            troppo_vicini_tra_loro = False
            for (px, pz) in centri_piazzati:
                if np.hypot(cx - px, cz - pz) < 3.5:  # Minimum center-to-center distance [m].
                    troppo_vicini_tra_loro = True
                    break
            
            if troppo_vicini_tra_loro:
                tentativi += 1
                continue
                
            # 2. Enforce clearance from the existing obstacle segments.
            troppo_vicino_ai_muri = False
            for seg in nuovi_segmenti:
                if seg[0][0] > 54.0:  # Ignore the far-end marker.
                    continue
                
                dist = dist_punto_segmento(cx, cz, seg[0], seg[1])
                # Clearance = 0.6 m diamond radius + 0.1 m drone radius + 0.5 m margin.
                if dist < 1.2: 
                    troppo_vicino_ai_muri = True
                    break
            
            # 3. Add the diamond and its edges if the clearance checks pass.
            if not troppo_vicino_ai_muri:
                rombo_traslato = [[x+cx, z+cz] for x, z in v_rombo]
                nuovi_poligoni.append(rombo_traslato)
                
                n = len(rombo_traslato)
                for j in range(n):
                    nuovi_segmenti.append([rombo_traslato[j], rombo_traslato[(j+1)%n]])
                    
                centri_piazzati.append((cx, cz))  # Store the center for later checks.
                rombi_aggiunti += 1
            
            tentativi += 1
            
        nuovi_ambienti.append((nuovi_poligoni, nuovi_segmenti))
        
    return nuovi_ambienti

def anteprima_ambiente(num_rombi=10):
    """Generate and display one environment with added diamonds for visual inspection."""
    # 1. Generate one base environment.
    ambienti_base, targets, roof = genera_ambienti_random(1)
    
    # 2. Add the requested diamonds.
    ambienti_densi = aggiungi_ostacoli_densi(ambienti_base, num_rombi=num_rombi)
    
    # 3. Extract polygons for plotting.
    poligoni, segmenti = ambienti_densi[0] 

    # 4. Configure the plot.
    fig, ax = plt.subplots(figsize=(14, 4))

    # Draw all obstacles, including the added diamonds.
    for poli in poligoni:
        ax.add_patch(patches.Polygon(poli, closed=True, facecolor='silver', edgecolor='black', alpha=0.8, linewidth=1.2))
        
    # Fixed reference elements (finish marker and start point).
    # ax.plot([50.0, 50.0], [0.0, 11.0], color='red', linestyle='--', linewidth=2, label="Traguardo")
    # ax.scatter(1.0, 5.0, color='blue', s=150, label="Start Drone", zorder=5)

    # Plot formatting.
    ax.set_xlim(-2, 52)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal') # Mantiene le proporzioni reali (non distorce i rombi)
    ax.grid(True, linestyle='--', alpha=0.5)
    # ax.set_title("Test Environment A")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    # ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

def main_statistico(N_TESTS):

    np.random.seed(18)
    random.seed(18)
    
    risultati = {"Successes": 0, "Timeout": 0, "Crashes": 0}
    traiettorie_riuscita = []
    all_box = []
    totale_recovery_attivate = 0
    crashes_per_feasibility = 0
    tutte_le_traiettorie = []
    registro_violazioni_globale = []

    # Weights used only for the reported closed-loop cost.
    Q_COST = np.diag([0.0001, 0.0001, 20.0, 100.0, 1.0, 1.0])
    R_COST = np.diag([0.0001, 0.0001])

    params = Parameters("sth") 
    params.act = 'gelu'
    params.build = True

    model = Model(params)
    controller = MpcController(model)
    u_hover = (model.mass * 9.81) / (2.0 * model.cf)

    DT = params.dt
    SIM_TIME = 50.0 # Extended time to cover all targets
    N_SIM = int(SIM_TIME / DT)
    
    
    ambienti, targets, roof = genera_ambienti_random(N_TESTS)

    ambienti = aggiungi_ostacoli_densi(ambienti, num_rombi=12)

    print(f"--- AVVIO TEST DI COPERTURA: {N_TESTS} TARGET ---")
    
    for test_idx, target_base in enumerate(targets):

        # Get the polygons and wall segments for this trial.
        poligoni, segmenti = ambienti[test_idx]

        # Motion code is available above; leave False for static-obstacle trials.
        ambiente_dinamico = False
        if ambiente_dinamico:
            velocita_ostacoli, info_ostacoli = inizializza_velocita_ostacoli(poligoni)

        # Fixed starting state.
        current_x = np.array([1.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        
        x_history = [current_x.copy()]
        u_history = []
        box_history = []
        
        recovery_in_questo_test = 0
        esito = "Successes" 

        # Stall-detection state.
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
        

        # # ==========================================
        # # Real-time video setup and frame capture (older single-camera version).
        # # ==========================================
        # plt.ion() # Enable real-time video mode
        # fig_anim, ax_anim = plt.subplots(figsize=(18, 5))
        
        # # Draw the fixed environment once.
        # ostacoli_patches = []
        # for poli in poligoni:
        #     patch = patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5)
        #     ax_anim.add_patch(patch)
        #     ostacoli_patches.append(patch)
        # # ax_anim.scatter(target_base[0], target_base[1], color='red', marker='X', s=200, zorder=8)
        # ax_anim.scatter(1.0, 5.0, color='blue', s=150, zorder=8)
        
        # # linea_traj, = ax_anim.plot([], [], color='black', linewidth=2, zorder=6)
        # # linea_pred, = ax_anim.plot([], [], color='orange', linewidth=2, zorder=5)
        # # punto_drone = ax_anim.scatter([], [], color='green', s=150, zorder=7)
        # drone_patch = Ellipse((0, 0), width=0.30, height=0.10, color='black', zorder=7)  # Width is twice the X semi-axis.
        # ax_anim.add_patch(drone_patch)

        # linea_angolo_drone, = ax_anim.plot([], [], color='red', linewidth=1.5, zorder=6)

        # punti_lidar_scatter = ax_anim.scatter([], [], color='red', s=10, zorder=9, label="LiDAR Hits")

        # box_corrente = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=3, alpha=0.3, zorder=4)
        # ax_anim.add_patch(box_corrente)
        
        # ax_anim.set_xlim(-2, 52)
        # ax_anim.set_ylim(-1, 12)
        # ax_anim.grid(True, linestyle='--', alpha=0.5)

        # # Store captured frames in this list for video export.
        # frames_animazione = []
        # # ==========================================

        # ==========================================
        # REAL-TIME VIDEO SETUP AND FRAME CAPTURE (DUAL CAMERA).
        # ==========================================
        plt.ion() 
        fig_anim = plt.figure(figsize=(10, 10)) # Figura quadrata per la camera circolare
        
        # --- Asse Principale (Zoom sul drone) ---
        ax_main = fig_anim.add_axes([0.1, 0.1, 0.8, 0.8])
        ax_main.set_aspect('equal')
        
        # Circular viewport mask (approximately 3 m viewing radius).
        circ_mask = patches.Circle((0.5, 0.5), 0.495, transform=ax_main.transAxes, fill=False, edgecolor='black', linewidth=3, linestyle='--', zorder=15)        
        ax_main.add_patch(circ_mask)
        ax_main.set_clip_path(circ_mask)
        for spine in ax_main.spines.values():
            spine.set_visible(False) # Nasconde il quadrato esterno
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        ax_main.grid(True, linestyle='--', alpha=0.5)

        # --- Asse PiP (Vista Globale, in basso a destra) - INGRANDITO ---
        # [left, bottom, width, height]; the PiP view was enlarged.
        ax_pip = fig_anim.add_axes([0.45, 0.02, 0.52, 0.32]) 
        ax_pip.set_aspect('equal')
        ax_pip.set_xlim(-2, 52)
        ax_pip.set_ylim(-1, 12)
        ax_pip.set_xticks([])
        ax_pip.set_yticks([])
        ax_pip.set_title("Global View", fontsize=16)
        
        # Draw the environment in both axes.
        for poli in poligoni:
            ax_main.add_patch(patches.Polygon(poli, closed=True, facecolor='silver', edgecolor='black', alpha=0.8, linewidth=1.2))
            ax_pip.add_patch(patches.Polygon(poli, closed=True, facecolor='silver', edgecolor='black', alpha=0.8, linewidth=1.2))
            
        ax_pip.scatter(1.0, 5.0, color='blue', s=20, zorder=8)  # Start marker in the global view only.
        
        # --- Elementi Camera Main ---
        drone_main = Ellipse((0, 0), width=0.30, height=0.10, color='blue', zorder=7)
        ax_main.add_patch(drone_main)
        linea_angolo_main, = ax_main.plot([], [], color='red', linewidth=1.5, zorder=8)
        punti_lidar_main = ax_main.scatter([], [], color='red', s=15, zorder=9)
        box_corrente_main = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=3, alpha=0.3, zorder=4)
        ax_main.add_patch(box_corrente_main)
        
        # --- Elementi Camera PiP ---
        drone_pip = Ellipse((0, 0), width=0.30, height=0.10, color='green', zorder=7)
        ax_pip.add_patch(drone_pip)
        linea_traj_pip, = ax_pip.plot([], [], color='cyan', linewidth=1.5, zorder=6)  # Trajectory is shown only in this view.
        box_corrente_pip = patches.Rectangle((0, 0), 0, 0, edgecolor='lime', facecolor='lime', linewidth=1.5, alpha=0.6, zorder=4)
        ax_pip.add_patch(box_corrente_pip)

        frames_animazione = []
        # ==========================================

        iterazioni_violazione_box = []

        costo_totale_run = 0.0
        distanza_percorsa = 0.0
        distanza_x = 0.0

        for t in range(N_SIM):

            if ambiente_dinamico:
                poligoni, segmenti = muovi_ostacoli_con_rimbalzi(
                    poligoni, velocita_ostacoli, info_ostacoli, DT, roof
                )

            # 0. Recovery timer management
            if in_recovery:
                x_ref_attuale = target_recovery
                timer_recovery -= 1
                if timer_recovery <= 0:
                    in_recovery = False
                    
                    #Restore constraints
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
                # Restore the base target if a previous stall changed it.
                if contatore_stallo == 0:
                    x_ref_attuale = current_target.copy()

            # 1. LiDAR mapping and local safe-box construction.
            hits, radii = get_lidar_hits_2d_qualsiasi(current_x[0], current_x[1], segmenti, num_rays=360, max_range=3.0)


            # ==========================================
            # --- CHECK CRASH FISICO (L'Arbitro) ---
            # ==========================================
            DRONE_RADIUS = 0.1
            if len(hits) > 0:
                # Compute the Euclidean distance from the drone to each LiDAR hit.
                distanze_ostacoli = np.linalg.norm(hits - current_x[0:2], axis=1)
                
                if np.min(distanze_ostacoli) <= DRONE_RADIUS:
                    esito = "Crashes"
                    costo_totale_run = 0.0
                    distanza_percorsa = 0.0
                    distanza_x = 0.0
                    print(f"💥 CRASH FISICO RILEVATO allo step {t}! Distanza dal muro: {np.min(distanze_ostacoli):.3f}m")
                    break
            # ==========================================

            
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                
            # This signal is desired velocity, not target displacement.
            target_rel_x = x_ref_attuale[3]
            target_rel_z = x_ref_attuale[4]

            dx = target_rel_x
            dz = target_rel_z


            # Alternative direction signals can be enabled here; the active
            # option below uses the previous MPC trajectory when available.

            # # --- Alternative: use measured velocity, falling back to target ---
            # dx = current_x[3] if abs(current_x[3]) > 0.1 else target_rel_x
            # dz = current_x[4] if abs(current_x[4]) > 0.1 else target_rel_z

            # Use the predicted vx/vz at a future MPC node as the direction signal.
            if x_sol_prev is not None and len(x_sol_prev) > 5:
                
                dx = x_sol_prev[5][3] #- current_x[0]
                dz = x_sol_prev[5][4] #- current_x[1]
                

                    # # If the predicted direction is nearly zero, fall back to the target direction.
                # if abs(dx) < 0.05 and abs(dz) < 0.05:#0.01
                #     dx, dz = target_rel_x, target_rel_z

            else:
                # Al primissimo passo (t=0)
                dx = target_rel_x
                dz = target_rel_z


      

            # ==========================================
            # WARM-START SAFE BOX FROM THE PREVIOUS PREDICTED TRAJECTORY.
            # ==========================================
            if x_sol_prev is not None:
                # 1. Find the min/max bounds of the full trajectory predicted at the previous step.
                traj_xmin = np.min(x_sol_prev[:, 0])
                traj_xmax = np.max(x_sol_prev[:, 0])
                traj_zmin = np.min(x_sol_prev[:, 1])
                traj_zmax = np.max(x_sol_prev[:, 1])
                
                # 2. Convert those bounds to coordinates relative to the current drone position.
                margin = 0.15
                box_prev_rel = [
                    traj_xmin - current_x[0]-margin,
                    traj_xmax - current_x[0]+margin,
                    traj_zmin - current_x[1]-margin,
                    traj_zmax - current_x[1]+margin
                ]
            else:
                box_prev_rel = None
            # ==========================================



            # Active safe-box method: warm-start from the previous predicted
            # trajectory bounds, then use directional expansion. The calls
            # below are alternatives for testing other box-selection methods.
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


            # # classical
            # xMin_r, xMax_r, zMin_r, zMax_r, _ = min_cube_select_base(
            #      Q_rel, radii, dx, dz, drone_radius=0.1, W=50, rel=0.1)
            
            
            box_abs = np.array([
                xMin_r + current_x[0], xMax_r + current_x[0], 
                zMin_r + current_x[1], zMax_r + current_x[1]
            ])
            box_history.append(box_abs.copy())
            

            # ==========================================
            # CHECK RECURSIVE FEASIBILITY
            # Check whether the previous predicted trajectory lies inside the newly mapped box.
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


            # 2. Solve the MPC problem.
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)

            if status not in [3, 4]:
                # The full MPC objective includes predicted stage costs and a
                # terminal cost, so summing solver costs at every step would
                # not match the APF closed-loop cost and could double-count terms.
                # costo_totale_run += controller.ocp_solver.get_cost()


                # Accumulate a one-step closed-loop cost for comparison with APF.
                err_x = current_x - x_ref_attuale
                # u_sol[0] is the motor input applied at the current step.
                err_u = u_sol[0] - np.array([u_hover, u_hover]) 
                
                costo_step = err_x.T @ Q_COST @ err_x + err_u.T @ R_COST @ err_u
                costo_totale_run += costo_step

            # # ==========================================
            # # DIAGNOSTIC BLOCK (MPC DEBUGGER)
            # # ==========================================
            # if t % 10 == 0: # Print every 10 steps
            #     print(f"\n--- DEBUG STEP {t} ---")
            #     print(f"1. Drone Position : X={current_x[0]:.2f}, Z={current_x[1]:.2f}")
            #     print(f"2. Green Box (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Z in [{box_abs[2]:.2f}, {box_abs[3]:.2f}]")
            #     print(f"3. Local target: X={x_ref_attuale[0]:.2f}, Z={x_ref_attuale[1]:.2f}")
                
                
            #     # Compute distance between drone and local target
            #     dist_to_local = np.linalg.norm(current_x[:2] - np.array([target_rel_x, target_rel_z])[:2])
            #     print(f"5. Distance to local target: {dist_to_local:.3f} m")
            # # ==========================================


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

                if status not in [3, 4]:
                    # Add the optimizer cost for this accepted recovery solve.
                    costo_totale_run += controller.ocp_solver.get_cost()

                # PLAN B: historical warm start
                if (status in [3, 4]) and not in_recovery and len(u_history) > 0:
                    for i in range(len(u_history) - 1, -1, -1):
                        past_u = u_history[i]
                        controller.ocp_solver.reset()
                        controller.x_guess = np.tile(current_x, (controller.N, 1))
                        controller.u_guess = past_u
                        x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                        if status in [0, 2]:
                            costo_totale_run += controller.ocp_solver.get_cost()
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

                    if status not in [3, 4]:
                        # Add the optimizer cost for this accepted recovery solve.
                        costo_totale_run += controller.ocp_solver.get_cost()
                    
                    if status in [3, 4]:
                        esito = "Crashes"
                        in_recovery = False
                        costo_totale_run = 0.0

                        # --- BEGIN CRASH DIAGNOSTICS ---
                        if x_sol_prev is not None:
                            punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3))
                            print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                            if punti_fuori > 0:
                                crashes_per_feasibility += 1
                        # --- END CRASH DIAGNOSTICS ---

                        break
                
                elif status in [3, 4] and in_recovery:
                    esito = "Crashes"
                    in_recovery = False
                    costo_totale_run = 0.0

                    # Check whether the previous predicted trajectory left the mapped box.
                    if x_sol_prev is not None:
                        punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3))
                        print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                        if punti_fuori > 0:
                                crashes_per_feasibility += 1
                    break

            if u_sol is not None:
                u_history.append(u_sol.copy())

            step_dist = np.linalg.norm(x_sol[1, 0:2] - current_x[0:2])
            distanza_percorsa += step_dist
            distanza_x += abs(current_x[0]-x_sol[1,0])
            

            current_x = x_sol[1]
            x_history.append(current_x.copy())

            x_sol_prev = x_sol.copy()
            box_abs_prev = box_abs.copy()

            

            # # ==========================================
            # # VIDEO FRAME UPDATE
            # # ==========================================
            # if t % 5 == 0:  # Update every five steps to reduce video overhead.

            #     if ambiente_dinamico:
            #         for idx_patch, poli in enumerate(poligoni):
            #             ostacoli_patches[idx_patch].set_xy(poli)

            #     # Update the trajectory line.
            #     # traj_attuale = np.array(x_history)
            #     # linea_traj.set_data(traj_attuale[:, 0], traj_attuale[:, 1])

            #     if x_sol is not None:
            #         traj_pred_array = np.array(x_sol)
            #         # linea_pred.set_data(traj_pred_array[:, 0], traj_pred_array[:, 1])
                
            #     # Update the drone's position
            #     # punto_drone.set_offsets([[current_x[0], current_x[1]]])
            #     drone_patch.set_center((current_x[0], current_x[1]))
            #     drone_patch.set_angle(np.degrees(current_x[2])) # Se vuoi anche che si inclini visivamente in base al pitch!

            #     lunghezza_corpo = 0.8
            #     theta = current_x[2] 
            #     dx = (lunghezza_corpo / 2.0) * np.cos(theta)
            #     dz = (lunghezza_corpo / 2.0) * -np.sin(theta)
            #     linea_angolo_drone.set_data([current_x[0] - dx, current_x[0] + dx], [current_x[1] - dz, current_x[1] + dz])
                
            #     # Update the coordinates and dimensions of the green safe-box
            #     box_w = box_abs[1] - box_abs[0]
            #     box_h = box_abs[3] - box_abs[2]
            #     box_corrente.set_xy((box_abs[0], box_abs[2]))
            #     box_corrente.set_width(box_w)
            #     box_corrente.set_height(box_h)

            #     if len(hits) > 0:
            #         punti_lidar_scatter.set_offsets(hits)
            #     else:
            #         punti_lidar_scatter.set_offsets(np.empty((0, 2)))

                
            #     # Change the color to red if the solver goes into Status 4 (Crash/Infeasibility)
            #     if status in [3, 4]:
            #         box_corrente.set_edgecolor('red')
            #         box_corrente.set_facecolor('red')
            #     else:
            #         box_corrente.set_edgecolor('lime')
            #         box_corrente.set_facecolor('lime')
                
            #     # Update the animation title.
            #     ax_anim.set_title(f"MPC-NN target {test_idx +1} | Step: {t} | Distance: {distanza_percorsa:.1f} m | Vx: {current_x[3]:.2f} | MPC status: {status}")
                
            #     # Draw the frame on the screen
            #     fig_anim.canvas.draw()
            #     fig_anim.canvas.flush_events()
                
            #     # (Rimosso plt.pause(0.1) per non rallentare l'esecuzione)

            #     # Convert the current canvas to an image frame for the video.
            #     image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
            #     image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
            #     frames_animazione.append(image)
            # # ==========================================

            # ==========================================
            # VIDEO FRAME UPDATE (DUAL CAMERA).
            # ==========================================
            if t % 5 == 0:

                if ambiente_dinamico:
                    for idx_patch, poli in enumerate(poligoni):
                        ax_main.patches[1+idx_patch].set_xy(poli)  # Offset by one to skip the circular mask patch.
                        ax_pip.patches[idx_patch].set_xy(poli)  

                # --- Update Main (Zoom Circolare) ---
                ax_main.set_xlim(current_x[0] - 3.0, current_x[0] + 3.0)
                ax_main.set_ylim(current_x[1] - 3.0, current_x[1] + 3.0)
                
                drone_main.set_center((current_x[0], current_x[1]))
                # Negate the angle so the ellipse aligns with the red orientation line.
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

                box_w = box_abs[1] - box_abs[0]
                box_h = box_abs[3] - box_abs[2]
                box_corrente_main.set_xy((box_abs[0], box_abs[2]))
                box_corrente_main.set_width(box_w)
                box_corrente_main.set_height(box_h)

                if status in [3, 4]:
                    box_corrente_main.set_edgecolor('red')
                    box_corrente_main.set_facecolor('red')
                else:
                    box_corrente_main.set_edgecolor('lime')
                    box_corrente_main.set_facecolor('lime')

                # --- Update PiP (Vista Globale) ---
                drone_pip.set_center((current_x[0], current_x[1]))
                # Apply the same angle convention in the global view.
                drone_pip.set_angle(np.degrees(-current_x[2]))
                
                traj_attuale = np.array(x_history)
                linea_traj_pip.set_data(traj_attuale[:, 0], traj_attuale[:, 1])
                
                box_corrente_pip.set_xy((box_abs[0], box_abs[2]))
                box_corrente_pip.set_width(box_w)
                box_corrente_pip.set_height(box_h)
                
                if status in [3, 4]:
                    box_corrente_pip.set_edgecolor('red')
                    box_corrente_pip.set_facecolor('red')
                else:
                    box_corrente_pip.set_edgecolor('lime')
                    box_corrente_pip.set_facecolor('lime')

                fig_anim.suptitle(f"Target {test_idx +1} | Step: {t} | Dist: {distanza_percorsa:.1f}m | Vx: {current_x[3]:.2f} | Status: {status}", fontsize=20)
                
                # Draw the frame on the screen
                fig_anim.canvas.draw()
                fig_anim.canvas.flush_events()
                
                image = np.frombuffer(fig_anim.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig_anim.canvas.get_width_height()[::-1] + (3,))
                frames_animazione.append(image)
            # ==========================================

            # # ==========================================
            # # Alternative stall handling: raise the waypoint, then return to base.
            # # ==========================================
            # # 1. End the trial when the target is reached.
            # dist_target_reale = np.linalg.norm(current_x[:2] - target_base[:2])
            # if dist_target_reale < 0.3:
            #     esito = "Successes"
            #     break
                            
            # if t > 0:
            #     # 2. NORMAL STALL HANDLING
            #     displacement = np.linalg.norm(current_x[:2] - x_history[-2][:2])
            #     if displacement < 0.01:
            #         contatore_stallo += 1
            #     else:
            #         contatore_stallo = 0 

            #     if contatore_stallo >= MAX_STALLO_ITER:
            #         print(f"\n⚠️ STALL DETECTED (Step {t})! Perturb the target upward.")
                    
            #         safe_z_max = roof - 0.2  # Keep the target 0.2 m below the ceiling.
            #         current_target[1] = min(current_target[1] + 0.20, safe_z_max)
            #         x_ref_attuale = current_target.copy()
            #         contatore_stallo = 0

            #     # 3. Check arrival at the local target.
            #     dist_al_locale = np.linalg.norm(current_x[:2] - current_target[:2])
            #     if dist_al_locale < 0.3:
            #         if not np.array_equal(current_target, target_base):
            #             print(f"\n✅ Elevated target reached. Descend toward the final ground target.")
            #             current_target = target_base.copy()
            #             x_ref_attuale = current_target.copy()
            #             contatore_stallo = 0


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
        print(f"Dist. Totale: {distanza_percorsa:.2f} m | Dist. X: {distanza_x:.2f} m | Costo: {costo_totale_run:.2f}")
        
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
            "costo": costo_totale_run,
            "distanza": distanza_percorsa,
            "distanza_x": distanza_x,
            "iterazione": t,
            "num_violazioni": len(iterazioni_violazione_box),
            "correlazione": tipo_correlazione
        })
        # ----------------------------------------


        # ==========================================
        # END OF VIDEO
        # ==========================================
        # ==========================================
        # EXPORT AND SAVE THE VIDEO.
        # ==========================================
        plt.ioff()
        plt.close(fig_anim)  # Close the real-time figure to free memory.

        if len(frames_animazione) > 0:
            cartella_video = "video_article"
            os.makedirs(cartella_video, exist_ok=True)

            print(f"Generazione del video per il Target {test_idx+1} in corso...")

            # Use higher resolution for the exported video.
            video_dpi = 220
            video_fps = 15
            fig_movie = plt.figure(figsize=(16, 9), dpi=video_dpi)
            ax_movie = fig_movie.add_subplot(111)
            ax_movie.axis('off')

            # Display the first frame.
            im = ax_movie.imshow(frames_animazione[0])

            def update_frame(i):
                im.set_data(frames_animazione[i])
                return [im]

            # Create the animation from the captured frames.
            ani = animation.FuncAnimation(fig_movie, update_frame, frames=len(frames_animazione), blit=True, interval=1000 // video_fps)

            # Configure the output video filename.
            nome_video = os.path.join(cartella_video, f"Video_MPC-NN_{test_idx+1:02d}_{esito}.mp4")

            # Save as high-resolution MP4 (requires FFmpeg).
            # If FFmpeg is unavailable, use a .gif extension and the Pillow writer.
            try:
                ani.save(
                    nome_video,
                    writer='pillow',
                    fps=video_fps,
                    dpi=video_dpi,
                    codec='libx264',
                    bitrate='8000k'
                )
                print(f"🎬 Video salvato con successo: {nome_video}")
            except Exception as e:
                print("Nota: 'ffmpeg' non trovato, provo a salvare in formato .gif...")
                nome_video_gif = nome_video.replace(".mp4", ".gif")
                ani.save(nome_video_gif, writer='pillow', fps=video_fps, dpi=video_dpi)
                print(f"🎬 Video (GIF) salvato con successo: {nome_video_gif}")

            plt.close(fig_movie)
        # ==========================================

        # # ==========================================
        # # Optional per-target statistical plot (place inside the test loop).
        # # Sostituisce la vecchia parte commentata "CORRECTION 3"
        # # ==========================================
        # fig_singolo, ax_singolo = plt.subplots(figsize=(18, 5))
        
        # # Draw only the obstacles for this environment.
        # for poli in poligoni:
        #     ax_singolo.add_patch(patches.Polygon(poli, closed=True, facecolor='gray', edgecolor='black', alpha=0.5))
            
        # # Draw sampled safe boxes (one every ten steps to avoid clutter).
        # if len(box_history) > 0:
        #     for box in box_history[::10]:
        #         box_w = box[1] - box[0]
        #         box_h = box[3] - box[2]
        #         ax_singolo.add_patch(patches.Rectangle((box[0], box[2]), box_w, box_h, edgecolor='lime', facecolor='none', linewidth=0.8, alpha=0.3))
                
        # # Plot the trajectory.
        # traj = np.array(x_history)
        # ax_singolo.plot(traj[:, 0], traj[:, 1], color='cyan', linewidth=2.5, label='Trajectory')
        # ax_singolo.plot([50.0, 50.0], [0.0, 11.0], color='red', linestyle='--', linewidth=2, label="Finish marker")
        # ax_singolo.scatter(1.0, 5.0, color='blue', s=120, label='Start', zorder=5)
        
        # ax_singolo.set_xlim(-2, 52)
        # ax_singolo.set_ylim(-1, 12)
        # ax_singolo.set_title(f"Test {test_idx+1} | Target Vx: {target_base[3]} m/s | Outcome: {esito}")
        # ax_singolo.grid(True, linestyle='--', alpha=0.5)
        # ax_singolo.legend(loc='lower left', fontsize=12)
        
        # plt.tight_layout()
        # plt.show()  # Display the plot before continuing to the next trial.


# ==========================================
    # FINAL GLOBAL SUMMARY (after the trial loop).
    # ==========================================
    print(f"\n--- RISULTATI FINALI ---")
    print(f"Successes: {risultati['Successes']} | Timeout: {risultati['Timeout']} | Crashes: {risultati['Crashes']}")
    
    print("\n" + "="*70)
    print("📊 RESOCONTO GLOBALE RECURSIVE FEASIBILITY (Traiettoria k in Box k+1)")
    print("="*70)
    for row in registro_violazioni_globale:
        print(f"Target {row['target']:02d} | Esito: {row['esito']:<9} | DistTot: {row['distanza']:<5.2f}m| DistX: {row['distanza_x']:<5.2f}m | Costo: {row['costo']:<8.2f} | Step Stop: {row['iterazione']:<4} | "
            f"Uscite dal box: {row['num_violazioni']:<3} | Correlazione: {row['correlazione']}")
    print("="*70 + "\n")

    # --- Calculate averages using successful trials only. ---
    costi_successi = [row['costo'] for row in registro_violazioni_globale if row['esito'] == "Successes"]
    dist_tot_successi = [row['distanza'] for row in registro_violazioni_globale if row['esito'] == "Successes"]
    dist_x_successi = [row['distanza_x'] for row in registro_violazioni_globale if row['esito'] == "Successes"]

    if len(costi_successi) > 0:
        print("="*70)
        print("📈 MEDIE CALCOLATE SOLO SUI RUN CON ESITO 'Successes'")
        print("="*70)
        print(f"Costo medio:         {np.mean(costi_successi):.2f}")
        print(f"Distanza Tot. media: {np.mean(dist_tot_successi):.2f} m")
        print(f"Distanza X media:    {np.mean(dist_x_successi):.2f} m")
        print("="*70 + "\n")
    else:
        print("="*70)
        print("⚠️ Nessun successo registrato. Impossibile calcolare le medie dei successi.")
        print("="*70 + "\n")
    # -------------------------------------------------

    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
    labels = list(risultati.keys())
    sizes = list(risultati.values())
    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.1, 0, 0) 
    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax_pie.set_title('Tasso di Successo Globale')
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    N_TESTS = 3
    
    main_statistico(N_TESTS)

    # anteprima_ambiente(num_rombi=0)  # Optional static preview of the generated map.
    