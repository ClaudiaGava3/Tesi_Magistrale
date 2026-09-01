import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import matplotlib.animation as animation


from test_lidar import get_lidar_hits_3d_qualsiasi, min_cube_warm_start_3d
from parser import Parameters
from mpc_abstract_obs import Model
from mpc_controller_obs import MpcController






def visualizza_su_meshcat(facce_ostacoli, target_pos):
    vis = meshcat.Visualizer()

    # --- SPEGNI LA GRIGLIA E GLI ASSI DI DEFAULT ---
    vis["/Grid"].set_property("visible", False)
    vis["/Axes"].set_property("visible", False)
    # ------------------------------------------------

    print("\n[Meshcat] Apri questo link nel browser:")
    print(vis.url())
    
    mat_ostacolo = g.MeshLambertMaterial(color=0x808080, opacity=1.0)
    for idx, faccia in enumerate(facce_ostacoli):
        f = np.array(faccia, dtype=np.float32)
        if len(f) == 4:
            verts = f
            faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
            mesh_geom = g.TriangularMeshGeometry(vertices=verts, faces=faces)
            vis["ambiente"][f"obs_{idx}"].set_object(mesh_geom, mat_ostacolo)
        elif len(f) == 3:
            verts = f
            faces = np.array([[0, 1, 2]], dtype=np.int32)
            mesh_geom = g.TriangularMeshGeometry(vertices=verts, faces=faces)
            vis["ambiente"][f"obs_{idx}"].set_object(mesh_geom, mat_ostacolo)

    # Drone
    vis["drone"]["corpo"].set_object(
      g.ObjMeshGeometry.from_file('drone_costum.obj')
    )
    vis["drone"]["corpo"].set_transform(tf.scale_matrix(0.05))

    # Target a "X" rossa tridimensionale
    mat_target = g.MeshLambertMaterial(color=0xFF0000, opacity=1.0)
    c1 = g.Cylinder(height=0.4, radius=0.03)
    vis["target"]["barra1"].set_object(c1, mat_target)
    vis["target"]["barra1"].set_transform(tf.rotation_matrix(np.pi/4, [1, 1, 0]))
    
    c2 = g.Cylinder(height=0.4, radius=0.03)
    vis["target"]["barra2"].set_object(c2, mat_target)
    vis["target"]["barra2"].set_transform(tf.rotation_matrix(-np.pi/4, [1, -1, 0]))
    vis["target"].set_transform(tf.translation_matrix(target_pos[:3]))

    # ELIMINATA: vis["/Cameras/default"].set_object(...) -> Creava il conflitto!

    time.sleep(1.0)
    return vis

def aggiorna_drone_meshcat(vis, current_x, box_abs=None):
    pos = current_x[0:3]
    roll, pitch, yaw = current_x[3], current_x[4], current_x[5]
    
    # 1. Movimento fisico del drone (Rotazione + Traslazione)
    rot_matrix = tf.euler_matrix(roll, pitch, yaw)
    transform = rot_matrix.copy()
    transform[0:3, 3] = pos
    vis["drone"].set_transform(transform)
    
    # 2. TELECAMERA FOLLOW (Trasla col drone ma non ruota)
    # Spostando il nodo /Cameras/default, stiamo spostando il BERSAGLIO della telecamera.
    # Meshcat manterrà l'angolazione 3D di default, seguendo il drone passo passo!
    vis["/Cameras/default"].set_transform(tf.translation_matrix(pos))
    
    # 3. DISEGNA IL SAFE-BOX VERDE IN 3D
    if box_abs is not None:
        w = box_abs[1] - box_abs[0]
        d = box_abs[3] - box_abs[2]
        h = box_abs[5] - box_abs[4]
        
        cx = (box_abs[1] + box_abs[0]) / 2.0
        cy = (box_abs[3] + box_abs[2]) / 2.0
        cz = (box_abs[5] + box_abs[4]) / 2.0
        
        mat_box = g.MeshLambertMaterial(color=0x00FF00, opacity=0.15, transparent=True)
        # Scommentato: ora si vedrà nel video!
        # vis["safe_box"].set_object(g.Box([w, d, h]), mat_box)
        # vis["safe_box"].set_transform(tf.translation_matrix([cx, cy, cz]))


def genera_ambiente_3d_test():
    facce_ostacoli = []
    
    # Allarghiamo la stanza a 20x20x6 per far respirare il randomizzatore
    ROOM_X, ROOM_Y, ROOF = 20.0, 20.0, 6.0
    
    # 1. Pavimento e Soffitto
    facce_ostacoli.append([[0, 0, 0], [ROOM_X, 0, 0], [ROOM_X, ROOM_Y, 0], [0, ROOM_Y, 0]])
    facce_ostacoli.append([[0, 0, ROOF], [ROOM_X, 0, ROOF], [ROOM_X, ROOM_Y, ROOF], [0, ROOM_Y, ROOF]])
    
    # --- FUNZIONI GENERATRICI DEI SOLIDI (Facce perfettamente chiuse) ---
    def create_box(xmin, xmax, ymin, ymax, zmin, zmax):
        p0, p1, p2, p3 = [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin]
        p4, p5, p6, p7 = [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]
        facce = [
            [p0, p3, p2, p1], [p4, p5, p6, p7], [p0, p1, p5, p4],
            [p3, p7, p6, p2], [p0, p4, p7, p3], [p1, p2, p6, p5]
        ]
        return facce, [xmin, xmax, ymin, ymax, zmin, zmax]

    def create_pyramid(cx, cy, zmin, l, h):
        v_base = [[cx-l/2, cy-l/2, zmin], [cx+l/2, cy-l/2, zmin], [cx+l/2, cy+l/2, zmin], [cx-l/2, cy+l/2, zmin]]
        apice = [cx, cy, zmin + h]
        facce = [v_base]
        for i in range(4):
            facce.append([v_base[i], v_base[(i+1)%4], apice])
        return facce, [cx-l/2, cx+l/2, cy-l/2, cy+l/2, zmin, zmin+h]

    def create_hex_prism(cx, cy, zmin, zmax, r):
        base, top = [], []
        for i in range(6):
            ang = 2 * np.pi * i / 6
            base.append([cx + r * np.cos(ang), cy + r * np.sin(ang), zmin])
            top.append([cx + r * np.cos(ang), cy + r * np.sin(ang), zmax])
        facce = [base, top]
        for i in range(6):
            facce.append([base[i], base[(i+1)%6], top[(i+1)%6], top[i]])
        return facce, [cx-r, cx+r, cy-r, cy+r, zmin, zmax]

    # --- SETUP POSIZIONI ---
    start = np.array([2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Target randomico: lo forziamo a nascere nella metà opposta della stanza (X e Y tra 12 e 18)
    # e ad un'altezza variabile tra 1.0 e 4.5 metri.
    t_x = np.random.uniform(12.0, 18.0)
    t_y = np.random.uniform(12.0, 18.0)
    t_z = np.random.uniform(1.0, 4.5)
    targets = [np.array([t_x, t_y, t_z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    
    # "Safe Zones": AABB di partenza e arrivo, qui dentro non può spawnare nulla!
    occupied_aabbs = [
        [0.0, 4.5, 0.0, 4.5, 0.0, ROOF], 
        
        # 2. Colonna sicura attorno al target random 
        [t_x - 1.5, t_x + 1.5, t_y - 1.5, t_y + 1.5, 0.0, ROOF]
    ]
    
    def check_collision(new_aabb):
        margin = 0.5 # Mantiene almeno mezzo metro di distanza vuota tra ogni ostacolo
        nx_min, nx_max, ny_min, ny_max, nz_min, nz_max = new_aabb
        for ab in occupied_aabbs:
            x_overlap = not (nx_max + margin < ab[0] or nx_min - margin > ab[1])
            y_overlap = not (ny_max + margin < ab[2] or ny_min - margin > ab[3])
            z_overlap = not (nz_max + margin < ab[4] or nz_min - margin > ab[5])
            if x_overlap and y_overlap and z_overlap:
                return True # C'è collisione!
        return False

    # --- RICETTARIO OSTACOLI ---
    # Grandi (Appoggiati a terra: tipo, [dimensioni chiave])
    specs = [
        ('box', [3.0, 2.5, 4.0]),      # dx, dy, dz
        ('box', [2.5, 4.0, 3.0]),
        ('pyramid', [3.0, 3.5]),       # lato base, altezza
        ('hex', [2.0, 3.5]),           # raggio, altezza
    ]
    # Piccoli (Fluttuanti in aria per far fare lo slalom in verticale al drone)
    for _ in range(7):
        specs.append(('small_box', [1.0, 1.0, 1.0]))

    # --- GENERATORE RANDOM ---
    # np.random.seed(42) # Scommenta questa riga se vuoi che il labirinto sia sempre uguale a ogni lancio
    
    for tipo, params in specs:
        for _ in range(150): # Prova fino a 150 volte a trovare un "buco" libero per l'ostacolo
            cx = np.random.uniform(3, ROOM_X - 3)
            cy = np.random.uniform(3, ROOM_Y - 3)
            
            if tipo == 'box':
                dx, dy, dz = params
                xmin, xmax = cx - dx/2, cx + dx/2
                ymin, ymax = cy - dy/2, cy + dy/2
                faces, aabb = create_box(xmin, xmax, ymin, ymax, 0.0, dz)
                
            elif tipo == 'small_box':
                dx, dy, dz = params
                xmin, xmax = cx - dx/2, cx + dx/2
                ymin, ymax = cy - dy/2, cy + dy/2
                # Fluttuano tra 1.5m e 4.5m di altezza, non toccano né terra né soffitto
                zmin = np.random.uniform(2.5, ROOF - dz - 0.5) 
                faces, aabb = create_box(xmin, xmax, ymin, ymax, zmin, zmin + dz)
                
            elif tipo == 'pyramid':
                l, h = params
                faces, aabb = create_pyramid(cx, cy, 0.0, l, h)
                
            elif tipo == 'hex':
                r, h = params
                faces, aabb = create_hex_prism(cx, cy, 0.0, h, r)
            
            # Controlla se la posizione è valida e non sbatte
            if not check_collision(aabb):
                facce_ostacoli.extend(faces)
                occupied_aabbs.append(aabb)
                break # Posto trovato! Passa al prossimo ostacolo
    
    return facce_ostacoli, start, targets, ROOF


def disegna_scia_drone(vis, x_history):
    """Disegna la linea della scia percorsa dal drone su Meshcat."""
    # Estrae le coordinate [x, y, z] dalla cronologia degli stati e le traspone (3xN)
    punti_scia = np.array([stato[:3] for stato in x_history], dtype=np.float32).T
    
    # Crea una linea luminosa (es. Azzurra/Cyan)
    materiale_linea = g.LineBasicMaterial(color=0x00FFFF, linewidth=5)
    vis["scia"].set_object(g.Line(g.PointsGeometry(punti_scia), materiale_linea))



def draw_3d_box(ax, box_abs, color='lime', alpha=0.3):
    """Disegna l'AABB 3D."""
    x_min, x_max, y_min, y_max, z_min, z_max = box_abs
    
    # Vertici del box
    v = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    
    facce = [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], # Basso, Alto
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]], # Davanti, Dietro
        [v[1], v[2], v[6], v[5]], [v[4], v[7], v[3], v[0]]  # Destra, Sinistra
    ]
    ax.add_collection3d(Poly3DCollection(facce, facecolors=color, linewidths=1, edgecolors=color, alpha=alpha))

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
    u_hover = (model.mass * 9.81) / (4.0 * model.cf)

    DT = params.dt
    SIM_TIME = 40.0 # Extended time to cover all targets
    N_SIM = int(SIM_TIME / DT)
    
    facce, start, targets, roof = genera_ambiente_3d_test()
    N_TESTS = len(targets)

    vis = visualizza_su_meshcat(facce, targets[0])

    
    
    for test_idx, target_base in enumerate(targets):
        # Punto di partenza fisso
        current_x = start.copy()
        
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


        iterazioni_violazione_box = []
        frames_animazione = []

        for t in range(N_SIM):

            # 0. Recovery timer management
            if in_recovery:
                x_ref_attuale = target_recovery
                timer_recovery -= 1
                if timer_recovery <= 0:
                    in_recovery = False
                    
                    # Restore constraints
                    controller.ocp_solver.constraints_set(controller.N, "lh", np.zeros(6))
                    
                    lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                    ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")

                    lbx_e_curr[6:] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
                    ubx_e_curr[6:] = [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]

                    controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                    controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                    
                    current_target = target_base.copy() 
                    x_ref_attuale = current_target.copy()
            else:
                # Restore the base target if it was changed by a previous stall
                if contatore_stallo == 0:
                    x_ref_attuale = current_target.copy()

            pos_3d = current_x[0:3]

            hits, radii = get_lidar_hits_3d_qualsiasi(pos_3d, facce, num_rays=1000, max_range=3.0)
            
            Q_rel = hits.copy()
            if len(hits) > 0:
                Q_rel[:, 0] -= current_x[0]
                Q_rel[:, 1] -= current_x[1]
                Q_rel[:, 2] -= current_x[2]
                
            # --- Direction calculation (final target) (Case 1) ---
            target_rel_x = x_ref_attuale[0] - current_x[0]
            target_rel_y = x_ref_attuale[1] - current_x[1]
            target_rel_z = x_ref_attuale[2] - current_x[2]

            dx, dy, dz = target_rel_x, target_rel_y, target_rel_z


  

            # --- Direction calculation: use predicted trajectory (Case 6) ---
            if x_sol_prev is not None and len(x_sol_prev) > 5:
                
                dx = x_sol_prev[5][0] - current_x[0]
                dy = x_sol_prev[5][1] - current_x[1]
                dz = x_sol_prev[5][2] - current_x[2]
              

            else:
                # Al primissimo passo (t=0)
                dx = target_rel_x
                dy = target_rel_y
                dz = target_rel_z


            # ==========================================
            # WARM START SAFE BOX (MODIFICATO)
            # ==========================================
            if x_sol_prev is not None:
                # 1. Calcola il bounding box (min e max) dell'INTERA traiettoria predetta al passo precedente
                traj_xmin = np.min(x_sol_prev[:, 0])
                traj_xmax = np.max(x_sol_prev[:, 0])
                traj_ymin = np.min(x_sol_prev[:, 1])
                traj_ymax = np.max(x_sol_prev[:, 1])
                traj_zmin = np.min(x_sol_prev[:, 2])
                traj_zmax = np.max(x_sol_prev[:, 2])

                # 2. Converti questo Bounding Box in coordinate RELATIVE rispetto al drone attuale (current_x)
                margin = 0.35
                box_prev_rel = [
                    traj_xmin - current_x[0] - margin,
                    traj_xmax - current_x[0] + margin,
                    traj_ymin - current_x[1] - margin,
                    traj_ymax - current_x[1] + margin,
                    traj_zmin - current_x[2] - margin,
                    traj_zmax - current_x[2] + margin
                ]
            else:
                box_prev_rel = None
            # ==========================================


            # warm start
            xMin_r, xMax_r, yMin_r, yMax_r, zMin_r, zMax_r, _ = min_cube_warm_start_3d(
                Q_rel, radii, dx, dy, dz, target_rel_x, target_rel_y, target_rel_z, drone_radius=0.1, box_prev=box_prev_rel, 
                expand_mode='directional',  # 'general', 'directional or 'score'
                W=50, rel=0.1, lidar_ray=3.0
            )

            
            box_abs = np.array([
                xMin_r + current_x[0], xMax_r + current_x[0], 
                yMin_r + current_x[1], yMax_r + current_x[1],
                zMin_r + current_x[2], zMax_r + current_x[2]
            ])
            box_history.append(box_abs.copy())


            # # ==========================================
            # # CHECK RECURSIVE FEASIBILITY
            # # Check if the trajectory at step K (x_sol_prev) is inside the box just generated at step K+1 (box_abs)
            # # ==========================================
            # if x_sol_prev is not None:
            #     is_outside = False
            #     for p in x_sol_prev:
            #         if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or 
            #             p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3 or
            #             p[2] < box_abs[4]-1e-3 or p[2] > box_abs[5]+1e-3):
            #             is_outside = True
            #             break
                
            #     if is_outside:
            #         iterazioni_violazione_box.append(t)
            # # ==========================================


            # 2. MPC Solve
            x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)


            # ==========================================
            # DIAGNOSTIC BLOCK (MPC DEBUGGER)
            # ==========================================
            if t % 10 == 0: # Print every 10 steps
                print(f"\n--- DEBUG STEP {t} ---")
                print(f"1. Drone Position : X={current_x[0]:.2f}, Y={current_x[1]:.2f}, Z={current_x[2]:.2f}")
                print(f"2. Green Box (AABB): X in [{box_abs[0]:.2f}, {box_abs[1]:.2f}] | Y in [{box_abs[2]:.2f}, {box_abs[3]:.2f}] | Z in [{box_abs[4]:.2f}, {box_abs[5]:.2f}]")
                print(f"3. Local Target  : X={x_ref_attuale[0]:.2f}, Y={x_ref_attuale[1]:.2f}, Z={x_ref_attuale[2]:.2f}")


                # Compute distance between drone and local target
                dist_to_local = np.linalg.norm(current_x[:3] - np.array([target_rel_x, target_rel_y, target_rel_z])[:3])
                print(f"5. Distance to local target: {dist_to_local:.3f} meters")
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
                    center_y = (box_sicuro[2] + box_sicuro[3]) / 2.0
                    center_z = (box_sicuro[4] + box_sicuro[5]) / 2.0
                    
                    target_recovery = target_recovery = np.array([center_x, center_y, center_z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    x_ref_attuale = target_recovery
                    
                    in_recovery = True
                    timer_recovery = 40 
                    
                    # Relax constraints
                    controller.ocp_solver.constraints_set(controller.N, "lh", np.full(6, -alpha_curr))
                   
                    lbx_e_curr = controller.ocp_solver.constraints_get(controller.N, "lbx")
                    ubx_e_curr = controller.ocp_solver.constraints_get(controller.N, "ubx")
                    lbx_e_curr[6:] = [-5.0, -5.0, -5.0,-5.0, -5.0, -5.0]
                    ubx_e_curr[6:] = [ 5.0,  5.0,  5.0, 5.0, 5.0, 5.0]
                    controller.ocp_solver.constraints_set(controller.N, "lbx", lbx_e_curr)
                    controller.ocp_solver.constraints_set(controller.N, "ubx", ubx_e_curr)
                    
                    x_sol, u_sol, alpha_curr, status = controller.solve_step(current_x, x_ref_attuale, box_abs)
                    
                    if status in [3, 4]:
                        esito = "Crashes"
                        in_recovery = False

                        # --- INIZIO CHECK AUTOPSIA CRASH ---
                        if x_sol_prev is not None:
                            punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3 or p[2] < box_abs[4]-1e-3 or p[2] > box_abs[5]+1e-3))
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
                        punti_fuori = sum(1 for p in x_sol_prev if (p[0] < box_abs[0]-1e-3 or p[0] > box_abs[1]+1e-3 or p[1] < box_abs[2]-1e-3 or p[1] > box_abs[3]+1e-3 or p[2] < box_abs[4]-1e-3 or p[2] > box_abs[5]+1e-3))
                        print(f"💀 AUTOPSIA CRASH (Step {t}): {punti_fuori}/{len(x_sol_prev)} punti della traiettoria precedente erano finiti FUORI dal box fatale!")

                        if punti_fuori > 0:
                                crashes_per_feasibility += 1
                    break

            if u_sol is not None:
                u_history.append(u_sol.copy())

            current_x = x_sol[1]
            x_history.append(current_x.copy())

            aggiorna_drone_meshcat(vis, current_x, box_abs)
            time.sleep(0.03) # Rallenta leggermente per godersi l'animazione nel browser

            # Cattura il frame dal browser (rallenterà leggermente la simulazione)
            frame = vis.get_image()
            if frame is not None:
                frames_animazione.append(np.asarray(frame))

            x_sol_prev = x_sol.copy()
            box_abs_prev = box_abs.copy()


            
            # ==========================================
            # STALL HANDLING AND RETURN TO BASE TARGET
            # ==========================================
            # 1. ABSOLUTE SUCCESS CHECK (end test with Success)
            dist_target_reale = np.linalg.norm(current_x[:3] - target_base[:3])
            if dist_target_reale < 0.3:
                esito = "Successes"
                break
                            
            if t > 0:
                # 2. NORMAL STALL HANDLING
                spostamento = np.linalg.norm(current_x[:3] - x_history[-2][:3])
                if spostamento < 0.01:
                    contatore_stallo += 1
                else:
                    contatore_stallo = 0 

                if contatore_stallo >= MAX_STALLO_ITER:
                    print(f"\n⚠️ STALL DETECTED (Step {t})! Perturb the target upward.")
                    
                    Z_MAX_SICURA = roof - 0.2  # Margine di sicurezza sotto il soffitto
                    current_target[2] = min(current_target[2] + 0.20, Z_MAX_SICURA)
                    x_ref_attuale = current_target.copy()
                    contatore_stallo = 0

                # 3. LOCAL TARGET ARRIVAL CHECK
                dist_al_locale = np.linalg.norm(current_x[:3] - current_target[:3])
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


        # # ==========================================
        # # FINAL REPORT OF THE SINGLE TARGET
        # # ==========================================
        # print(f"\n[{'='*40}]")
        # print(f"🎯 REPORT TARGET {test_idx+1}/{N_TESTS}")
        # print(f"Esito finale: {esito} (Iterazione di stop: {t})")
        # print(f"Recovery attivate: {recovery_in_questo_test}")
        
        # if len(iterazioni_violazione_box) > 0:
        #     print(f"⚠️ PROBLEMA: La traiettoria k è uscita dal box k+1 per {len(iterazioni_violazione_box)} volte.")
        #     print(f"Iterazioni esatte in cui è successo: {iterazioni_violazione_box}")
            
        #     # Correlation Analysis:
        #     if esito == "Crashes":
        #         if iterazioni_violazione_box[-1] >= t - 5:
        #             print("--> 🔴 FORTE CORRELAZIONE: L'ultima uscita dal box è avvenuta a ridosso dello schianto!")
        #         else:
        #             print("--> 🟠 CORRELAZIONE DEBOLE: Il drone è uscito dal box in passato, ma si è schiantato molto dopo.")
        # else:
        #     print("✅ OTTIMO: La traiettoria predetta è rimasta SEMPRE all'interno del box al passo k+1.")
        # print(f"[{'='*40}]\n")


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


    # ==========================================
    # END OF VIDEO
    # ==========================================
    # ==========================================
    # ESPORTAZIONE E SALVATAGGIO DEL VIDEO
    # ==========================================
        

    if len(frames_animazione) > 0:
        cartella_video = "video_traiettorie_telecamera"
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
        nome_video = os.path.join(cartella_video, f"Video_Target_N10_{test_idx+7:02d}_{esito}.mp4")
            
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
                

        # ==========================================
        # ==========================================

    return facce, tutte_le_traiettorie, vis



if __name__ == "__main__":
    facce_ostacoli, traiettorie, vis = main_statistico()

    if len(traiettorie) > 0:
        traj_completa, esito = traiettorie[0]
        # vis è l'istanza del visualizzatore creata nel main
        disegna_scia_drone(vis, traj_completa)
    
