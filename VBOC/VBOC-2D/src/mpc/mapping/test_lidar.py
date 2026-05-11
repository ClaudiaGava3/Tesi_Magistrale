import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

# =============================================================================
# 1. SIMULAZIONE LIDAR E SFERE TANGENTI
# =============================================================================
def get_lidar_hits_2d(drone_x, drone_z, obstacles, num_rays=360, max_range=5.0):
    hits = []
    distances = []
    
    # Calcolo dell'angolo tra un raggio e l'altro
    angle_step_rad = (2 * np.pi) / num_rays
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    for angle in angles:
        dx, dz = np.cos(angle), np.sin(angle)
        ray_min_dist = max_range
        
        for obs in obstacles:
            x_min, x_max, z_min, z_max = obs
            
            t_x1 = (x_min - drone_x) / (dx + 1e-6)
            t_x2 = (x_max - drone_x) / (dx + 1e-6)
            t_z1 = (z_min - drone_z) / (dz + 1e-6)
            t_z2 = (z_max - drone_z) / (dz + 1e-6)
            
            t_enter = max(min(t_x1, t_x2), min(t_z1, t_z2))
            t_exit = min(max(t_x1, t_x2), max(t_z1, t_z2))
            
            if t_enter <= t_exit and t_exit >= 0:
                dist = max(t_enter, 0)
                if dist < ray_min_dist:
                    ray_min_dist = dist
                    
        if ray_min_dist < max_range:
            # Salvataggio punti in 2D: [X, Z]
            hits.append([drone_x + dx * ray_min_dist, drone_z + dz * ray_min_dist])
            distances.append(ray_min_dist)
            
    hits = np.array(hits)
    distances = np.array(distances)
    
    # Raggio = distanza * tan(angolo/2). Si aggiunge un 5% (1.05) di sicurezza 
    # per far compenetrare leggermente le sfere ed evitare "buchi" numerici.
    if len(distances) > 0:
        radii = distances * np.tan(angle_step_rad / 2) * 1.05
    else:
        radii = np.array([])
        
    return hits, radii


# =============================================================================
# 2. ALGORITMO DI MAX (Adattato al 2D)
# =============================================================================
# def min_cube_select_2d(Q, R, drone_radius=0.1):
#     """
#     Q: array Nx2 dei punti di intersezione
#     R: array N dei raggi delle sfere
#     """
#     if len(Q) == 0:
#         # Se non c'è nulla, il box è il massimo possibile
#         return -5.0, 5.0, -5.0, 5.0, 1
        
#     LIMIT = 5.0 
#     # Box 2D: [xMin, xMax, zMin, zMax]
#     box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    
#     # Il drone deve starci dentro
#     box[0] = min(box[0], -drone_radius)
#     box[1] = max(box[1],  drone_radius)
#     box[2] = min(box[2], -drone_radius)
#     box[3] = max(box[3],  drone_radius)

#     for _ in range(100):
#         intersecting = _spheres_intersect_box_2d(Q, R, box)
#         if not np.any(intersecting):
#             break

#         box, moved = _push_faces_2d(box, Q[intersecting], R[intersecting], drone_radius)
#         if not moved:
#             break

#     exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
#     return box[0], box[1], box[2], box[3], exitflag

def min_cube_select_2d(Q, R, target_rel_x, target_rel_z, drone_radius=0.1):
    """
    Q: array Nx2 dei punti di intersezione (ostacoli visti dal lidar, coordinate relative)
    R: array N dei raggi delle sfere
    target_rel_x, target_rel_z: posizione del target relativa al drone!
    """
    if len(Q) == 0:
        # Se non c'è nulla, il box è il massimo possibile
        #return -1.0, 1.0, -1.0, 1.0, 1
        return -5.0, 5.0, -5.0, 5.0, 1
        
    LIMIT = 5.0 
    # Box 2D: [xMin, xMax, zMin, zMax]
    box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    
    # Il drone deve starci dentro
    box[0] = min(box[0], -drone_radius)
    box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius)
    box[3] = max(box[3],  drone_radius)

    for _ in range(100):
        intersecting = _spheres_intersect_box_2d(Q, R, box)
        if not np.any(intersecting):
            break

        # Passiamo target_rel_x e target_rel_z alla funzione di spinta
        box, moved = _push_faces_2d(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z)
        if not moved:
            break

    exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], exitflag

def _spheres_intersect_box_2d(Q, R, box, tol=1e-6):
    cx = np.clip(Q[:, 0], box[0], box[1])
    cz = np.clip(Q[:, 1], box[2], box[3])
    dist2 = (Q[:, 0] - cx)**2 + (Q[:, 1] - cz)**2
    return dist2 < (R**2 - tol)

# def _push_faces_2d(box, Qi, Ri, drone_radius):
#     xMin, xMax, zMin, zMax = box
#     moved = False

#     for i in range(len(Qi)):
#         cx, cz = Qi[i]
#         r = Ri[i]
#         candidates = []

#         # Valuta di spingere i 4 bordi
#         new_xMin = cx + r + 1e-4
#         if -5.0 <= new_xMin <= 0:
#             candidates.append((0, new_xMin, (xMax - new_xMin) * (zMax - zMin)))

#         new_xMax = cx - r - 1e-4
#         if 0 <= new_xMax <= 5.0:
#             candidates.append((1, new_xMax, (new_xMax - xMin) * (zMax - zMin)))

#         new_zMin = cz + r + 1e-4
#         if -5.0 <= new_zMin <= 0:
#             candidates.append((2, new_zMin, (xMax - xMin) * (zMax - new_zMin)))

#         new_zMax = cz - r - 1e-4
#         if 0 <= new_zMax <= 5.0:
#             candidates.append((3, new_zMax, (xMax - xMin) * (new_zMax - zMin)))

#         if not candidates:
#             continue

#         # Scegli la faccia che lascia l'AREA maggiore
#         face_idx, val, _ = max(candidates, key=lambda c: c[2])
#         new_box = box.copy()
#         new_box[face_idx] = val

#         # Assicurati che non stritoli il drone
#         if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
#                 new_box[2] > -drone_radius or new_box[3] < drone_radius):
#             box = new_box
#             moved = True

#     return box, moved

def _push_faces_2d(box, Qi, Ri, drone_radius, target_rel_x, target_rel_z):
    xMin, xMax, zMin, zMax = box
    moved = False

    for i in range(len(Qi)):
        cx, cz = Qi[i]
        r = Ri[i]
        candidates = []

        # Valuta di spingere i 4 bordi
        new_xMin = cx + r + 1e-4
        if -5.0 <= new_xMin <= 0:
            candidates.append((0, new_xMin))

        new_xMax = cx - r - 1e-4
        if 0 <= new_xMax <= 5.0:
            candidates.append((1, new_xMax))

        new_zMin = cz + r + 1e-4
        if -5.0 <= new_zMin <= 0:
            candidates.append((2, new_zMin))

        new_zMax = cz - r - 1e-4
        if 0 <= new_zMax <= 5.0:
            candidates.append((3, new_zMax))

        if not candidates:
            continue

        # --- NUOVA LOGICA: Punteggio = Area + Bonus Direzionale ---
        best_score = -float('inf')
        best_face_idx = -1
        best_val = 0
        
        # W è il "Peso" dell'attrazione verso il target.
        # Se 0.0 -> Torna ad essere l'algoritmo originale di Max.
        # Se troppo alto -> Ignora l'area e fa box strettissimi lunghi verso il target.
        # 15.0 o 20.0 di solito è un buon compromesso.
        W = 15.0 

        for face_idx, val in candidates:
            # Creiamo un box fittizio per calcolare come sarebbe l'area
            test_box = box.copy()
            test_box[face_idx] = val
            
            # 1. Calcola l'Area del box candidato
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])
            
            # 2. Calcola il Bonus Direzionale
            # Se il target è a destra (X positiva), premiamo test_box[1] (x_max)
            # Se il target è a sinistra (X negativa), premiamo quanto test_box[0] (x_min) va verso sinistra
            if target_rel_x > 0.5:
                bonus_x = test_box[1]
            elif target_rel_x < -0.5:
                bonus_x = - test_box[0]
            else:
                bonus_x = 0.0
                
            if target_rel_z > 0.5:
                bonus_z = test_box[3]
            elif target_rel_z < -0.5:
                bonus_z = -test_box[2]
            else:
                bonus_z = 0.0

                
            # 3. Punteggio totale
            score = area + W * (bonus_x + bonus_z)
            
            if score > best_score:
                best_score = score
                best_face_idx = face_idx
                best_val = val

        # Applica la scelta migliore
        new_box = box.copy()
        new_box[best_face_idx] = best_val

        # Assicurati che non stritoli il drone
        if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
                new_box[2] > -drone_radius or new_box[3] < drone_radius):
            box = new_box
            moved = True
        

    return box, moved


# =============================================================================
# 3. TEST E PLOTTING
# =============================================================================
def run_test_and_plot():
    # Stanza 3 (Pareti Sfalsate)
    obstacles = [
        [1.0, 2.0, -1.0, 3.0],   # Muro alto
        [3.0, 4.0, -5.0, -1.5]   # Muro basso
    ]
    
    drone_x, drone_z = 4.0, -1.0
    
    # 1. Raggi e Sfere tangenti
    hits, radii = get_lidar_hits_2d(drone_x, drone_z, obstacles, num_rays=360)
    
    # 2. Algoritmo di Max (Coordinate relative)
    Q_relative = hits.copy()
    Q_relative[:, 0] -= drone_x
    Q_relative[:, 1] -= drone_z
    
    xMin, xMax, zMin, zMax, status = min_cube_select_2d(Q_relative, radii, drone_radius=0.1)
    
    # Coordinate assolute per il plot
    box_abs = [xMin + drone_x, xMax + drone_x, zMin + drone_z, zMax + drone_z]
    
    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Ostacoli
    for obs in obstacles:
        ax.add_patch(patches.Rectangle((obs[0], obs[2]), obs[1]-obs[0], obs[3]-obs[2], color='gray', alpha=0.5))
        
    # Punti LiDAR e cerchi (esattamente come nel tuo disegno)
    if len(hits) > 0:
        ax.scatter(hits[:, 0], hits[:, 1], color='red', s=10, label='Lidar Hits')
        for i in range(len(hits)):
            circle = plt.Circle((hits[i, 0], hits[i, 1]), radii[i], color='red', alpha=0.2)
            ax.add_patch(circle)

    # Box Asimmetrico (Rosso nei tuoi disegni, qui Verde)
    box_w = box_abs[1] - box_abs[0]
    box_h = box_abs[3] - box_abs[2]
    ax.add_patch(patches.Rectangle((box_abs[0], box_abs[2]), box_w, box_h, 
                                   linewidth=3, edgecolor='green', facecolor='lime', alpha=0.3, label='Box Asymmetric'))
    
    # Drone
    ax.scatter(drone_x, drone_z, color='blue', s=100, label='Drone')

    ax.set_aspect('equal')
    ax.set_xlim([-1, 6])
    ax.set_ylim([-4, 4])
    ax.grid(True, linestyle=':')
    #ax.legend(loc='upper right')
    plt.title("Asymmetric Box with Tangent Spheres (Lidar)")
    plt.show()
    
if __name__ == '__main__':
    run_test_and_plot()