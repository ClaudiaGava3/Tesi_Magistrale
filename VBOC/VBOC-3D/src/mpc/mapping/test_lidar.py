import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
# 1. SIMULAZIONE LIDAR E SFERE TANGENTI (3D Sferico)
# =============================================================================
def get_lidar_hits_3d(drone_x, drone_y, drone_z, obstacles, num_rays=1000, max_range=1.5):
    """
    Simula un LiDAR 3D sparando raggi sferici usando la spirale di Fibonacci.
    Restituisce i punti d'impatto (hits) e il raggio della bolla d'ostacolo.
    """
    hits = []
    distances = []
    
    # Ripartizione uniforme su sfera con spirale di Fibonacci
    indices = np.arange(0, num_rays, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_rays)
    theta = np.pi * (1 + 5**0.5) * indices
    
    # Direzioni dei raggi
    dx = np.cos(theta) * np.sin(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(phi)
    
    # Angolo solido per calcolare il raggio approssimato della bolla
    # Area media di copertura = (4 * pi) / num_rays
    # r_bolla = r_distanza * sqrt(Area / pi) -> r_distanza * 2 / sqrt(num_rays)
    cone_angle = 2.0 / np.sqrt(num_rays)
    
    for i in range(num_rays):
        ray_dx, ray_dy, ray_dz = dx[i], dy[i], dz[i]
        ray_min_dist = max_range
        
        for obs in obstacles:
            x_min, x_max, y_min, y_max, z_min, z_max = obs
            
            # Intersezione Raggio-AABB 3D
            t_x1 = (x_min - drone_x) / (ray_dx + 1e-6)
            t_x2 = (x_max - drone_x) / (ray_dx + 1e-6)
            t_y1 = (y_min - drone_y) / (ray_dy + 1e-6)
            t_y2 = (y_max - drone_y) / (ray_dy + 1e-6)
            t_z1 = (z_min - drone_z) / (ray_dz + 1e-6)
            t_z2 = (z_max - drone_z) / (ray_dz + 1e-6)
            
            t_enter = max(min(t_x1, t_x2), min(t_y1, t_y2), min(t_z1, t_z2))
            t_exit = min(max(t_x1, t_x2), max(t_y1, t_y2), max(t_z1, t_z2))
            
            if t_enter <= t_exit and t_exit >= 0:
                dist = max(t_enter, 0)
                if dist < ray_min_dist:
                    ray_min_dist = dist
                    
        if ray_min_dist < max_range:
            # Salvataggio punti in 3D: [X, Y, Z]
            hits.append([
                drone_x + ray_dx * ray_min_dist, 
                drone_y + ray_dy * ray_min_dist,
                drone_z + ray_dz * ray_min_dist
            ])
            distances.append(ray_min_dist)
            
    hits = np.array(hits)
    distances = np.array(distances)
    
    # 5% margine sicurezza per evitare buchi
    if len(distances) > 0:
        radii = distances * cone_angle * 1.05
    else:
        radii = np.array([])
        
    return hits, radii


import numpy as np

def triangula_facce(facce_poligonali):
    """Scompone poligoni complanari (di N vertici) in una lista di triangoli 3D."""
    triangoli = []
    for faccia in facce_poligonali:
        p0 = faccia[0] # Vertice a ventaglio
        for i in range(1, len(faccia) - 1):
            triangoli.append((p0, faccia[i], faccia[i+1]))
    return np.array(triangoli)

def get_lidar_hits_3d_qualsiasi(drone_pos, facce_ostacoli, num_rays=2000, max_range=2.5):
    """
    Rilevamento ostacoli 3D per poligoni qualsiasi (Möller-Trumbore vettorializzato).
    drone_pos: [x, y, z]
    facce_ostacoli: lista di facce, dove ogni faccia è una lista di [x, y, z]
    """
    if not facce_ostacoli:
        return np.array([]), np.array([])
        
    # 1. Triangolazione automatica
    triangoli = triangula_facce(facce_ostacoli)
    A = triangoli[:, 0, :]
    B = triangoli[:, 1, :]
    C = triangoli[:, 2, :]
    
    # Vettori spigolo del triangolo
    E1 = B - A
    E2 = C - A
    
    # 2. Generazione Raggi Omogenei (Sfera di Fibonacci)
    indices = np.arange(0, num_rays, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_rays)
    theta = np.pi * (1 + 5**0.5) * indices
    dx = np.cos(theta) * np.sin(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(phi)
    rays = np.column_stack((dx, dy, dz))
    
    hits = []
    distances = []
    O = np.array(drone_pos)
    
    # 3. Intersezione Vettorializzata (Un raggio contro TUTTI i triangoli)
    for D in rays:
        P = np.cross(D, E2)
        det = np.sum(E1 * P, axis=1)
        
        # Escludiamo i triangoli paralleli al raggio
        valid_mask = np.abs(det) > 1e-8
        
        # Regola di Cramer
        inv_det = np.zeros_like(det)
        inv_det[valid_mask] = 1.0 / det[valid_mask]
        
        T_vec = O - A
        u = np.sum(T_vec * P, axis=1) * inv_det
        Q = np.cross(T_vec, E1)
        v = np.sum(D * Q, axis=1) * inv_det
        t = np.sum(E2 * Q, axis=1) * inv_det
        
        # Condizioni Fisiche: t>0 (davanti), u,v in [0,1] (dentro il triangolo)
        hit_mask = valid_mask & (u >= 0) & (v >= 0) & ((u + v) <= 1) & (t > 0)
        
        valid_t = t[hit_mask]
        if len(valid_t) > 0:
            min_t = np.min(valid_t)
            if min_t < max_range:
                hits.append(O + min_t * D)
                distances.append(min_t)
                
    hits = np.array(hits)
    distances = np.array(distances)
    
    # Raggio per AABB: calcoliamo l'angolo solido approssimato per raggio
    angle_step_rad = np.sqrt(4 * np.pi / num_rays)
    if len(distances) > 0:
        radii = distances * np.tan(angle_step_rad / 2) * 1.05
    else:
        radii = np.array([])
        
    return hits, radii



# =============================================================================
# 2. ALGORITMO DI MAX (Adattato al 3D)
# =============================================================================
def min_cube_select_3d(Q, R, target_rel_x, target_rel_y, target_rel_z, drone_radius=0.1, W=0, rel=0.1, lidar_ray=3.0):
    """
    Q: array Nx3 dei punti di intersezione
    R: array N dei raggi delle sfere
    """

    LIMIT = lidar_ray/np.sqrt(2)  

    if len(Q) == 0:
        return -LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT, 1
         
    # Box 3D: [xMin, xMax, yMin, yMax, zMin, zMax]
    box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    
    # Il drone deve starci dentro
    box[0] = min(box[0], -drone_radius); box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius); box[3] = max(box[3],  drone_radius)
    box[4] = min(box[4], -drone_radius); box[5] = max(box[5],  drone_radius)

    for _ in range(100):
        intersecting = _spheres_intersect_box_3d(Q, R, box)
        if not np.any(intersecting):
            break

        box, moved = _push_faces_3d(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_y, target_rel_z, W, rel, lidar_ray)
        if not moved:
            break

    exitflag = 1 if not np.any(_spheres_intersect_box_3d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], box[4], box[5], exitflag

def min_cube_warm_start_3d(Q, R, target_rel_x, target_rel_y, target_rel_z, targetx, targety, targetz, drone_radius=0.1, box_prev=None, expand_mode='general', W=0, rel=0.1, lidar_ray=3.0):
    
    LIMIT = lidar_ray/np.sqrt(2) 

    if box_prev is None:
        box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    else:
        box = np.array([
            max(-LIMIT, box_prev[0]), min( LIMIT, box_prev[1]), 
            max(-LIMIT, box_prev[2]), min( LIMIT, box_prev[3]),
            max(-LIMIT, box_prev[4]), min( LIMIT, box_prev[5])
        ], dtype=float)

    box[0] = min(box[0], -drone_radius); box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius); box[3] = max(box[3],  drone_radius)
    box[4] = min(box[4], -drone_radius); box[5] = max(box[5],  drone_radius)

    if len(Q) == 0:
        return -LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT, 1

    for _ in range(100):
        intersecting = _spheres_intersect_box_3d(Q, R, box)
        if not np.any(intersecting):
            break
        box, moved = _push_faces_3d(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_y, target_rel_z, W, rel, lidar_ray)
        if not moved:
            break

    if expand_mode == 'directional':
        box = _expand_faces_directional_3d(box, Q, R, target_rel_x, target_rel_y, target_rel_z, targetx, targety, targetz, LIMIT=LIMIT)
    # Aggiungi qui gli altri expand_mode se vuoi testarli (general, score)

    intersecting = _spheres_intersect_box_3d(Q, R, box)
    if np.any(intersecting):
        box, _ = _push_faces_3d(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_y, target_rel_z, W, rel, lidar_ray)

    exitflag = 1 if not np.any(_spheres_intersect_box_3d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], box[4], box[5], exitflag

def _spheres_intersect_box_3d(Q, R, box, tol=1e-6):
    cx = np.clip(Q[:, 0], box[0], box[1])
    cy = np.clip(Q[:, 1], box[2], box[3])
    cz = np.clip(Q[:, 2], box[4], box[5])
    dist2 = (Q[:, 0] - cx)**2 + (Q[:, 1] - cy)**2 + (Q[:, 2] - cz)**2
    return dist2 < (R**2 - tol)

def _push_faces_3d(box, Qi, Ri, drone_radius, target_rel_x, target_rel_y, target_rel_z, W=0, rel=0.1, lidar_ray=3.0):
    moved = False

    LIMIT = lidar_ray/np.sqrt(2)

    for i in range(len(Qi)):
        cx, cy, cz = Qi[i]
        r = Ri[i]
        candidates = []


        # Valuta di spingere i 6 bordi
        new_xMin = cx + r + 1e-4
        if -LIMIT <= new_xMin <= 0: candidates.append((0, new_xMin))
        new_xMax = cx - r - 1e-4
        if 0 <= new_xMax <= LIMIT:  candidates.append((1, new_xMax))

        new_yMin = cy + r + 1e-4
        if -LIMIT <= new_yMin <= 0: candidates.append((2, new_yMin))
        new_yMax = cy - r - 1e-4
        if 0 <= new_yMax <= LIMIT:  candidates.append((3, new_yMax))

        new_zMin = cz + r + 1e-4
        if -LIMIT <= new_zMin <= 0: candidates.append((4, new_zMin))
        new_zMax = cz - r - 1e-4
        if 0 <= new_zMax <= LIMIT:  candidates.append((5, new_zMax))

        if not candidates: continue

        best_score = -float('inf')
        best_face_idx = -1
        best_val = 0
        # W = 20 #20 per ostacoli paralleli, 0.2 per ostacoli obliqui

        for face_idx, val in candidates:
            test_box = box.copy()
            test_box[face_idx] = val
            
            # Calcola VOLUME
            volume = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2]) * (test_box[5] - test_box[4])
            
            # Bonus direzionale 3D
            # rel ora a 0.1 come 2D prima era 0.4
            if target_rel_x > rel:
                bonus_x = test_box[1]
            elif target_rel_x < -rel:
                bonus_x = - test_box[0]
            else:
                bonus_x = 0.0
            if target_rel_y > rel:
                bonus_y = test_box[3]
            elif target_rel_y < -rel:
                bonus_y = -test_box[2]
            else:
                bonus_y = 0.0                
            if target_rel_z > rel:
                bonus_z = test_box[5]
            elif target_rel_z < -rel:
                bonus_z = -test_box[4]
            else:
                bonus_z = 0.0

                
            score = volume + W * (bonus_x + bonus_y + bonus_z)
            
            if score > best_score:
                best_score = score
                best_face_idx = face_idx
                best_val = val

        new_box = box.copy()
        new_box[best_face_idx] = best_val

        # Assicurati che non stritoli il drone
        if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
                new_box[2] > -drone_radius or new_box[3] < drone_radius or
                new_box[4] > -drone_radius or new_box[5] < drone_radius):
            box = new_box
            moved = True

    return box, moved

def _expand_faces_directional_3d(box, Q, R, dx, dy, dz, targetx, targety, targetz, LIMIT=1.0):
    """
    Espansione direzionale 3D: Coerente col 2D. Usa dx, dy, dz per il verso,
    e targetx, targety, targetz per le priorità di espansione.
    """
    new_box = box.copy()

    if len(Q) == 0:
        return np.array([-LIMIT, LIMIT, -LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)

    # 1. Determina il verso in base al vettore predittivo
    x_first = 1 if dx >= 0 else 0; x_last = 0 if dx >= 0 else 1
    y_first = 3 if dy >= 0 else 2; y_last = 2 if dy >= 0 else 3
    z_first = 5 if dz >= 0 else 4; z_last = 4 if dz >= 0 else 5

    # 2. Ordina gli assi in base al target assoluto (come abs(targetx) >= abs(targetz) nel 2D)
    priorities = [
        (abs(targetx), x_first, x_last),
        (abs(targety), y_first, y_last),
        (abs(targetz), z_first, z_last)
    ]
    priorities.sort(key=lambda item: item[0], reverse=True) # Dal più grande al più piccolo

    # 3. L'array definisce l'ordine: prima le facce frontali prioritarie, poi le opposte
    ordine_facce = [
        priorities[0][1], priorities[1][1], priorities[2][1],
        priorities[0][2], priorities[1][2], priorities[2][2]
    ]
    
    # INTRODUZIONE DELLA TOLLERANZA (come nel 2D)
    TOL = 0.01

    # 4. Esegui l'espansione seguendo l'ordine
    for faccia in ordine_facce:
        if faccia == 1: # ESPANDI A DESTRA (+X)
            mask = (Q[:, 0] > new_box[1]) & \
                   (Q[:, 1] + R > new_box[2]+TOL) & (Q[:, 1] - R < new_box[3]-TOL) & \
                   (Q[:, 2] + R > new_box[4]+TOL) & (Q[:, 2] - R < new_box[5]-TOL)
            if np.any(mask):
                new_box[1] = min(LIMIT, np.min(Q[mask, 0] - R[mask]) - 1e-4)
            else:
                new_box[1] = LIMIT
                
        elif faccia == 0: # ESPANDI A SINISTRA (-X)
            mask = (Q[:, 0] < new_box[0]) & \
                   (Q[:, 1] + R > new_box[2]+TOL) & (Q[:, 1] - R < new_box[3]-TOL) & \
                   (Q[:, 2] + R > new_box[4]+TOL) & (Q[:, 2] - R < new_box[5]-TOL)
            if np.any(mask):
                new_box[0] = max(-LIMIT, np.max(Q[mask, 0] + R[mask]) + 1e-4)
            else:
                new_box[0] = -LIMIT
                
        elif faccia == 3: # ESPANDI IN AVANTI (+Y)
            mask = (Q[:, 1] > new_box[3]) & \
                   (Q[:, 0] + R > new_box[0]+TOL) & (Q[:, 0] - R < new_box[1]-TOL) & \
                   (Q[:, 2] + R > new_box[4]+TOL) & (Q[:, 2] - R < new_box[5]-TOL)
            if np.any(mask):
                new_box[3] = min(LIMIT, np.min(Q[mask, 1] - R[mask]) - 1e-4)
            else:
                new_box[3] = LIMIT

        elif faccia == 2: # ESPANDI INDIETRO (-Y)
            mask = (Q[:, 1] < new_box[2]) & \
                   (Q[:, 0] + R > new_box[0]+TOL) & (Q[:, 0] - R < new_box[1]-TOL) & \
                   (Q[:, 2] + R > new_box[4]+TOL) & (Q[:, 2] - R < new_box[5]-TOL)
            if np.any(mask):
                new_box[2] = max(-LIMIT, np.max(Q[mask, 1] + R[mask]) + 1e-4)
            else:
                new_box[2] = -LIMIT
                
        elif faccia == 5: # ESPANDI IN ALTO (+Z)
            mask = (Q[:, 2] > new_box[5]) & \
                   (Q[:, 0] + R > new_box[0]+TOL) & (Q[:, 0] - R < new_box[1]-TOL) & \
                   (Q[:, 1] + R > new_box[2]+TOL) & (Q[:, 1] - R < new_box[3]-TOL)
            if np.any(mask):
                new_box[5] = min(LIMIT, np.min(Q[mask, 2] - R[mask]) - 1e-4)
            else:
                new_box[5] = LIMIT
                
        elif faccia == 4: # ESPANDI IN BASSO (-Z)
            mask = (Q[:, 2] < new_box[4]) & \
                   (Q[:, 0] + R > new_box[0]+TOL) & (Q[:, 0] - R < new_box[1]-TOL) & \
                   (Q[:, 1] + R > new_box[2]+TOL) & (Q[:, 1] - R < new_box[3]-TOL)
            if np.any(mask):
                new_box[4] = max(-LIMIT, np.max(Q[mask, 2] + R[mask]) + 1e-4)
            else:
                new_box[4] = -LIMIT

    return new_box