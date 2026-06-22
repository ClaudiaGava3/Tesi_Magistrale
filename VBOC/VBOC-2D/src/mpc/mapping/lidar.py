import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for all other text
})

# ============================================================================
# 1. SIMULATION LIDAR AND TANGENT SPHERES
# ============================================================================

# FOR PARALLEL OBSTACLES
def get_lidar_hits_2d(drone_x, drone_z, obstacles, num_rays=360, max_range=1.5):
    hits = []
    distances = []
    
    # Compute the angle between rays
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
            # Save 2D points: [X, Z]
            hits.append([drone_x + dx * ray_min_dist, drone_z + dz * ray_min_dist])
            distances.append(ray_min_dist)
            
    hits = np.array(hits)
    distances = np.array(distances)
    
    # Radius = distance * tan(angle/2). Add a 5% safety margin to allow spheres to overlap slightly and avoid numerical gaps.
    if len(distances) > 0:
        radii = distances * np.tan(angle_step_rad / 2) * 1.05
    else:
        radii = np.array([])
        
    return hits, radii

# FOR ARBITRARY OBSTACLES
def get_lidar_hits_2d_qualsiasi(drone_x, drone_z, segments, num_rays=360, max_range=1.5):
    hits = []
    distances = []
    
    # Compute the angle between rays
    angle_step_rad = (2 * np.pi) / num_rays
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    for angle in angles:
        dx, dz = np.cos(angle), np.sin(angle)
        ray_min_dist = max_range
        
        for seg in segments:
            # Extract the wall endpoints A and B
            (x_A, z_A), (x_B, z_B) = seg

            # Wall direction vector (from A to B)
            s_x = x_B - x_A
            s_z = z_B - z_A
            

            # 1. Compute the denominator (2D cross product)
            den = dx * s_z - dz * s_x

            # If the denominator is 0, the ray and wall are perfectly parallel
            if den != 0:
                # Distance vector from drone to wall start point A
                diff_x = x_A - drone_x
                diff_z = z_A - drone_z

                # 2. Solve the linear system (Cramer's rule)
                # t = distance along the drone ray
                t = (diff_x * s_z - diff_z * s_x) / den
                
                # u = impact point position along the wall segment
                u = (diff_x * dz - diff_z * dx) / den
                
                # 3. Physical impact condition
                # The ray moves forward (t > 0) and physically hits the wall segment (0 <= u <= 1)
                if t > 0 and 0 <= u <= 1:
                    # 4. Find the nearest wall
                    if t < ray_min_dist:
                        ray_min_dist = t

            
        # If the ray hits something within max_range, compute the exact impact point
        if ray_min_dist < max_range:
            hit_x = drone_x + ray_min_dist * dx
            hit_z = drone_z + ray_min_dist * dz
            hits.append([hit_x, hit_z])
            distances.append(ray_min_dist)
            
    hits = np.array(hits)
    distances = np.array(distances)
    
    # Radius = distance * tan(angle/2). Add a 5% (1.05) safety margin 
    # to allow the spheres to overlap slightly and avoid numerical "gaps"
    if len(distances) > 0:
        radii = distances * np.tan(angle_step_rad / 2) * 1.05
    else:
        radii = np.array([])
        
    return hits, radii

# =============================================================================
# 2. ORIGINAL MAX ALGORITHM (Adapted to 2D)
# =============================================================================
# def min_cube_select_2d(Q, R, drone_radius=0.1):
#     """
#     Q: array Nx2 of intersection points
#     R: array N of sphere radii
#     """
#     if len(Q) == 0:
#         # If nothing is present, the box is the largest possible
#         return -5.0, 5.0, -5.0, 5.0, 1
        
#     LIMIT = 5.0 
#     # 2D box: [xMin, xMax, zMin, zMax]
#     box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    
#     # The drone must fit inside
#     box[0] = min(box[0], -drone_radius)
#     box[1] = max(box[1],  drone_radius)
#     box[2] = min(box[2], -drone_radius)
#     box[3] = max(box[3],  drone_radius)

#     for _ in range(100):
#         intersecting = _spheres_intersect_box_2d(Q, R, box)
#         if not np.any(intersecting):
#             break

#         box, moved = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius)
#         if not moved:
#             break

#     exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
#     return box[0], box[1], box[2], box[3], exitflag

def min_cube_select_base(Q, R, target_rel_x, target_rel_z, drone_radius=0.1):
    """
    Q: array Nx2 of intersection points (obstacles seen by the lidar, relative coordinates)
    R: array N of sphere radii
    target_rel_x, target_rel_z: target position relative to the drone
    """

    LIMIT = 1.0

    if len(Q) == 0:
        # If nothing is present, return the largest possible box
        return -LIMIT, LIMIT, -LIMIT, LIMIT, 1
        
    
    # 2D box: [xMin, xMax, zMin, zMax]
    box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    
    # Ensure the drone fits inside
    box[0] = min(box[0], -drone_radius)
    box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius)
    box[3] = max(box[3],  drone_radius)

    for _ in range(100):
        intersecting = _spheres_intersect_box_2d(Q, R, box)
        if not np.any(intersecting):
            break


        # CHOOSE BETWEEN 2 MODES FOR PUSHING BOX FACES

        # push with bonus
        box, moved = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z)

        # push with penalty
        # box, moved = _push_faces_malus(box, Q[intersecting], R
        # [intersecting], drone_radius, target_rel_x, target_rel_z)

        if not moved:
            break

    exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], exitflag

def min_cube_select_directional(Q, R, target_rel_x, target_rel_z, drone_radius=0.1):
    LIMIT = 1.0
    MARGIN = 0.15 #margin beyond the projection

    # 1. Box initialization
    xMin_init = -LIMIT
    xMax_init =  LIMIT
    zMin_init = -LIMIT
    zMax_init =  LIMIT

    locked_faces = []

    # 2. "Projection-controlled faces"
    # If we go right, we lock the right face to the right projection
    if target_rel_x > 0:
        xMax_init = min(LIMIT, target_rel_x + drone_radius + MARGIN)
        locked_faces.append(1)
    elif target_rel_x < 0:
        xMin_init = max(-LIMIT, target_rel_x - drone_radius - MARGIN)
        locked_faces.append(0)

    # If we go up, we lock the top face to the top projection
    if target_rel_z > 0:
        zMax_init = min(LIMIT, target_rel_z + drone_radius + MARGIN)
        locked_faces.append(3)
    elif target_rel_z < 0:
        zMin_init = max(-LIMIT, target_rel_z - drone_radius - MARGIN)
        locked_faces.append(2)

    box = np.array([xMin_init, xMax_init, zMin_init, zMax_init], dtype=float)

    if len(Q) == 0:
        # If nothing is present, return the largest possible box
        return -LIMIT, LIMIT, -LIMIT, LIMIT, 1

    # Ensure the drone fits inside
    box[0] = min(box[0], -drone_radius)
    box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius)
    box[3] = max(box[3],  drone_radius)

    for _ in range(100):
        intersecting = _spheres_intersect_box_2d(Q, R, box)
        if not np.any(intersecting):
            break


        # CHOOSE BETWEEN 3 MODES FOR PUSHING BOX FACES

        # push with bonus
        box, moved = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z)

        # push with penalty
        # box, moved = _push_faces_malus(box, Q[intersecting], R
        # [intersecting], drone_radius, target_rel_x, target_rel_z)

        # push not blocked faces
        # box, moved = _push_faces_notblocked(box, Q[intersecting], R
        # [intersecting], drone_radius, target_rel_x, target_rel_z, locked_faces)

        if not moved:
            break


    # 2 MODES FOR BOX EXPANSION
    #box = _expand_faces(box, Q, R, LIMIT=1.0)
    box = _expand_faces_directional(box, Q, R, target_rel_x, target_rel_z, LIMIT=1.0)


    intersecting = _spheres_intersect_box_2d(Q, R, box)
    if np.any(intersecting):
        box, _ = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z)
        
    exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], exitflag


def _spheres_intersect_box_2d(Q, R, box, tol=1e-6):
    cx = np.clip(Q[:, 0], box[0], box[1])
    cz = np.clip(Q[:, 1], box[2], box[3])
    dist2 = (Q[:, 0] - cx)**2 + (Q[:, 1] - cz)**2
    return dist2 < (R**2 - tol)


# =============================================================================
# 2. ORIGINAL MAX ALGORITHM (Adapted to 2D)
# =============================================================================
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

#         # Choose the face that leaves the largest AREA
#         face_idx, val, _ = max(candidates, key=lambda c: c[2])
#         new_box = box.copy()
#         new_box[face_idx] = val

#         # Ensure it does not squeeze the drone
#         if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
#                 new_box[2] > -drone_radius or new_box[3] < drone_radius):
#             box = new_box
#             moved = True

#     return box, moved

def _push_faces_bonus(box, Qi, Ri, drone_radius, target_rel_x, target_rel_z):
    xMin, xMax, zMin, zMax = box
    moved = False

    LIMIT = 1.0

    for i in range(len(Qi)):
        cx, cz = Qi[i]
        r = Ri[i]
        candidates = []

        # Consider pushing the 4 edges
        new_xMin = cx + r + 1e-4
        if -LIMIT <= new_xMin <= 0:
            candidates.append((0, new_xMin))

        new_xMax = cx - r - 1e-4
        if 0 <= new_xMax <= LIMIT:
            candidates.append((1, new_xMax))

        new_zMin = cz + r + 1e-4
        if -LIMIT <= new_zMin <= 0:
            candidates.append((2, new_zMin))

        new_zMax = cz - r - 1e-4
        if 0 <= new_zMax <= LIMIT:
            candidates.append((3, new_zMax))

        if not candidates:
            continue

        # --- NEW LOGIC: Score = Area + Directional Bonus ---
        best_score = -float('inf')
        best_face_idx = -1
        best_val = 0
        
        # W is the weight of attraction toward the target.
        # If 0.0 -> revert to the original Max algorithm.
        # If too high -> ignore area and create very narrow boxes biased toward the target.
        # 15.0 or 20.0 is usually a good compromise.
        W = 50.0 #15.0 #50

        for face_idx, val in candidates:
            # Create a temporary box to evaluate its area
            test_box = box.copy()
            test_box[face_idx] = val
            
            # 1. Compute the candidate box area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])
            
            # 2. Compute the directional bonus
            # If the target is to the right (positive X), reward test_box[1] (x_max)
            # If the target is to the left (negative X), reward how far test_box[0] (x_min) extends left
            if target_rel_x > 0.1:
                bonus_x = test_box[1]
            elif target_rel_x < -0.1:
                bonus_x = - test_box[0]
            else:
                bonus_x = 0.0
                
            if target_rel_z > 0.1:
                bonus_z = test_box[3]
            elif target_rel_z < -0.1:
                bonus_z = -test_box[2]
            else:
                bonus_z = 0.0

                
            # 3. Total score
            score = area + W * (bonus_x + bonus_z)
            
            if score > best_score:
                best_score = score
                best_face_idx = face_idx
                best_val = val

        # Apply the best choice
        new_box = box.copy()
        new_box[best_face_idx] = best_val

        # Ensure the box does not crush the drone
        if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
                new_box[2] > -drone_radius or new_box[3] < drone_radius):
            box = new_box
            moved = True
        

    return box, moved


def _push_faces_malus(box, Qi, Ri, drone_radius, dx, dz):
    xMin, xMax, zMin, zMax = box
    moved = False
    

    LIMIT = 1.0

    for i in range(len(Qi)):
        cx, cz = Qi[i]
        r = Ri[i]
        candidates = []

        # Consider pushing the 4 edges
        new_xMin = cx + r + 1e-4
        if -LIMIT <= new_xMin <= 0:
            candidates.append((0, new_xMin))

        new_xMax = cx - r - 1e-4
        if 0 <= new_xMax <= LIMIT:
            candidates.append((1, new_xMax))

        new_zMin = cz + r + 1e-4
        if -LIMIT <= new_zMin <= 0:
            candidates.append((2, new_zMin))

        new_zMax = cz - r - 1e-4
        if 0 <= new_zMax <= LIMIT:
            candidates.append((3, new_zMax))

        if not candidates:
            continue

        best_score = -float('inf')
        best_face_idx = -1
        best_val = 0

        # Very large weight to protect the forward face
        WEIGHT = 50.0 

        for face_idx, val in candidates:
            test_box = box.copy()
            test_box[face_idx] = val
            
            # Remaining area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])

            # DIRECTIONAL PENALTY: If moving right (dx > 0) and the candidate cuts the right face (face_idx == 1), apply a strong penalty.
            penalty = 0.0
            if face_idx == 1 and dx > 0.1: penalty = -WEIGHT * (box[1] - val)
            if face_idx == 0 and dx < -0.1: penalty = -WEIGHT * (val - box[0])
            if face_idx == 3 and dz > 0.1: penalty = -WEIGHT * (box[3] - val)
            if face_idx == 2 and dz < -0.1: penalty = -WEIGHT * (val - box[2])

            score = area + penalty

            if score > best_score:
                best_score = score
                best_face_idx = face_idx
                best_val = val

        new_box = box.copy()
        new_box[best_face_idx] = best_val

        if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
                new_box[2] > -drone_radius or new_box[3] < drone_radius):
            box = new_box
            moved = True

    return box, moved

def _push_faces_notblocked(box, Qi, Ri, drone_radius, dx, dz, locked_faces):
    xMin, xMax, zMin, zMax = box
    moved = False

    if locked_faces is None:
        locked_faces = []

    LIMIT = 1.0

    for i in range(len(Qi)):
        cx, cz = Qi[i]
        r = Ri[i]
        candidates = []

        # Consider pushing the 4 edges
        if 0 not in locked_faces:
            new_xMin = cx + r + 1e-4
            if -LIMIT <= new_xMin <= 0:
                candidates.append((0, new_xMin))

        if 1 not in locked_faces:
            new_xMax = cx - r - 1e-4
            if 0 <= new_xMax <= LIMIT:
                candidates.append((1, new_xMax))

        if 2 not in locked_faces:
            new_zMin = cz + r + 1e-4
            if -LIMIT <= new_zMin <= 0:
                candidates.append((2, new_zMin))

        if 3 not in locked_faces:
            new_zMax = cz - r - 1e-4
            if 0 <= new_zMax <= LIMIT:
                candidates.append((3, new_zMax))

        if not candidates:
            continue

        best_score = -float('inf')
        best_face_idx = -1
        best_val = 0

        # Very large weight to protect the forward face
        WEIGHT = 0.0 

        for face_idx, val in candidates:
            test_box = box.copy()
            test_box[face_idx] = val
            
            # Remaining area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])

            # DIRECTIONAL PENALTY: If moving right (dx > 0) and the candidate cuts the right face (face_idx == 1), apply a strong penalty.
            penalty = 0.0
            if face_idx == 1 and dx > 0.1: penalty = -WEIGHT * (box[1] - val)
            if face_idx == 0 and dx < -0.1: penalty = -WEIGHT * (val - box[0])
            if face_idx == 3 and dz > 0.1: penalty = -WEIGHT * (box[3] - val)
            if face_idx == 2 and dz < -0.1: penalty = -WEIGHT * (val - box[2])

            score = area + penalty

            if score > best_score:
                best_score = score
                best_face_idx = face_idx
                best_val = val

        new_box = box.copy()
        new_box[best_face_idx] = best_val

        if not (new_box[0] > -drone_radius or new_box[1] < drone_radius or 
                new_box[2] > -drone_radius or new_box[3] < drone_radius):
            box = new_box
            moved = True

    return box, moved


def _expand_faces(box, Q, R, LIMIT=1.0):
    """
    Algoritmo di espansione avida. Prende un box e ne spinge le facce 
    verso l'esterno finché non sbattono contro un ostacolo o raggiungono LIMIT.
    """
    new_box = box.copy()

    if len(Q) == 0:
        return np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)

    # 1. ESPANDI A DESTRA (+X, indice 1)
    # Cerchiamo ostacoli che sono più a destra della faccia attuale (Q[:, 0] > new_box[1])
    # E che sono allineati verticalmente con il box (sovrapposizione in Z)
    mask_right = (Q[:, 0] > new_box[1]) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3])
    if np.any(mask_right):
        closest_obs_x = np.min(Q[mask_right, 0] - R[mask_right])
        new_box[1] = min(LIMIT, closest_obs_x - 1e-4) # Spinge fino all'ostacolo
    else:
        new_box[1] = LIMIT # Nessun ostacolo, spinge al massimo

    # 2. ESPANDI A SINISTRA (-X, indice 0)
    mask_left = (Q[:, 0] < new_box[0]) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3])
    if np.any(mask_left):
        closest_obs_x = np.max(Q[mask_left, 0] + R[mask_left])
        new_box[0] = max(-LIMIT, closest_obs_x + 1e-4)
    else:
        new_box[0] = -LIMIT

    # 3. ESPANDI IN ALTO (+Z, indice 3)
    mask_top = (Q[:, 1] > new_box[3]) & (Q[:, 0] + R > new_box[0]) & (Q[:, 0] - R < new_box[1])
    if np.any(mask_top):
        closest_obs_z = np.min(Q[mask_top, 1] - R[mask_top])
        new_box[3] = min(LIMIT, closest_obs_z - 1e-4)
    else:
        new_box[3] = LIMIT

    # 4. ESPANDI IN BASSO (-Z, indice 2)
    mask_bottom = (Q[:, 1] < new_box[2]) & (Q[:, 0] + R > new_box[0]) & (Q[:, 0] - R < new_box[1])
    if np.any(mask_bottom):
        closest_obs_z = np.max(Q[mask_bottom, 1] + R[mask_bottom])
        new_box[2] = max(-LIMIT, closest_obs_z + 1e-4)
    else:
        new_box[2] = -LIMIT

    return new_box

def _expand_faces_directional(box, Q, R, dx, dz, LIMIT=1.0):
    """
    Espansione direzionale: spinge verso l'esterno SOLO le facce 
    nella direzione del moto (dx, dz). Le facce opposte restano ferme.
    """
    new_box = box.copy()

    if len(Q) == 0:
        # Nel vuoto, spara al limite solo le facce direzionali
        if dx > 0.05: new_box[1] = LIMIT
        elif dx < -0.05: new_box[0] = -LIMIT
        
        if dz > 0.05: new_box[3] = LIMIT
        elif dz < -0.05: new_box[2] = -LIMIT
        return new_box

    # 1. ESPANDI A DESTRA (+X), solo se andiamo a destra
    if dx > 0.05:
        mask_right = (Q[:, 0] > new_box[1]) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3])
        if np.any(mask_right):
            closest_obs_x = np.min(Q[mask_right, 0] - R[mask_right])
            new_box[1] = min(LIMIT, closest_obs_x - 1e-4)
        else:
            new_box[1] = LIMIT

    # 2. ESPANDI A SINISTRA (-X), solo se andiamo a sinistra
    elif dx < -0.05:
        mask_left = (Q[:, 0] < new_box[0]) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3])
        if np.any(mask_left):
            closest_obs_x = np.max(Q[mask_left, 0] + R[mask_left])
            new_box[0] = max(-LIMIT, closest_obs_x + 1e-4)
        else:
            new_box[0] = -LIMIT

    # 3. ESPANDI IN ALTO (+Z), solo se andiamo in alto
    if dz > 0.05:
        mask_top = (Q[:, 1] > new_box[3]) & (Q[:, 0] + R > new_box[0]) & (Q[:, 0] - R < new_box[1])
        if np.any(mask_top):
            closest_obs_z = np.min(Q[mask_top, 1] - R[mask_top])
            new_box[3] = min(LIMIT, closest_obs_z - 1e-4)
        else:
            new_box[3] = LIMIT

    # 4. ESPANDI IN BASSO (-Z), solo se andiamo in basso
    elif dz < -0.05:
        mask_bottom = (Q[:, 1] < new_box[2]) & (Q[:, 0] + R > new_box[0]) & (Q[:, 0] - R < new_box[1])
        if np.any(mask_bottom):
            closest_obs_z = np.max(Q[mask_bottom, 1] + R[mask_bottom])
            new_box[2] = max(-LIMIT, closest_obs_z + 1e-4)
        else:
            new_box[2] = -LIMIT

    return new_box


# =============================================================================
# 3. TEST AND PLOTTING
# =============================================================================
def run_test_and_plot():
    # Room 3 (Staggered Walls)
    obstacles = [
        [1.0, 2.0, -1.0, 3.0],   # High wall
        [3.0, 4.0, -5.0, -1.5]   # Low wall
    ]
    
    drone_x, drone_z = 4.0, -1.0
    
    # 1. Rays and tangent spheres
    hits, radii = get_lidar_hits_2d(drone_x, drone_z, obstacles, num_rays=360)
    
    # 2. Max algorithm (relative coordinates)
    Q_relative = hits.copy()
    Q_relative[:, 0] -= drone_x
    Q_relative[:, 1] -= drone_z
    
    xMin, xMax, zMin, zMax, status = min_cube_select_base(Q_relative, radii, drone_radius=0.1)
    
    # Absolute coordinates for plotting
    box_abs = [xMin + drone_x, xMax + drone_x, zMin + drone_z, zMax + drone_z]
    
    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Obstacles
    for obs in obstacles:
        ax.add_patch(patches.Rectangle((obs[0], obs[2]), obs[1]-obs[0], obs[3]-obs[2], color='gray', alpha=0.5))
        
    # LiDAR points and circles (matching the sketch)
    if len(hits) > 0:
        ax.scatter(hits[:, 0], hits[:, 1], color='red', s=10, label='Lidar Hits')
        for i in range(len(hits)):
            circle = plt.Circle((hits[i, 0], hits[i, 1]), radii[i], color='red', alpha=0.2)
            ax.add_patch(circle)

    # Asymmetric box (red in the sketch, green here)
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

def run_test_and_plot2():
    # 1. Drone and target position
    drone_x, drone_z = 2.5,  7.0
    target_rel_x, target_rel_z = 10.0, 0.0  # The target is at X=10, Z=5

    # 2. Create an oblique wall (diagonal from X=3,Z=1 to X=7,Z=9)
    num_points = 50
    wall_x = np.linspace(3.0, 7.0, num_points)
    wall_z = np.linspace(1.0, 9.0, num_points)
    hits = np.column_stack((wall_x, wall_z))
    radii = np.full(num_points, 0.05) # Approximate radius of LiDAR points

    # Convert points to relative coordinates for the Max algorithm
    Q_rel = hits.copy()
    Q_rel[:, 0] -= drone_x
    Q_rel[:, 1] -= drone_z

    # 3. Run the Max algorithm
    xMin_r, xMax_r, zMin_r, zMax_r, status = min_cube_select_base(
        Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1
    )

    # Convert the safe box to absolute coordinates
    box_abs = [
        xMin_r + drone_x, xMax_r + drone_x, 
        zMin_r + drone_z, zMax_r + drone_z
    ]

    # 4. Plot for presentation
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw the real oblique wall
    ax.plot(wall_x, wall_z, color='black', linewidth=4, label='Real Oblique Wall')
    ax.scatter(hits[:, 0], hits[:, 1], color='red', s=15, zorder=5, label='LiDAR Points')

    # Draw the obstacle over-approximation (the ghost block seen by the AABB)
    rect_x_min, rect_x_max = np.min(wall_x), np.max(wall_x)
    rect_z_min, rect_z_max = np.min(wall_z), np.max(wall_z)
    ax.add_patch(patches.Rectangle(
        (rect_x_min, rect_z_min), rect_x_max - rect_x_min, rect_z_max - rect_z_min, 
        edgecolor='red', facecolor='red', alpha=0.15, linestyle='--', linewidth=2, 
        label='Perceived obstacle (AABB)'
    ))

    # Draw the Max safe box
    box_w = box_abs[1] - box_abs[0]
    box_h = box_abs[3] - box_abs[2]
    ax.add_patch(patches.Rectangle(
        (box_abs[0], box_abs[2]), box_w, box_h, 
        edgecolor='lime', facecolor='none', linewidth=3, zorder=10,
        label='Safe-Box (Algoritmo Max)'
    ))

    # Drone and target
    ax.scatter(drone_x, drone_z, color='green', s=150, zorder=10, label='Drone')
    ax.scatter(drone_x + target_rel_x, drone_z + target_rel_z, color='orange', marker='X', s=150, zorder=10, label='Target')

    # Appearance
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])
    ax.set_aspect('equal')
    ax.set_title("Fallimento Algoritmo AABB con Ostacoli Obliqui")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    #ax.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("Fallimento_Obliquo.png", dpi=300)
    plt.show()
    
if __name__ == '__main__':
    run_test_and_plot()