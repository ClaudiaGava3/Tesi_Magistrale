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
    filtered_segments = []

    # # filtraggio segmenti
    # for seg in segments:
    #     (x_A, z_A), (x_B, z_B) = seg
        
    #     # Bounding box del segmento vs Bounding box del drone
    #     if (min(x_A, x_B) <= drone_x + max_range and max(x_A, x_B) >= drone_x - max_range and
    #         min(z_A, z_B) <= drone_z + max_range and max(z_A, z_B) >= drone_z - max_range):
            
    #         filtered_segments.append((x_A, z_A, x_B, z_B))
    
    # Compute the angle between rays
    angle_step_rad = (2 * np.pi) / num_rays
    num_half_rays = num_rays // 2  # Solo 180 iterazioni
    angles = np.linspace(0, np.pi, num_half_rays, endpoint=False)

    
    for angle in angles:
        dx, dz = np.cos(angle), np.sin(angle)
        ray_min_dist_fwd = max_range
        ray_min_dist_bwd = max_range
        
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
                if 0<= u <= 1:
                    # 4. Find the nearest wall
                    if t > 0:
                        # Raggio in avanti (Fronte)
                        if t < ray_min_dist_fwd:
                            ray_min_dist_fwd = t
                    elif t < 0:
                        # Raggio all'indietro (Retro): la distanza è positiva (-t)
                        dist_bwd = -t
                        if dist_bwd < ray_min_dist_bwd:
                            ray_min_dist_bwd = dist_bwd

            
        # If the ray hits something within max_range, compute the exact impact point
        if ray_min_dist_fwd < max_range:
            hit_x_fwd = drone_x + ray_min_dist_fwd * dx
            hit_z_fwd = drone_z + ray_min_dist_fwd * dz
            hits.append([hit_x_fwd, hit_z_fwd])
            distances.append(ray_min_dist_fwd)

        if ray_min_dist_bwd < max_range:
            # Per il retro, la direzione spaziale è invertita (-dx, -dz)
            hit_x_bwd = drone_x - ray_min_dist_bwd * dx
            hit_z_bwd = drone_z - ray_min_dist_bwd * dz
            hits.append([hit_x_bwd, hit_z_bwd])
            distances.append(ray_min_dist_bwd)
            
    hits = np.array(hits)
    distances = np.array(distances)
    
    # Radius = distance * tan(angle/2). Add a 5% (1.05) safety margin 
    # to allow the spheres to overlap slightly and avoid numerical "gaps"
    if len(distances) > 0:
        radii = distances * np.tan(angle_step_rad / 2) * 1.05
    else:
        radii = np.array([])
        
    return hits, radii

def get_lidar_hits_2d_qualsiasi2_0(drone_x, drone_z, segments, num_rays=360, max_range=1.5):
    # Compute the angle between rays
    angle_step_rad = (2 * np.pi) / num_rays
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    # 1. Prepare ray matrices (Shape: 360 x 1)
    dx = np.cos(angles).reshape(-1, 1)
    dz = np.sin(angles).reshape(-1, 1)
    
    # 2. Prepare segment matrices (Shape: M muri)
    seg_arr = np.array(segments)
    x_A, z_A = seg_arr[:, 0, 0], seg_arr[:, 0, 1]
    x_B, z_B = seg_arr[:, 1, 0], seg_arr[:, 1, 1]
    
    s_x = x_B - x_A
    s_z = z_B - z_A
    
    # 3. Vectorized Math (Broadcasting 360x1 against 1xM -> Result is 360xM)
    den = dx * s_z - dz * s_x
    
    # Ignora i warning temporanei per divisione per zero (li filtriamo dopo)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_x = x_A - drone_x
        diff_z = z_A - drone_z
        
        t = (diff_x * s_z - diff_z * s_x) / den
        u = (diff_x * dz - diff_z * dx) / den
        
    # 4. Valid impact conditions mask
    valid_hit = (den != 0) & (t > 0) & (u >= 0) & (u <= 1)
    
    # 5. Filter distances: Set invalid hits to infinity
    t_valid = np.where(valid_hit, t, np.inf)
    
    # 6. Find the nearest wall for each ray (Min t across the walls axis)
    ray_min_dist = np.min(t_valid, axis=1)
    
    # 7. Apply max_range filter
    valid_rays_mask = ray_min_dist < max_range
    final_dists = ray_min_dist[valid_rays_mask]
    final_angles = angles[valid_rays_mask]
    
    # 8. Compute hits and radii
    if len(final_dists) > 0:
        hits_x = drone_x + final_dists * np.cos(final_angles)
        hits_z = drone_z + final_dists * np.sin(final_angles)
        hits = np.column_stack((hits_x, hits_z))
        radii = final_dists * np.tan(angle_step_rad / 2) * 1.05
    else:
        hits = np.array([])
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

def min_cube_select_base(Q, R, target_rel_x, target_rel_z, drone_radius=0.1, W=0, rel=0.1):
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
        box, moved = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z, W, rel)

        # push with penalty
        # box, moved = _push_faces_malus(box, Q[intersecting], R
        # [intersecting], drone_radius, target_rel_x, target_rel_z, W)

        if not moved:
            break

    exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], exitflag

def min_cube_select_directional(Q, R, target_rel_x, target_rel_z, drone_radius=0.1, expand_mode='general', W=0, rel=0.1):
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
        box, moved = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z, W, rel)

        # push with malus
        # box, moved = _push_faces_malus(box, Q[intersecting], R
        # [intersecting], drone_radius, target_rel_x, target_rel_z, W)

        # push not blocked faces
        # box, moved = _push_faces_notblocked(box, Q[intersecting], R
        # [intersecting], drone_radius, target_rel_x, target_rel_z, locked_faces, W)

        if not moved:
            break


    # 2 MODES FOR BOX EXPANSION
    if expand_mode == 'general':
        box = _expand_faces(box, Q, R, LIMIT=LIMIT)
    elif expand_mode == 'directional':
        box = _expand_faces_directional(box, Q, R, target_rel_x, target_rel_z, LIMIT=LIMIT)
    elif expand_mode == 'score':
        box = _expand_faces_score(box, Q, R, target_rel_x, target_rel_z, W=W, LIMIT=LIMIT, rel=rel)


    intersecting = _spheres_intersect_box_2d(Q, R, box)
    if np.any(intersecting):
        box, _ = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z, W, rel)
        
    exitflag = 1 if not np.any(_spheres_intersect_box_2d(Q, R, box)) else 0
    return box[0], box[1], box[2], box[3], exitflag



def min_cube_warm_start(Q, R, target_rel_x, target_rel_z, targetx, targetz, drone_radius=0.1, box_prev=None, expand_mode='general', W=0, rel=0.1):
    """
    Initializes from the previous box, checks for obstacles, protects the drone, and expands.
    expand_mode can be: 'general' or 'directional' or 'score'
    """
    LIMIT = 2.12
    
    if box_prev is None:
        box = np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)
    else:
        box = np.array([
            max(-LIMIT, box_prev[0]), 
            min( LIMIT, box_prev[1]), 
            max(-LIMIT, box_prev[2]), 
            min( LIMIT, box_prev[3])
        ], dtype=float)

    box[0] = min(box[0], -drone_radius)
    box[1] = max(box[1],  drone_radius)
    box[2] = min(box[2], -drone_radius)
    box[3] = max(box[3],  drone_radius)

    if len(Q) == 0:
        return -LIMIT, LIMIT, -LIMIT, LIMIT, 1

    for _ in range(100):
        intersecting = _spheres_intersect_box_2d(Q, R, box)
        if not np.any(intersecting):
            break
        box, moved = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z, W)
        if not moved:
            break

    
    if expand_mode == 'general':
        box = _expand_faces(box, Q, R, LIMIT=LIMIT)
    elif expand_mode == 'directional':
        box = _expand_faces_directional(box, Q, R, target_rel_x, target_rel_z, targetx, targetz, LIMIT=LIMIT)
    elif expand_mode == 'score':
        box = _expand_faces_score(box, Q, R, target_rel_x, target_rel_z, W=W, LIMIT=LIMIT, rel=rel)

    intersecting = _spheres_intersect_box_2d(Q, R, box)
    if np.any(intersecting):
        box, _ = _push_faces_bonus(box, Q[intersecting], R[intersecting], drone_radius, target_rel_x, target_rel_z, W)

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

def _push_faces_bonus(box, Qi, Ri, drone_radius, target_rel_x, target_rel_z, W=0, rel=0.1):
    xMin, xMax, zMin, zMax = box
    moved = False

    LIMIT = 2.12

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
        #W = 0.0 #15.0 #50

        for face_idx, val in candidates:
            # Create a temporary box to evaluate its area
            test_box = box.copy()
            test_box[face_idx] = val
            
            # 1. Compute the candidate box area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])
            
            # 2. Compute the directional bonus
            # If the target is to the right (positive X), reward test_box[1] (x_max)
            # If the target is to the left (negative X), reward how far test_box[0] (x_min) extends left

        
            if target_rel_x > rel:
                bonus_x = test_box[1]
            elif target_rel_x < -rel:
                bonus_x = - test_box[0]
            else:
                bonus_x = 0.0
                
            if target_rel_z > rel:
                bonus_z = test_box[3]
            elif target_rel_z < -rel:
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


def _push_faces_malus(box, Qi, Ri, drone_radius, dx, dz, W=0):
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
        #W = 50.0 

        for face_idx, val in candidates:
            test_box = box.copy()
            test_box[face_idx] = val
            
            # Remaining area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])

            # DIRECTIONAL PENALTY: If moving right (dx > 0) and the candidate cuts the right face (face_idx == 1), apply a strong penalty.
            penalty = 0.0
            if face_idx == 1 and dx > 0.1: penalty = -W * (box[1] - val)
            if face_idx == 0 and dx < -0.1: penalty = -W * (val - box[0])
            if face_idx == 3 and dz > 0.1: penalty = -W * (box[3] - val)
            if face_idx == 2 and dz < -0.1: penalty = -W * (val - box[2])

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

def _push_faces_notblocked(box, Qi, Ri, drone_radius, dx, dz, locked_faces, W=0):
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
        W = 0.0 

        for face_idx, val in candidates:
            test_box = box.copy()
            test_box[face_idx] = val
            
            # Remaining area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])

            # DIRECTIONAL PENALTY: If moving right (dx > 0) and the candidate cuts the right face (face_idx == 1), apply a strong penalty.
            penalty = 0.0
            if face_idx == 1 and dx > 0.1: penalty = -W * (box[1] - val)
            if face_idx == 0 and dx < -0.1: penalty = -W * (val - box[0])
            if face_idx == 3 and dz > 0.1: penalty = -W * (box[3] - val)
            if face_idx == 2 and dz < -0.1: penalty = -W * (val - box[2])

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


def _expand_faces_score(box, Q, R, target_rel_x, target_rel_z, W=0, LIMIT=1.0, rel=0.1):
    """
    Espansione unificata: calcola le 4 possibili espansioni (su, giù, dx, sx) 
    fino al primo ostacolo e sceglie quella che massimizza (Area + W * Bonus).
    Ripete finché il box non è incastrato su tutti i lati.
    """
    new_box = box.copy()

    if len(Q) == 0:
        return np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)

    moved = True
    while moved:
        moved = False
        candidates = []

        # --- FASE 1: Trova il limite massimo di espansione per ogni faccia ---
        
        # 0. Faccia Sinistra (-X)
        mask_left = (Q[:, 0] < new_box[0]) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3])
        val_left = np.max(Q[mask_left, 0] + R[mask_left]) + 1e-4 if np.any(mask_left) else -LIMIT
        if val_left < new_box[0] - 1e-3: # Se c'è spazio per espandersi
            candidates.append((0, val_left))

        # 1. Faccia Destra (+X)
        mask_right = (Q[:, 0] > new_box[1]) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3])
        val_right = np.min(Q[mask_right, 0] - R[mask_right]) - 1e-4 if np.any(mask_right) else LIMIT
        if val_right > new_box[1] + 1e-3:
            candidates.append((1, val_right))

        # 2. Faccia Bassa (-Z)
        mask_bottom = (Q[:, 1] < new_box[2]) & (Q[:, 0] + R > new_box[0]) & (Q[:, 0] - R < new_box[1])
        val_bottom = np.max(Q[mask_bottom, 1] + R[mask_bottom]) + 1e-4 if np.any(mask_bottom) else -LIMIT
        if val_bottom < new_box[2] - 1e-3:
            candidates.append((2, val_bottom))

        # 3. Faccia Alta (+Z)
        mask_top = (Q[:, 1] > new_box[3]) & (Q[:, 0] + R > new_box[0]) & (Q[:, 0] - R < new_box[1])
        val_top = np.min(Q[mask_top, 1] - R[mask_top]) - 1e-4 if np.any(mask_top) else LIMIT
        if val_top > new_box[3] + 1e-3:
            candidates.append((3, val_top))

        # Se nessuna faccia può espandersi, fermiamo il ciclo
        if not candidates:
            break 

        # --- FASE 2: Valuta lo Score (Identico al _push_faces_bonus) ---
        best_score = -float('inf')
        best_face_idx = -1
        best_val = 0

        for face_idx, val in candidates:
            test_box = new_box.copy()
            test_box[face_idx] = val
            
            # 1. Calcolo Area
            area = (test_box[1] - test_box[0]) * (test_box[3] - test_box[2])
            
            # 2. Calcolo Bonus Direzionale
            if target_rel_x > rel:
                bonus_x = test_box[1]
            elif target_rel_x < -rel:
                bonus_x = -test_box[0]
            else:
                bonus_x = 0.0
                
            if target_rel_z > rel:
                bonus_z = test_box[3]
            elif target_rel_z < -rel:
                bonus_z = -test_box[2]
            else:
                bonus_z = 0.0
                
            # 3. Score Totale
            score = area + W * (bonus_x + bonus_z)
            
            if score > best_score:
                best_score = score
                best_face_idx = face_idx
                best_val = val

        # --- FASE 3: Applica l'espansione vincente ---
        if best_face_idx != -1:
            new_box[best_face_idx] = best_val
            moved = True # Il box è cambiato, facciamo un altro giro!

    return new_box


def _expand_faces(box, Q, R, LIMIT=1.0):
    """
    Algoritmo di espansione avida. Prende un box e ne spinge le facce 
    verso l'esterno finché non sbattono contro un ostacolo o raggiungono LIMIT.
    """
    new_box = box.copy()


    if len(Q) == 0:
        return np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)


    # INTRODUZIONE DELLA TOLLERANZA (5 cm)
    # Permette al box di scivolare sui muri piatti senza essere strangolato 
    # dalle sfere tangenti espanse dei raggi obliqui.
    TOL = 0.01

    # 1. ESPANDI A DESTRA (+X, indice 1)
    # Cerchiamo ostacoli che sono più a destra della faccia attuale (Q[:, 0] > new_box[1])
    # E che sono allineati verticalmente con il box (sovrapposizione in Z)
    mask_right = (Q[:, 0] > new_box[1] + TOL) & (Q[:, 1] + R > new_box[2]) & (Q[:, 1] - R < new_box[3] - TOL)
    if np.any(mask_right):
        closest_obs_x = np.min(Q[mask_right, 0] - R[mask_right])
        new_box[1] = min(LIMIT, closest_obs_x - 1e-4) # Spinge fino all'ostacolo
    else:
        new_box[1] = LIMIT # Nessun ostacolo, spinge al massimo

    # 2. ESPANDI A SINISTRA (-X, indice 0)
    mask_left = (Q[:, 0] < new_box[0]) & (Q[:, 1] + R > new_box[2] + TOL) & (Q[:, 1] - R < new_box[3] - TOL)
    if np.any(mask_left):
        closest_obs_x = np.max(Q[mask_left, 0] + R[mask_left])
        new_box[0] = max(-LIMIT, closest_obs_x + 1e-4)
    else:
        new_box[0] = -LIMIT

    # 3. ESPANDI IN ALTO (+Z, indice 3)
    mask_top = (Q[:, 1] > new_box[3]) & (Q[:, 0] + R > new_box[0] + TOL) & (Q[:, 0] - R < new_box[1] - TOL)
    if np.any(mask_top):
        closest_obs_z = np.min(Q[mask_top, 1] - R[mask_top])
        new_box[3] = min(LIMIT, closest_obs_z - 1e-4)
    else:
        new_box[3] = LIMIT

    # 4. ESPANDI IN BASSO (-Z, indice 2)
    mask_bottom = (Q[:, 1] < new_box[2]) & (Q[:, 0] + R > new_box[0] + TOL) & (Q[:, 0] - R < new_box[1] - TOL)
    if np.any(mask_bottom):
        closest_obs_z = np.max(Q[mask_bottom, 1] + R[mask_bottom])
        new_box[2] = max(-LIMIT, closest_obs_z + 1e-4)
    else:
        new_box[2] = -LIMIT

    return new_box

def _expand_faces_directional(box, Q, R, dx, dz, targetx, targetz, LIMIT=1.0):
    """
    Espansione direzionale ibrida: espande TUTTE le facce fino al LIMIT, 
    ma processa prima quelle verso cui è diretta la predizione (dx, dz).
    """
    new_box = box.copy()

    # Nel vuoto, espandi tutto al massimo
    if len(Q) == 0:
        return np.array([-LIMIT, LIMIT, -LIMIT, LIMIT], dtype=float)

    # 1. Determina l'ordine in base al vettore direzionale
    # Asse X: 1 = Destra (+X), 0 = Sinistra (-X)
    x_first = 1 if dx >= 0 else 0
    x_last = 0 if dx >= 0 else 1
    
    # Asse Z: 3 = Alto (+Z), 2 = Basso (-Z)
    z_first = 3 if dz >= 0 else 2
    z_last = 2 if dz >= 0 else 3
    
    # # L'array definisce l'ordine: prima le direzionali, poi le opposte
    # ordine_facce = [x_first, z_first, x_last, z_last]
    
    if abs(targetx) >= abs(targetz):
        # Necessità X: garantiamo prima lo spazio orizzontale in avanti
        ordine_facce = [x_first, z_first, x_last, z_last]
    else:
        # Necessità Z: il drone sta salendo/scendendo ripidamente, 
        # garantiamo prima lo spazio verticale
        ordine_facce = [z_first, x_first, z_last, x_last]
    

    # INTRODUZIONE DELLA TOLLERANZA (5 cm)
    # Permette al box di scivolare sui muri piatti senza essere strangolato 
    # dalle sfere tangenti espanse dei raggi obliqui.
    TOL = 0.01

    # 2. Esegui l'espansione seguendo l'ordine calcolato
    for faccia in ordine_facce:
        if faccia == 1: # ESPANDI A DESTRA (+X)
            
            mask = (Q[:, 0] > new_box[1]) & (Q[:, 1] + R > new_box[2]+TOL) & (Q[:, 1] - R < new_box[3]-TOL)
            if np.any(mask):
                new_box[1] = min(LIMIT, np.min(Q[mask, 0] - R[mask]) - 1e-4)
            else:
                new_box[1] = LIMIT
                
        elif faccia == 0: # ESPANDI A SINISTRA (-X)
            mask = (Q[:, 0] < new_box[0]) & (Q[:, 1] + R > new_box[2]+TOL) & (Q[:, 1] - R < new_box[3]-TOL)
            if np.any(mask):
                new_box[0] = max(-LIMIT, np.max(Q[mask, 0] + R[mask]) + 1e-4)
            else:
                new_box[0] = -LIMIT
                
        elif faccia == 3: # ESPANDI IN ALTO (+Z)
            mask = (Q[:, 1] > new_box[3]) & (Q[:, 0] + R > new_box[0]+TOL) & (Q[:, 0] - R < new_box[1]-TOL)
            if np.any(mask):
                new_box[3] = min(LIMIT, np.min(Q[mask, 1] - R[mask]) - 1e-4)
            else:
                new_box[3] = LIMIT
                
        elif faccia == 2: # ESPANDI IN BASSO (-Z)
            mask = (Q[:, 1] < new_box[2]) & (Q[:, 0] + R > new_box[0]+TOL) & (Q[:, 0] - R < new_box[1]-TOL)
            if np.any(mask):
                new_box[2] = max(-LIMIT, np.max(Q[mask, 1] + R[mask]) + 1e-4)
            else:
                new_box[2] = -LIMIT

    return new_box


def force_trajectory_in_box(box_abs, x_sol_prev):
    """
    Controlla se la traiettoria predetta al passo precedente è contenuta nel nuovo box.
    In caso contrario, espande i limiti del box per includerla forzatamente.
    """
    # Se non c'è una traiettoria precedente (es. step 0), restituisci il box intatto
    if x_sol_prev is None or len(x_sol_prev) == 0:
        return box_abs

    # 1. Estrai tutte le coordinate X e Z dalla traiettoria predetta
    traj_x = [p[0] for p in x_sol_prev]
    traj_z = [p[1] for p in x_sol_prev]

    # 2. Trova i limiti geometrici della traiettoria (il suo "Bounding Box")
    traj_xmin = min(traj_x)
    traj_xmax = max(traj_x)
    traj_zmin = min(traj_z)
    traj_zmax = max(traj_z)

    # 3. Creiamo una copia del box per modificarla
    new_box = list(box_abs)

    # 4. Allarghiamo il box se la traiettoria sporge dai bordi
    if traj_xmin < new_box[0]: 
        new_box[0] = traj_xmin  # Allarga a sinistra
        
    if traj_xmax > new_box[1]: 
        new_box[1] = traj_xmax  # Allarga a destra
        
    if traj_zmin < new_box[2]: 
        new_box[2] = traj_zmin  # Allarga in basso
        
    if traj_zmax > new_box[3]: 
        new_box[3] = traj_zmax  # Allarga in alto

    return new_box


def test_confronto_metodi_obliqui():
    # # 1. Setup Ambiente: Un imbuto obliquo (V-shape) definito a segmenti
    # segments = [
    #     [(-2.5, 0.0), (-0.5, 2.5)],  # Muro obliquo sinistro
    #     [( 2.5, 0.0), ( 0.5, 2.5)],  # Muro obliquo destro
    #     [(-0.5, 2.5), (-0.5, 4.0)],  # Muro dritto sinistro (uscita imbuto)
    #     [( 0.5, 2.5), ( 0.5, 4.0)]   # Muro dritto destro (uscita imbuto)
    # ]
    
    # traiettoria = [
    #     (0.0,  0.00),(0.0,  0.10),(0.0,  0.20),(0.0,  0.30),(0.0,  0.40),
    #     (0.0,  0.50),(0.0,  0.60),(0.0,  0.70),(0.0,  0.80),(0.0,  0.90),
    #     (0.0,  1.00),(0.0,  1.10),(0.0,  1.20),(0.0,  1.30),(0.0,  1.40),
    #     (0.0,  1.50),(0.0,  1.60),(0.0,  1.70),(0.0,  1.80),(0.0,  1.90),
    #     (0.0,  2.00),(0.0,  2.10),(0.0,  2.20),(0.0,  2.30)
    # ]
    # target_abs_x, target_abs_z = 0.0, 6.0 # Target molto oltre l'imbuto
    
    # # Inizializzazione Box per WARM START (assoluto) 
    # boxW_abs = [-1.0, 1.0, -1.0, 1.0] 

    # # 2. Setup Ambiente: Curva a L verso destra
    # segments = [
    #     [(-1.0, -1.0), (-1.0, 2.0)], # Muro esterno sinistro (verticale)
    #     [(-1.0, 2.0), (3.0, 2.0)],   # Muro esterno alto (orizzontale) -> spigolo concavo in (-1, 2)
    #     [(1.0, -1.0), (1.0, 0.0)],   # Muro interno destro (verticale)
    #     [(1.0, 0.0), (3.0, 0.0)]     # Muro interno basso (orizzontale) -> spigolo convesso in (1, 0)
    # ]
    
    # traiettoria = [
    #     (0.5, -0.80), (0.5, -0.65), (0.5, -0.50), (0.5, -0.35), 
    #     (0.5, -0.20), (0.5, -0.05), (0.5,  0.10), (0.5,  0.25), 
    #     (0.5,  0.40), (0.5,  0.55), (0.5,  0.70), (0.5,  0.85), 
    #     (0.5,  1.00), (0.65, 1.00), (0.80, 1.00), (0.95, 1.00), 
    #     (1.10, 1.00), (1.25, 1.00), (1.40, 1.00), (1.55, 1.00), 
    #     (1.70, 1.00), (1.85, 1.00), (2.00, 1.00), (2.15, 1.00)
    # ]
    # target_abs_x, target_abs_z = 4.0, 1.0
    
    # # Box di partenza centrato sul primo step
    # boxW_abs = [-0.5, 1.5, -1.8, 0.2]

    # # 3. Setup Ambiente: Chicane Obliqua a Zig-Zag
    # segments = [
    #     [(-2.0, 0.0), (0.5, 1.5)],   # Sale obliquo verso destra
    #     [(0.5, 1.5), (-1.5, 3.0)],   # Torna indietro (Spigolo Convesso in 0.5, 1.5)
    #     [(1.0, -0.5), (3.0, 1.0)],   # Sale obliquo verso destra
    #     [(3.0, 1.0), (1.0, 2.5)]     # Torna indietro (Spigolo Concavo in 3.0, 1.0)
    # ]
    
    # traiettoria = [
    #     (0.0, 0.0), (0.1, 0.12), (0.199, 0.24), (0.295, 0.36), 
    #     (0.386, 0.48), (0.47, 0.6), (0.546, 0.72), (0.613, 0.84), 
    #     (0.669, 0.96), (0.714, 1.08), (0.748, 1.2), (0.771, 1.32), 
    #     (0.783, 1.44), (0.783, 1.56), (0.771, 1.68), (0.748, 1.8), 
    #     (0.714, 1.92), (0.669, 2.04), (0.613, 2.16), (0.546, 2.28), 
    #     (0.47, 2.4), (0.386, 2.52), (0.295, 2.64), (0.199, 2.76)
    # ]
    # target_abs_x, target_abs_z = -1.0, 4.0
    
    # # Box di partenza
    # boxW_abs = [-1.0, 1.0, -1.0, 1.0]


    # 4. Setup Ambiente: Il Diamante
    segments = [
        [(-2.0, -1.0), (-2.0, 4.0)], # Muro di contenimento sinistro
        [(2.0, -1.0), (2.0, 4.0)],   # Muro di contenimento destro
        [(0.0, 1.0), (1.0, 2.0)],    # Lato in basso a destra
        [(1.0, 2.0), (0.0, 3.0)],    # Lato in alto a destra
        [(0.0, 3.0), (-1.0, 2.0)],   # Lato in alto a sinistra
        [(-1.0, 2.0), (0.0, 1.0)]    # Lato in basso a sinistra (Il drone sfiora questo)
    ]
    
    traiettoria = [
        (0.0, 0.0), (-0.188, 0.15), (-0.372, 0.3), (-0.547, 0.45), 
        (-0.713, 0.6), (-0.865, 0.75), (-1.002, 0.9), (-1.122, 1.05), 
        (-1.223, 1.2), (-1.303, 1.35), (-1.361, 1.5), (-1.396, 1.65), 
        (-1.4, 1.8), (-1.382, 1.95), (-1.341, 2.1), (-1.278, 2.25), 
        (-1.194, 2.4), (-1.09, 2.55), (-0.969, 2.7), (-0.832, 2.85), 
        (-0.68, 3.0), (-0.517, 3.15), (-0.344, 3.3), (-0.165, 3.45)
    ]
    target_abs_x, target_abs_z = 0.0, 5.0
    
    # Box di partenza
    boxW_abs = [-1.0, 1.0, -1.0, 1.0]

    
    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    fig.suptitle("Comparative test of expansion methods (score)", fontsize=20)

    
    for step, (drone_x, drone_z) in enumerate(traiettoria):
        ax = axes.flatten()[step]
        
        # --- LIDAR QUALSIASI ---
        hits, radii = get_lidar_hits_2d_qualsiasi(drone_x, drone_z, segments, num_rays=360, max_range=1.5)
        
        # Coordinate relative
        Q_rel = hits.copy()
        if len(Q_rel) > 0:
            Q_rel[:, 0] -= drone_x
            Q_rel[:, 1] -= drone_z
        target_rel_x = target_abs_x - drone_x
        target_rel_z = target_abs_z - drone_z
        
        # ==========================================
        # BINARIO 1: CASO 6 (Indipendente)
        # ==========================================
        xMin6, xMax6, zMin6, zMax6, _ = min_cube_select_base(
            Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1, W=50.0
        )
        # Salvato solo per il plot
        box6_abs = [xMin6 + drone_x, xMax6 + drone_x, zMin6 + drone_z, zMax6 + drone_z]
        
        # ==========================================
        # BINARIO 2: WARM START (Auto-alimentato)
        # ==========================================
        # Il box prev è il boxW_abs dello step precedente
        box_warm_prev_rel = [
            boxW_abs[0] - drone_x, boxW_abs[1] - drone_x, 
            boxW_abs[2] - drone_z, boxW_abs[3] - drone_z
        ]
        
        xMinW, xMaxW, zMinW, zMaxW, _ = min_cube_warm_start(
            Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1, 
            box_prev=box_warm_prev_rel, expand_mode='score', W=50.0
        )
        # Sovrascrivo la variabile per usarla al prossimo loop
        boxW_abs = [xMinW + drone_x, xMaxW + drone_x, zMinW + drone_z, zMaxW + drone_z]

        xMinD, xMaxD, zMinD, zMaxD, _ = min_cube_select_directional(
            Q_rel, radii, target_rel_x, target_rel_z, drone_radius=0.1, 
            expand_mode='score', W=50.0
        )
        boxD_abs = [xMinD + drone_x, xMaxD + drone_x, zMinD + drone_z, zMaxD + drone_z]
        
        # --- PLOTTING ---
        # Disegna Segmenti
        for seg in segments:
            (xA, zA), (xB, zB) = seg
            ax.plot([xA, xB], [zA, zB], color='black', linewidth=4, zorder=2)
        
        # Disegna punti LiDAR
        if len(hits) > 0:
            ax.scatter(hits[:, 0], hits[:, 1], color='gray', s=10, zorder=3)
            
        # Disegna i Box
        ax.add_patch(patches.Rectangle((box6_abs[0], box6_abs[2]), box6_abs[1]-box6_abs[0], box6_abs[3]-box6_abs[2], 
                                       linewidth=3, edgecolor='green', facecolor='none', zorder=5, label='Caso 6 (Taglio)'))
        ax.add_patch(patches.Rectangle((boxW_abs[0], boxW_abs[2]), boxW_abs[1]-boxW_abs[0], boxW_abs[3]-boxW_abs[2], 
                                       linewidth=3, edgecolor='red', linestyle='--', facecolor='none', zorder=6, label='Warm Start (Espansione)'))
        ax.add_patch(patches.Rectangle((boxD_abs[0], boxD_abs[2]), boxD_abs[1]-boxD_abs[0], boxD_abs[3]-boxD_abs[2], 
                                       linewidth=3, edgecolor='blue', linestyle=':', facecolor='none', zorder=6, label='Directional (Espansione)'))
        
        
        # Disegna Drone
        ax.scatter(drone_x, drone_z, color='blue', s=150, zorder=7, label='Drone')
        
        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([-2.0, 4.5])
        #ax.set_title(f"Step {step+1}: Z={drone_z}")
        #if step == 0:
        #    ax.legend(loc='lower left', fontsize=2)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.show()


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
    # run_test_and_plot()
    test_confronto_metodi_obliqui()
