from __future__ import annotations
# Standard library
import os
import random
import shutil
import time
import warnings
from multiprocessing import Pool, Value

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import torch
import adam
from adam.numpy import KinDynComputations
from mpl_toolkits.mplot3d import Axes3D
from rich.traceback import install
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
from urdf_parser_py.urdf import URDF

# Local
from src.VBOC.abstract_copy import Model
from src.VBOC.controller_copy import ViabilityController
from src.VBOC.learning import NeuralNetwork, NovelNeuralNetwork, RegressionNN, plot_brs
from src.VBOC.parser import Parameters, parse_args

# Impostazioni globali per le dimensioni del font
plt.rcParams.update({
    'axes.titlesize': 28,     # Dimensione titolo
    'axes.labelsize': 24,     # Dimensione etichette assi (X e Y)
    'xtick.labelsize': 12,    # Dimensione numeri asse X
    'ytick.labelsize': 12,    # Dimensione numeri asse Y
    'legend.fontsize': 22,    # Dimensione legenda
    'font.size': 22           # Dimensione base per tutto il resto
})

install()

progress_var = Value('i', 0)
np.set_printoptions(linewidth=np.inf)


def plot_histogram(
    data: np.ndarray,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
    saving_dir: str = "plots/histograms_randB/",
    xticks: list = None,
    subplot_titles: list = None
) -> None:
    """
    Plot a grid of histograms (up to 6) for each dimension of the input data.

    The figure is saved as a PNG file in the specified directory and
    automatically closed after saving.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n,) or (n, d), where d is the number of
        dimensions to plot (max 6).
    title : str
        Title of the figure and name of the output PNG file.
    xlabel : str
        Label for the x-axis of each subplot.
    ylabel : str
        Label for the y-axis of each subplot.
    bins : int
        Number of bins for each histogram.
    saving_dir : str
        Directory where the PNG file will be saved.
    """
    #in 3D
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))

    fig.suptitle(title)
    axes = axes.flatten()

    # Hide all subplots by default, show only those needed
    for ax in axes:
        ax.set_visible(False)

    # Ensure data is 2D (n, d) even if 1D input is provided
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    for i in range(data.shape[1]):
        axes[i].set_visible(True)
        axes[i].hist(data[:, i], bins=bins, edgecolor='black', alpha=0.7)

        # Gestione Titoli personalizzati
        if subplot_titles and i < len(subplot_titles):
            axes[i].set_title(subplot_titles[i], fontsize=20)
        else:
            axes[i].set_title(f"Dimension {i+1}", fontsize=20)
            
        # Gestione Etichette asse X personalizzate
        if isinstance(xlabel, list) and i < len(xlabel):
            axes[i].set_xlabel(xlabel[i], fontsize=11)
        else:
            axes[i].set_xlabel(xlabel, fontsize=11)
        
        axes[i].set_ylabel(ylabel, fontsize=18)
        axes[i].grid(True, which='both', alpha=0.75)

    if xticks is not None:
            axes[i].set_xticks(xticks)
        
        
    plt.savefig( os.path.join(saving_dir, title + ".png"))
    plt.close(fig)

def ensure_clean_dir(path: str) -> None:
    """
    Ensure that a directory exists and is empty.

    If the directory exists, all files inside are deleted.
    If it does not exist, it is created (including any missing parent 
    directories).

    Parameters
    ----------
    path : str
        Path to the directory to clean or create.
    """
    if os.path.exists(path):
        # Remove all files inside the directory
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(path)

def compute_data_on_border(
    q_init: np.ndarray, # DEVE ESSERE COMPOSTO DA POS E VEL
    ref_box: np.ndarray,
    box_guess: float,
    N_guess: int,
    N_increment: int,
    vboc_repeat: int,
    #box_min_values: np.ndarray,
    #box_max_values: np.ndarray,
    #random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute a single data point on the border of the viability kernel.

    Solves a VBOC problem from a given configuration along a random (or fixed)
    velocity direction, extending the horizon iteratively if needed.

    Parameters
    ----------
    q_init : np.ndarray
        Initial joint configuration of shape (nq,).
    N_guess : int
        Initial prediction horizon length.
    N_increment : int
        Number of time steps added per VBOC iteration.
    vboc_repeat : int
        Maximum number of VBOC solve repetitions before declaring failure.
    box_min_values : np.ndarray
        Lower bounds of the obstacle bounding box.
    box_max_values : np.ndarray
        Upper bounds of the obstacle bounding box.
    random_seed : int
        Seed for NumPy's RNG, used to sample the velocity direction.

    Returns
    -------
    x0 : np.ndarray or None
        First state of the optimal trajectory (the border point); None if 
        infeasible.
    x_star : np.ndarray or None
        Full optimal state trajectory of shape (N, nx); None if infeasible.
    u_star : np.ndarray or None
        Optimal control sequence of shape (N, nu); None if infeasible.
    box_min_values : np.ndarray
        Unchanged lower obstacle bounds.
    box_max_values : np.ndarray
        Unchanged upper obstacle bounds.
    status : int
        Solver exit status (0 = success).
    d : np.ndarray
        Unit velocity direction used in the OCP.
    """
    global progress_var
    
    controller.resetHorizon(N_guess)

    # --- Velocity direction ---
    # if params.check:
    #     # Fixed direction for debug/check mode
    #     d = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    # else:
    #     # Normal distribution ensures uniform sampling on the unit sphere
    #     np.random.seed(random_seed)
    #     d = np.array([np.random.normal() for _ in range(model.nv)])
    # d /= np.linalg.norm(d)

    # --- Initial guess: stationary at q_init with gravity compensation ---
    #in 3D
    x_guess = np.zeros((N_guess, model.nx))
    # Manteniamo la posizione piatta e lo scaling costante per tutto il guess
    x_guess[:, :3] = q_init[:3]  
    # Guess iniziale per i 6 lati (immaginiamo che partano cubi a proporzione 1.0)
    x_guess[:, 12:18] = 1.0
    x_guess[:, 18] = box_guess 

    # MA al primissimo istante di tempo (nodo 0), diciamo ad Acados esattamente
    # come parte il drone nella realtà (incluse le velocità e gli angoli veri)
    x_guess[0, :12] = q_init

    # x_static = np.hstack((q_init, np.zeros(model.nx - model.nq)))
    # gravity_wrench = np.array([0, 0, model.mass * model.g])
    # allocation_matrix = model.R(x_static).full() @ model.F
    # u_hover = np.linalg.pinv(allocation_matrix) @ gravity_wrench

    # x_static = np.hstack((q_init, np.full(4, box_guess)))
    # allocation_matrix = model.R(x_static).full() @ model.F

    # 1. Definiamo uno stato orizzontale per il guess iniziale
    x_flat = np.zeros(19)
    x_flat[:3] = q_init[:3]  # Copia solo le posizioni (X, Y, Z)
    x_flat[18] = box_guess    # Copia lo scaling (Indice 6 in 2D)

    # 2. Costruiamo la matrice di allocazione per il 2D: Matrice 6x2.
    allocation_matrix = np.vstack((model.F, model.M))

    # 3. Obiettivo fisico: [Fx, Fy, Fz, Mx, My, Mz]
    # Bilanciamo la gravità su Z e annulliamo le coppie
    wrench_hover = np.array([0.0, 0.0, model.mass * model.g, 0.0, 0.0, 0.0])
    
    # 4. Spinta bilanciata per i 4 motori
    u_hover = np.linalg.pinv(allocation_matrix) @ wrench_hover
    u_guess = np.tile(u_hover, (N_guess, 1))
    #u_guess = np.full((N_guess, model.nu), u_hover)

    controller.setGuess(x_guess, u_guess)

    # --- Solve the OCP ---
    x_star, u_star, N_final, status = controller.solve_vboc(
        q_init, ref_box, N_guess, n=N_increment,
        repeat=vboc_repeat
    )

    # --- Update progress ---
    with progress_var.get_lock():   
        progress_var.value += 1     
        if progress_var.value % 100 == 0: 
            print(
                f" Progress: {progress_var.value} / \
                {controller.model.params.prob_num}"
            )

    # --- Return results ---
    if x_star is None:
        return None, None, None, None, status
    else:
        return (
            x_star[0], x_star, u_star, N_final, status
        )
    
# def fixed_velocity_dir(
#     N_guess: int,
#     N_increment: int,
#     vboc_repeat: int,
#     n_pts: int = 50
# ) -> tuple[list, list]:
#     """
#     Compute data on a section of the viability kernel.

#     For each position DOF, solves the VBOC problem along a grid of points
#     in both the positive and negative velocity directions.

#     Parameters
#     ----------
#     N_guess : int
#         Initial prediction horizon length.
#     N_increment : int
#         Number of time steps added per VBOC iteration.
#     vboc_repeat : int
#         Maximum number of VBOC solve repetitions before declaring failure.
#     n_pts : int, optional
#         Number of grid points per DOF. Default is 50.

#     Returns
#     -------
#     sec_pts : list of np.ndarray
#         Section points for each position DOF.
#     status_list : list of np.ndarray
#         Solver status vector for each position DOF.
#     """
#     sec_pts = []
#     status_list = []
#     controller.resetHorizon(N_guess)

#     # Gravity-compensating hover thrust at the origin 
#     # (shared across all iterations)
#     u_hover = (
#         np.linalg.pinv(model.R(np.zeros(model.nx)).full() @ model.F) @ 
#         np.array([0, 0, model.mass * model.g])
#     )

#     for i in range(model.npos):
#         # --- Build position grid for DOF i, mapped to box dimensions ---

#         q_lo = model.env_dimensions[i] - model.drone_occupancy[i]
#         q_hi = (
#             model.env_dimensions[i+model.npos] 
#             - model.drone_occupancy[i+model.npos]
#         )
#         q_grid = np.linspace(q_lo, q_hi, n_pts)

#         box_max_grid = np.empty(n_pts) * np.nan
#         box_min_grid = np.empty(n_pts) * np.nan
#         for k in range(n_pts):
#             box_max_grid[k] = min(
#                 model.env_dimensions[i+3], 
#                 model.env_dimensions[i+3] - q_grid[k]
#             )
#             box_min_grid[k] = -max(
#                 model.env_dimensions[i], 
#                 model.env_dimensions[i] - q_grid[k]
#             )

#         # Duplicate grid for positive (j < n_pts) and negative 
#         # (j >= n_pts) directions
#         q_grid = np.tile(q_grid, 2)
#         box_max_grid = np.tile(box_max_grid, 2) 
#         box_min_grid = np.tile(box_min_grid, 2)

#         # --- Storage for this DOF ---
#         x_sec = np.empty((0, model.nx)) * np.nan 
#         status_vec = np.empty(n_pts * 2) * np.nan
        
#         for j in tqdm(range(n_pts * 2), desc=f"DOF {i+1}/{model.npos}"):
            
#             # Box bounds: start from environment limits, then override DOF i
#             box_max_values = model.env_dimensions[3:].copy()
#             box_min_values = -model.env_dimensions[:3].copy()
#             box_max_values[i] = box_max_grid[j]
#             box_min_values[i] = box_min_grid[j]

#             # Unit velocity direction: +1 for first half, -1 for second half
#             d = np.zeros(model.nv)
#             d[i] = 1 if j < n_pts else -1

#             # Warm-start guess: stationary at the origin
#             q_init = np.zeros(model.nq)             
#             x_guess = np.zeros((N_guess, model.nx))
#             u_guess = np.full((N_guess, model.nu), u_hover)
#             controller.setGuess(x_guess, u_guess)

#             # --- Solve VBOC ---
#             x_star, _, _, status = controller.solve_vboc(
#                 q_init, d, box_min_values, box_max_values, N_guess,
#                 n=N_increment, repeat=vboc_repeat
#             )
            
#             if status == 0:
#                 # Replace the optimised position with the grid value,
#                 # since the OCP fixes velocity direction, not position.
#                 x_star[0, i] = q_grid[j]
#                 x_sec = np.vstack([x_sec, x_star[0]])

#             status_vec[j] = status

#         sec_pts.append(x_sec)
#         status_list.append(status_vec)

#     return sec_pts, status_list

def generate_constrained_rpy(
    min_inclination: float,
    max_inclination: float,
    n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate uniformly distributed orientations within an inclination range.

    Uses rejection sampling on random quaternions to produce ``n_samples``
    orientations whose Z-axis tilt angle lies in
    ``[min_inclination, max_inclination]``.

    Parameters
    ----------
    min_inclination : float
        Minimum angle between the world Z-axis and the rotated Z-axis (rad).
    max_inclination : float
        Maximum angle between the world Z-axis and the rotated Z-axis (rad).
    n_samples : int
        Number of valid orientations to generate.

    Returns
    -------
    roll : np.ndarray, shape (n_samples,)
        Roll angles in radians.
    pitch : np.ndarray, shape (n_samples,)
        Pitch angles in radians.
    yaw : np.ndarray, shape (n_samples,)
        Yaw angles in radians.

    Raises
    ------
    ValueError
        If arguments are out of range or of the wrong type.
    RuntimeError
        If the sampler exhausts ``max_tries`` before collecting enough samples.
    """
    # --- Input validation ---
    if not (
        isinstance(min_inclination, (int, float)) and
        isinstance(max_inclination, (int, float)) and
        isinstance(n_samples, int) and
        0 <= min_inclination <= max_inclination <= 180 and
        n_samples >= 0
    ):
        raise ValueError("Invalid input arguments.  \
            Check ranges (0<=a<=b<=180) and types."
        )

    if n_samples == 0:
        return np.array([]), np.array([]), np.array([])
    
    roll_list, pitch_list, yaw_list = [], [], []
    count = 0
    max_tries = max(n_samples * 1000, 10000)

    # --- Rejection sampling ---
    for _ in range(max_tries):
        if count == n_samples:
            break

        # rot[2, 2] is the cosine of the tilt angle between Z-axes
        rot = Rot.random().as_matrix()
        theta = np.arccos(np.clip(rot[2, 2], -1.0, 1.0))

        if min_inclination <= theta <= max_inclination:
            # ZYX convention returns [yaw, pitch, roll]
            yaw, pitch, roll = Rot.from_matrix(rot).as_euler('ZYX')
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)
            count += 1

    if count < n_samples:
        raise RuntimeError(
            f"Max tries ({max_tries}) exceeded: "
            f"found {count}/{n_samples} valid samples."
        )

    return np.array(roll_list), np.array(pitch_list), np.array(yaw_list)

def set_axes_equal(ax: Axes3D) -> None:
    """
    Set equal aspect ratio for a 3D Matplotlib axis.

    Rescales all three axes to share the same range, centred on the
    midpoint of each axis's current limits.

    Parameters
    ----------
    ax : Axes3D
        A Matplotlib 3D axis object to rescale.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    midpoints = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]).max() / 2

    ax.set_xlim3d([midpoints[0] - half_range, midpoints[0] + half_range])
    ax.set_ylim3d([midpoints[1] - half_range, midpoints[1] + half_range])
    ax.set_zlim3d([midpoints[2] - half_range, midpoints[2] + half_range])

def normalize_data(data: np.ndarray, indexes: list[int]) -> np.ndarray:
    """
    Normalize specific columns of an array to the [0, 1] range.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n, d), modified in-place.
    indexes : list of int
        Column indices to normalize.

    Returns
    -------
    np.ndarray
        The array with the specified columns normalized.
    """
    for idx in indexes:
        col = data[:, idx]
        col_min, col_max = col.min(), col.max()
        data[:, idx] = (col - col_min) / (col_max - col_min)

    return data

class Sine(torch.nn.Module):  
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sin(self.alpha * x)

class OverMSELoss(torch.nn.Module): 
    """ Custom MSE loss that penalizes more overestimates """
    def __init__(self, alpha=1., beta=0.6):
        super(OverMSELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        l2 = torch.mean((y_pred - y_true) ** 2)
        l2_over = torch.mean(torch.relu(y_pred - y_true) ** 2) 
        return self.alpha * l2 + self.beta * l2_over
    
class RAELoss(torch.nn.Module): 
    """ Relative Absolute Error loss """
    def __init__(self):
        super(RAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        num = torch.sum(torch.abs(y_true - y_pred))
        den = torch.sum(torch.abs(y_true - torch.mean(y_true)))
        return num / den
    
class CustomLoss(torch.nn.Module):  
    """ Custom loss function (MSE + RE on overestimates) """
    def __init__(self, alpha=1., beta=0.6):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        l2 = torch.mean((y_pred - y_true) ** 2)
        l1_over = torch.mean(torch.relu(y_pred - y_true))
        return self.alpha * l2 + self.beta * l1_over 
    
if __name__ == '__main__':
    start_time = time.time()

    # --- Parse command-line arguments ---
    global args, params
    args = parse_args()
    robotic_system = args['system']
    available_systems = ['sth']
    try:
        if robotic_system not in available_systems:
            raise NameError
    except NameError:
        print('\nSystem not available! Available: ', available_systems, '\n')
        exit()
    params = Parameters(robotic_system) 
    params.generation = args['generation']
    params.check = args['check']
    params.build = args['build']
    params.plot = args['plot']
    params.training = args['training']
    params.act = args['activation']
    params.weight_decay = args['weightDecay']

    # --- Initialize model and controller ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global model, controller
    model = Model(params)
    controller = ViabilityController(model)
    nq = model.nq
    nu = model.nu
    if not os.path.exists(params.DATA_DIR):
        os.makedirs(params.DATA_DIR)
    if not os.path.exists(params.NN_DIR):
        os.makedirs(params.NN_DIR)
    N = params.N
    N_increment = params.N_increment
    vboc_repeat = params.vboc_repeat
    horizon = args['horizon']
    plots_dir = params.PLOTS_DIR
    if horizon is not None:
        try:
            if horizon < 1:
                raise ValueError
        except ValueError:
            print('\nThe horizon must be greater than 0!\n')
            exit()
        if horizon < N:
            N = horizon
    nls = {
        'relu': torch.nn.ReLU(),
        'elu': torch.nn.ELU(),
        'tanh': torch.nn.Tanh(),
        'sine': Sine(),
        'gelu': torch.nn.GELU(approximate='tanh'),
        'silu': torch.nn.SiLU(),
        'sigm': torch.nn.Sigmoid()
    }
    act_fun = nls[params.act]
    nn_filename = f'{params.NN_DIR}{robotic_system}_{params.act}_randB_2mln.pt'
    ub = 1

    # =========================================================================
    # DATA GENERATION
    # =========================================================================
    if params.generation:

        # In check mode, solve a single problem with a fixed configuration
        if params.check:
            params.prob_num = 1
        
        # --- Initial position: origin for all problems ---
        pos_init = np.zeros((params.prob_num, model.npos))

        # --- Initial orientation: sampled within the allowed inclination  
        # range ---
        if(params.orient_g_rej):
            #min_phi = 0.0
            max_phi = np.pi/2
        else:
            #min_phi = model.phi_max
            max_phi = np.pi/2
        #roll, pitch, yaw = generate_constrained_rpy(
        #    min_phi, max_phi, params.prob_num
        #)
        if params.check:
            orient_init = np.zeros((params.prob_num, model.nori))
            vel_init = np.zeros((params.prob_num, model.nv))
        else:
            #orient_init = np.column_stack([roll, pitch, yaw])
            # Generiamo pitch casuale
            #in 3D
            orient_init = np.random.uniform(-max_phi, max_phi, (params.prob_num, model.nori))

            # Generiamo VELOCITÀ casuali (vx, vy, vz, p, q, r)
            # in 3D: model.nv = 6
            vel_init = np.random.uniform(-1.0, 1.0, (params.prob_num, model.nv))

        # Creiamo il vettore di stato iniziale di 12 elementi (3 pos + 3 ori + 6 vel)
        q_init = np.hstack([pos_init, orient_init, vel_init])
        b_init_raw = np.random.uniform(0.1, 1.0, (params.prob_num, model.nbox))
        b_init = b_init_raw / np.linalg.norm(b_init_raw, axis=1, keepdims=True)

        # --- Obstacle box bounds --- 
        box_guess=1e1

        # --- Random seeds, one per problem ---
        randomSeeds = [random.randint(0, params.prob_num) 
                       for _ in range(params.prob_num)
        ]

        # --- Accumulators for results across all batches ---
        #all_x_0, all_x_t, all_u_t, all_b_m, all_b_M, all_status, all_d_list = \
        #[],[],[],[],[],[],[]
        all_x_0, all_x_t, all_u_t, all_n_final, all_status = [], [], [], [], []
        all_failed_q_init = [] # <--- NUOVA LISTA PER I FALLIMENTI
        all_test_dataset = []

        # Split the problems into sub-batches to allow intermediate saves
        if params.check:
            sub_batch = 1
        else:
            sub_batch = 100
        n_batch = int(params.prob_num/sub_batch)

        
        #in 3D
        ref_box = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # Box di riferimento 1:1

        t_start_step = time.perf_counter()

        print('Start data generation')
        for nb in range(n_batch):  
            with Pool(params.cpu_num) as p:
                res = p.starmap(
                    compute_data_on_border, 
                    [(q0, b0, box_guess, N, N_increment, vboc_repeat) 
                     for q0, b0 in zip(q_init[(nb*sub_batch):((nb+1)*sub_batch)], b_init[(nb*sub_batch):((nb+1)*sub_batch)])]
                )

            # --- Unpack parallel results (Ora sono 5, non più 7!) ---
            x_0, x_t, u_t, n_final_list, status = zip(*res)
            all_x_0.extend(x_0)
            all_x_t.extend(x_t)
            all_u_t.extend(u_t)
            all_n_final.extend(n_final_list)
            all_status.extend(status)

            # --- NUOVO BLOCCO PER IL SET DI TEST COMPLETO ---
            q0_batch = q_init[(nb*sub_batch):((nb+1)*sub_batch)]
            b0_batch = b_init[(nb*sub_batch):((nb+1)*sub_batch)]
            
            for i in range(len(x_0)):
                # Se ha successo prendiamo lo scale (indice 10), se fallisce mettiamo -1.0
                scale_val = x_0[i][10] if x_0[i] is not None else -1000.0
                
                # q0_batch[i][2:6] salta x e z, prendendo solo theta, vx, vz, wy
                row = np.hstack([q0_batch[i][3:12], b0_batch[i], status[i], scale_val])
                all_test_dataset.append(row)
            # ------------------------------------------------

            # === NUOVO CODICE: CATTURA I CASI FALLITI ===
            q0_batch = q_init[(nb*sub_batch):((nb+1)*sub_batch)]
            for i in range(len(x_0)):
                if x_0[i] is None:
                    # Se ha fallito, salviamo la sua condizione iniziale
                    all_failed_q_init.append(q0_batch[i])
            # ============================================

            if all(item is None for item in x_0):
                warnings.warn(f'No solution found for any problem in batch {nb}.', RuntimeWarning)
                continue
            
            if all(item is None for item in all_x_0):
                warnings.warn('No solution found for any problem. Exiting.', RuntimeWarning)
                exit()


            x_data = np.vstack([i for i in all_x_0 if i is not None])
            x_traj = [i for i in all_x_t if i is not None]
            u_traj = [i for i in all_u_t if i is not None]
            n_data = np.array([all_n_final[i] for i in range(len(all_n_final)) if all_x_0[i] is not None])  # Filtriamo N_final usando la stessa esatta logica di x_data per mantenere l'allineamento
            status_list = list(all_status)
            
            # Il "box" ottimizzato è solo un fattore di scala!
            b_optimized = x_data[:, 18].reshape(-1, 1)

            actual_boxes = x_data[:, 12:18] * b_optimized

            traj_kinematics = np.vstack([traj[:, 3:12] for traj in x_traj])
            
            np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc_randB_2mln', x_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_b_vboc_randB_2mln', b_optimized)
            np.save(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc_randB_2mln', n_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_status_vboc_randB_2mln', status_list)
            # === SALVATAGGIO DEI FALLIMENTI ===
            np.save(f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc_randB', np.array(all_failed_q_init))

            np.save(f'{params.DATA_DIR}{robotic_system}_actual_boxes_vboc_randB_2mln', actual_boxes)
            np.save(f'{params.DATA_DIR}{robotic_system}_traj_kinematics_vboc_randB_2mln', traj_kinematics)
            np.save(f'{params.DATA_DIR}{robotic_system}_u_traj_vboc_randB', np.vstack(u_traj))

            np.save(f'{params.DATA_DIR}{robotic_system}_TEST_dataset_classification_2mln', np.array(all_test_dataset))

            
            solved = len(x_data)
            print(f'Batch {nb}: Total number of points saved until now: {solved}')

        t_end_step = time.perf_counter()
        
        print(f'Training time: {t_end_step - t_start_step:.2f} seconds')

        print('Total number of points solved: %d' % solved)


 # =========================================================================
    # PLOT
# =================================================

    #in 3D
        # --- Plot generated trajectories ---
        if params.plot:

            pose_label = ['x [m]', 'y [m]', 'z [m]', 'phi [deg]', 'theta [deg]', 'psi [deg]']
            vel_label = ['v_x [m/s]', 'v_y [m/s]', 'v_z [m/s]', 'p [deg/s]', 'q [deg/s]', 'r [deg/s]']

            traj_dir = os.path.join(plots_dir, 'trajectories')
            pose_dir = os.path.join(plots_dir, 'poses')
            velocity_dir = os.path.join(plots_dir, 'velocities')
            input_dir = os.path.join(plots_dir, 'inputs')
            threeD_dir = os.path.join(plots_dir, '3D_flights')
            
            plots_subdirs = [traj_dir, pose_dir, velocity_dir, input_dir, threeD_dir]
            for subdir in plots_subdirs:
                ensure_clean_dir(subdir)

            normals = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

            sub_plot = 1 if params.check else max(1, params.prob_num // 10)

            for k in range(len(x_traj)):
                if k % sub_plot == 0:
                    horizon_ = x_traj[k].shape[0]
                    colors = np.linspace(0, 1, horizon_)
                    t = np.linspace(0, horizon_ * params.dt, horizon_)

                    # Estrazione box 3D
                    scale = x_data[k, 18]
                    l_x_plus, l_y_plus, l_z_plus = x_data[k, 12:15]
                    l_x_minus, l_y_minus, l_z_minus = x_data[k, 15:18]

                    box_max = np.array([scale * l_x_plus, scale * l_y_plus, scale * l_z_plus])
                    box_min = np.array([-scale * l_x_minus, -scale * l_y_minus, -scale * l_z_minus])

                    # 1. Poses
                    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
                    ax = ax.flatten()
                    for i in range(nq):
                        ax[i].grid(True)
                        if i < model.npos:
                            line, = ax[i].plot(t, x_traj[k][:, i], label=f'{pose_label[i]}')
                            ellips_r = [np.sqrt(normals[i].T @ model.Q(x_traj[k][h, :]).full() @ normals[i]) for h in range(len(t))]
                            ax[i].plot(t, x_traj[k][:, i] + ellips_r, color=line.get_color(), linestyle='--')
                            ax[i].plot(t, x_traj[k][:, i] - ellips_r, color=line.get_color(), linestyle='--')
                            ax[i].axhline(box_max[i], color='r', linestyle=':', label='Box Max')
                            ax[i].axhline(box_min[i], color='r', linestyle=':', label='Box Min')
                        else:
                            ax[i].plot(t, np.rad2deg(x_traj[k][:, i]), label=f'{pose_label[i]}')
                        ax[i].set_xlabel('Time [s]')
                        ax[i].legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(pose_dir, f'pose_{k + 1}.png'))
                    plt.close(fig)

                    # 2. Velocities
                    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
                    ax = ax.flatten()
                    for i in range(nq):
                        ax[i].grid(True)
                        data_to_plot = x_traj[k][:, nq + i] if i < model.npos else np.rad2deg(x_traj[k][:, nq + i])
                        ax[i].plot(t, data_to_plot, label=f'{vel_label[i]}')
                        ax[i].set_xlabel('Time [s]')
                        ax[i].legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(velocity_dir, f'vel_{k + 1}.png'))
                    plt.close(fig)

                    # 3. Inputs (4 motori)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for i in range(nu):
                        ax.plot(t, u_traj[k][:, i], label=f'Motor {i + 1}')
                    ax.axhline(model.u_bar, color='r', linestyle='--', label='Max Power')
                    ax.grid(True)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(input_dir, f'input_{k + 1}.png'))
                    plt.close(fig)

                    # 4. 3D Flight Path
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    sc = ax.scatter(
                        x_traj[k][:, 0], x_traj[k][:, 1], x_traj[k][:, 2],
                        c=colors, cmap='coolwarm', s=10
                    )
                    
                    # Disegna gli assi del drone lungo la traiettoria
                    step_arrow = max(1, len(x_traj[k]) // 10)
                    for i in range(0, len(x_traj[k]), step_arrow):
                        phi, theta, psi = x_traj[k][i, 3:6]
                        # Rotazione da Eulero ZYX (Yaw, Pitch, Roll) a matrice
                        rot_matrix = Rot.from_euler('ZYX', [psi, theta, phi]).as_matrix()
                        x_arrow = rot_matrix[:, 0] * model.min_width
                        y_arrow = rot_matrix[:, 1] * model.min_length
                        z_arrow = rot_matrix[:, 2] * model.min_height
                        
                        ax.quiver(x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                                  x_arrow[0], x_arrow[1], x_arrow[2], color='r')
                        ax.quiver(x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                                  y_arrow[0], y_arrow[1], y_arrow[2], color='g')
                        ax.quiver(x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
                                  z_arrow[0], z_arrow[1], z_arrow[2], color='b')
                    
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('Z [m]')
                    ax.set_title(f'3D Flight Path {k + 1}')
                    set_axes_equal(ax)
                    plt.tight_layout()
                    plt.savefig(os.path.join(threeD_dir, f'3D_traj_{k + 1}.png'))
                    plt.close(fig)
    
    # =========================================================================
    # NEURAL NETWORK TRAINING
    # =========================================================================
    if params.training: 

        # --- Load data ---
        x_data = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc_randB_2mln.npy')
        b_data = np.load(f'{params.DATA_DIR}{robotic_system}_b_vboc_randB_2mln.npy')
        b_all_data = np.load(params.DATA_DIR + 'sth_b_all_vboc.npy')
        d_data = np.load(params.DATA_DIR + 'sth_d_vboc.npy')
        status_data = np.load(params.DATA_DIR + 'sth_status_vboc_randB_2mln.npy')

        actual_boxes = np.load(f'{params.DATA_DIR}{robotic_system}_actual_boxes_vboc_randB_2mln.npy')
        traj_kinematics = np.load(f'{params.DATA_DIR}{robotic_system}_traj_kinematics_vboc_randB_2mln.npy')
        u_traj = np.load(f'{params.DATA_DIR}{robotic_system}_u_traj_vboc_randB_2mln.npy')
        n_data = np.load(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc_randB.npy')
        

        # --- Histograms of raw data distributions ---
        if params.plot:
            hist_dir = os.path.join(plots_dir, 'histograms_randB')
            ensure_clean_dir(hist_dir)
            
            
            plot_histogram(
                #in 3D
                x_data[:, 3:12],
                title="Inputs_Angles_and_Velocities", 
                xlabel="Value",
                ylabel="Frequency",
                subplot_titles=["\\psi_0","$\\theta_0$","$\\phi_0", "$v_x0$", "v_y0", "$v_z0$", "$\omega_x0$","$\omega_y0$", "$\omega_z0$"], 
                bins=50, 
                saving_dir=hist_dir
            )
            # 2. Target (Scaling)
            plot_histogram(
                b_data, 
                title="Target_Scaling_Factor", 
                xlabel="Value [m]", 
                ylabel="Frequency", 
                bins=50, 
                saving_dir=hist_dir
            )
            # 3. Solver Status
            plot_histogram(
                status_data, 
                title="Solver_Status",
                xlabel="Status Code", 
                ylabel="Frequency", 
                bins=3, 
                saving_dir=hist_dir,
                xticks=[0, 2, 4]
            )

            # 5. Dimensioni del Box Normalizzato
            plot_histogram(
                x_data[:, 12:18],
                title="5_Normalized_Box_Dimensions", 
                xlabel="Normalized Value",
                ylabel="Frequency",
                subplot_titles=["$X_{max}$ norm", "$Y_{max}$ norm", "$Z_{max}$ norm", "$X_{min}$ norm", "$Y_{min}$ norm","$Z_{min}$ norm"],
                bins=50, 
                saving_dir=hist_dir
            )

            ## 4. Istogramma degli Orizzonti di Convergenza (N)       
            plot_histogram(
                n_data, 
                title="Distribution_of_Converged_Horizons_N", 
                xlabel="Horizon Length (N steps)", 
                ylabel="Frequency", 
                bins=20, # Lascia un numero generico di colonne, deciderà lui l'ampiezza
                saving_dir=hist_dir
            )

            # 6. Dimensioni del Box Effettivo (Lati calcolati in metri)
            plot_histogram(
                actual_boxes,
                title="Actual_Box_Dimensions", 
                xlabel="Length [m]",
                ylabel="Frequency",
                subplot_titles=["$X_{max}$ [m]", "$Y_{max}$ [m]", "$Z_{max}$ [m]", "$X_{min}$ [m]", "$Y_{min}$ [m]", "$Z_{min}$ [m]"],
                bins=50, 
                saving_dir=hist_dir
            )

            # 5. Istogramma dei casi FALLITI
            failed_file = f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc_randB.npy'
            if os.path.exists(failed_file):
                failed_data = np.load(failed_file)
                if len(failed_data) > 0:
                    # In 3D
                    plot_histogram(
                        failed_data[:, 3:12],
                        title="Failed_Cases_Initial_Conditions", 
                        xlabel="Value",
                        ylabel="Frequency", 
                        bins=20, 
                        saving_dir=hist_dir
                    )

        # 7. Traiettorie:
        traj_data_to_plot = np.hstack((
            traj_kinematics,
            u_traj # motori
        ))

        plot_histogram(
                traj_data_to_plot,
                title="Trajectory_Kinematics_and_Motors", 
                xlabel="Value",
                ylabel="Frequency",
                subplot_titles=["Traj $\\phi$", "Traj $\\theta$", "Traj $\\psi$", "Traj $v_x$",  "Traj $v_y$", "Traj $v_z$",  "Traj $\\omega_x$", "Traj $\\omega_y$",  "Traj $\\omega_z$", "Motor $u_1$", "Motor $u_2$", "Motor $u_3$", "Motor $u_4$"], 
                bins=50,
                saving_dir=hist_dir
            )

        # Drop position columns and prepend box dimensions as input features
        #x_data = np.hstack((b_data, x_data[:, model.npos:]))
        #in 3D
        dataset = np.hstack(( x_data[:, 3:12],x_data[:, 12:18], b_data))
        np.random.shuffle(dataset)

        # Dividiamo in Input (x_data) e Target (y_data)
        #in 3D
        x_data = dataset[:, :15]
        y_data = dataset[:, 15:]

        # --- Shuffle and split into training / validation / test sets ---
        #np.random.shuffle(x_data)
        n = len(x_data)        
        #nbori = model.nbox + model.nori
        train_size = int(params.train_ratio * n)
        val_size = int(params.val_ratio * n)
        test_size = n - train_size - val_size
        
        x_train = x_data[:train_size]
        x_val = x_data[train_size:train_size + val_size]
        x_test = x_data[train_size + val_size:]

        # --- Standardize box + orientation features using training statistics 
        # ---
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        for x_input in [x_train, x_val, x_test]:
            x_input[:] = (x_input - mean) / std

        # --- Split outputs ---
        y_train = y_data[:train_size]
        y_val = y_data[train_size:train_size + val_size]
        y_test = y_data[train_size + val_size:]
        
        # --- Build model, loss, and optimiser ---
        #nx_train = nbori+model.nv
        #in 3D
        nx_train = 15
    

        nn_model = NeuralNetwork(
            nx_train, 
            params.hidden_size, 
            1,
            params.hidden_layers, 
            act_fun, 
            ub
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), 
                                     lr=params.learning_rate,
                                     weight_decay=2e-5,
                                     amsgrad=True)
        regressor = RegressionNN(params, nn_model, loss_fn, optimizer)

        # --- Convert arrays to tensors ---
        x_train = torch.Tensor(x_train).to(device)
        y_train = torch.Tensor(y_train).to(device)
        x_val = torch.Tensor(x_val).to(device)
        y_val = torch.Tensor(y_val).to(device)
        x_test = torch.Tensor(x_test).to(device)
        y_test = torch.Tensor(y_test).to(device)

        t_start_step = time.perf_counter()

        # --- Train ---
        print('***START TRAINING***\n')
        train_val_dir = os.path.join(plots_dir, 'training_validation_randB')
        ensure_clean_dir(train_val_dir)

        train_evol, val_evol = regressor.training(
            x_train, 
            y_train, 
            x_val, 
            y_val, 
            args['epochs']
        )
        print('***TRAINING COMPLETED***\n')

        t_end_step = time.perf_counter()

        print(f'Training time: {t_end_step - t_start_step:.2f} seconds')

        # --- Evaluate on training+validation and test sets ---
        print('***MODEL EVALUATION***')
        rmse_train, rel_err = regressor.testing(
            torch.cat((x_train, x_val), dim=0), 
            torch.cat((y_train, y_val), dim=0)
        )
        print(f'RMSE on Training data: {rmse_train:.5f}')
        print('Maximum error wrt training data: ' f'{torch.max(torch.abs(rel_err)).item():.5f}')
        rmse_test, rel_err = regressor.testing(x_test, y_test)
        print('---')
        print(f'RMSE on Test data: {rmse_test:.5f}')
        print('99 % of the data has a relative error lower than: ' \
              f'{torch.quantile(rel_err, 0.99).item():.5f}%')
        print(f'Maximum relative error wrt test data: {torch.max(torch.abs(rel_err)).item():.5f}')
        print('*---*---*---*\n')

        # --- Save model weights and normalisation statistics ---
        torch.save({
            'model': nn_model.state_dict(),
            'mean': mean,
            'std': std,
        }, nn_filename)

        # --- Plot training and validation loss curves ---
        loss_dir = os.path.join(plots_dir, 'loss_evolution_randB')
        ensure_clean_dir(loss_dir)
        fig = plt.figure(figsize=(10, 6))
        plt.grid(True, which='both')
        plt.semilogy(train_evol, label='Training', c='b', lw=2)
        plt.semilogy(val_evol, label='Validation', c='g', lw=2)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (LP filtered)')
        plt.title(f'Training evolution, horizon {N}')
        plt.savefig(os.path.join(loss_dir, f'evolution_{N}.png'))
        plt.close(fig)

    # =========================================================================
    # VIABILITY KERNEL PLOTTING
    # =========================================================================
    # if params.plot and not params.generation: 
        
    #     # --- Load trained network ---
    #     device = torch.device("cpu")
    #     nbori = model.nbox+model.nori
    #     nx_train = nbori+model.nv
    #     #nn_data = torch.load(nn_filename)
    #     nn_data = torch.load(nn_filename, map_location=device)
    #     nn_model = NeuralNetwork(
    #         nx_train, 
    #         params.hidden_size, 
    #         1, 
    #         params.hidden_layers, 
    #         act_fun, 
    #         ub
    #     ).to(device)        
    #     nn_model.load_state_dict(nn_data['model'])
    #     print('***PLOTTING BRS***\n')

    #     # Compute fixed-direction section data if not already cached
    #     if not os.path.exists(
    #         f'{params.DATA_DIR}{robotic_system}_x_fixed_vboc.npy'
    #     ):
    #         x_fixed, x_status = fixed_velocity_dir(
    #             N, 
    #             N_increment, 
    #             vboc_repeat, 
    #             n_pts=100
    #         )
    #         np.save(
    #             f'{params.DATA_DIR}{robotic_system}_x_fixed_vboc', 
    #             np.array(x_fixed, dtype=object), 
    #             allow_pickle=True
    #         )
    #         np.save(f'{params.DATA_DIR}{robotic_system}_status_fixed_vboc', 
    #                 np.array(x_status, dtype=object), 
    #                 allow_pickle=True
    #         )
    #     else:
    #         x_fixed = np.load(
    #             f'{params.DATA_DIR}{robotic_system}_x_fixed_vboc.npy',
    #             allow_pickle=True
    #         )
    #         x_status = np.load(
    #             f'{params.DATA_DIR}{robotic_system}_status_fixed_vboc.npy',
    #             allow_pickle=True
    #         )

    #     brs_dir = os.path.join(plots_dir, 'brs')
    #     ensure_clean_dir(brs_dir)

    #     plot_brs(
    #         params, 
    #         model, 
    #         controller, 
    #         nn_model, 
    #         nn_data['mean'], 
    #         nn_data['std'], 
    #         x_fixed, 
    #         x_status
    #     )
 
    # print('***ALL DONE***')
    # elapsed_time = time.time() - start_time
    # hours = int(elapsed_time // 3600)
    # minutes = int((elapsed_time % 3600) // 60)
    # seconds = int(elapsed_time % 60)
    # print(f'Elapsed time: {hours}:{minutes:2d}:{seconds:2d}')

    # os.system('aplay /home/maxbertus/Music/notification.wav > /dev/null 2>&1')


    # =========================================================================
    # VIABILITY KERNEL PLOTTING 3D
    # =========================================================================
    if params.plot and not params.generation: 
        
        # --- Load trained network ---
        device = torch.device("cpu")
        nx_train = 9
        
        nn_data = torch.load(nn_filename, map_location=device)
        nn_model = NeuralNetwork(
            nx_train, 
            params.hidden_size, 
            1,
            params.hidden_layers, 
            act_fun, 
            ub
        ).to(device)        
        nn_model.load_state_dict(nn_data['model'])
        print('***PLOTTING BRS***\n')

        brs_dir = os.path.join(plots_dir, 'brs')
        ensure_clean_dir(brs_dir)

        # Ricarichiamo i dati simulati per i puntini nel grafico
        x_data_raw = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc.npy')
        # Estraiamo solo i 4 input [theta, vx, vz, wy] per il plot BRS
        x_data_plot = x_data_raw[:, 3:12]

        # Chiamata pulita
        plot_brs(
            params, 
            model, 
            controller, 
            nn_model, 
            nn_data['mean'], 
            nn_data['std'],
            x_data_plot
        )
    
    # =========================================================================
        # 2. PLOT DELL'ANALISI DI SENSIBILITA' (Velocità vs Dimensione Stanza 3D)
        # =========================================================================
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        # Nel 3D, gli indici in x_data_raw sono: vx=6, vy=7, vz=8, scale=18
        v_x = x_data_raw[:, 6]
        v_y = x_data_raw[:, 7]
        v_z = x_data_raw[:, 8]
        scale = x_data_raw[:, 18]
        
        # v_x vs Scaling
        ax[0].scatter(v_x, scale, alpha=0.5, s=5, c='blue')
        ax[0].set_xlabel('v_x [m/s]', fontsize=12)
        ax[0].set_ylabel('Optimized Scale [m]', fontsize=12)
        ax[0].set_title('Dipendenza da v_x')
        ax[0].grid(True)

        # v_y vs Scaling
        ax[1].scatter(v_y, scale, alpha=0.5, s=5, c='green')
        ax[1].set_xlabel('v_y [m/s]', fontsize=12)
        ax[1].set_title('Dipendenza da v_y')
        ax[1].grid(True)

        # v_z vs Scaling
        ax[2].scatter(v_z, scale, alpha=0.5, s=5, c='red')
        ax[2].set_xlabel('v_z [m/s]', fontsize=12)
        ax[2].set_title('Dipendenza da v_z')
        ax[2].grid(True)

        plt.suptitle('Dimensioni della stanza in funzione delle velocità iniziali (3D)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(brs_dir, 'Velocities_vs_Scale_3D.png'))
        plt.close(fig)