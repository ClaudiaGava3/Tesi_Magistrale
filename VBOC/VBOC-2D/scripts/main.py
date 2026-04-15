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
from src.vboc.abstract import Model
from src.vboc.controller import ViabilityController
from src.vboc.learning import NeuralNetwork, NovelNeuralNetwork, RegressionNN, plot_brs
from vboc.parser import Parameters, parse_args

install()

progress_var = Value('i', 0)
np.set_printoptions(linewidth=np.inf)


def plot_histogram(
    data: np.ndarray,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
    saving_dir: str = "plots/histograms/",
    xticks: list = None
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
    #in 2D
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

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
        axes[i].set_title(f"Dimension {i+1}")
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
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
    #in 2D
    x_guess = np.zeros((N_guess, model.nx))
    # Manteniamo la posizione piatta e lo scaling costante per tutto il guess
    x_guess[:, :2] = q_init[:2]  
    # Guess iniziale per i 4 lati (immaginiamo che partano quadrati a proporzione 1.0)
    x_guess[:, 6:10] = 1.0
    x_guess[:, 10] = box_guess 

    # MA al primissimo istante di tempo (nodo 0), diciamo ad Acados esattamente
    # come parte il drone nella realtà (incluse le velocità e gli angoli veri)
    x_guess[0, :6] = q_init

    # x_static = np.hstack((q_init, np.zeros(model.nx - model.nq)))
    # gravity_wrench = np.array([0, 0, model.mass * model.g])
    # allocation_matrix = model.R(x_static).full() @ model.F
    # u_hover = np.linalg.pinv(allocation_matrix) @ gravity_wrench

    # x_static = np.hstack((q_init, np.full(4, box_guess)))
    # allocation_matrix = model.R(x_static).full() @ model.F

    # 1. Definiamo uno stato orizzontale per il guess iniziale
    x_flat = np.zeros(11)
    x_flat[:2] = q_init[:2]  # Copia solo le posizioni (X, Z)
    x_flat[10] = box_guess    # Copia lo scaling (Indice 6 in 2D)

    # 2. Costruiamo la matrice di allocazione per il 2D: Matrice 6x2.
    allocation_matrix = np.vstack((model.F, model.M))

    # 3. Obiettivo fisico: [Fx, Fy, Fz, Mx, My, Mz]
    # Bilanciamo la gravità su Z e annulliamo le coppie
    wrench_hover = np.array([0.0, 0.0, model.mass * model.g, 0.0, 0.0, 0.0])
    
    # 4. Spinta bilanciata per i 2 motori
    u_hover = np.linalg.pinv(allocation_matrix) @ wrench_hover
    u_guess = np.full((N_guess, model.nu), u_hover)

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
    nn_filename = f'{params.NN_DIR}{robotic_system}_{params.act}.pt'
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
            #in 2D
            orient_init = np.random.uniform(-max_phi, max_phi, params.prob_num).reshape(-1, 1)

            # Generiamo VELOCITÀ casuali (vx, vz, wy)
            vel_init = np.random.uniform(-1.0, 1.0, (params.prob_num, model.nv))

        # Creiamo il vettore di stato iniziale di 6 elementi: [x, z, theta, vx, vz, wy]
        q_init = np.hstack([pos_init, orient_init, vel_init])

        # --- Obstacle box bounds --- 
        box_guess=1e1
        # box_min_values = np.empty((params.prob_num, model.npos))
        # box_max_values = np.empty((params.prob_num, model.npos))
        # if params.check:
        #     # Fixed maximum box in check mode
        #     for i in range(params.prob_num):
        #         box_min_values[i,:] = np.array(
        #             [model.max_width, model.max_length, model.max_height]
        #         )
        #         box_max_values[i,:] = np.array(
        #             [model.max_width, model.max_length, model.max_height]
        #         )
        # else:
        #     # Random box, with minimum size determined by the drone's 
        #     # ellipsoidal occupancy
        #     for i in range(params.prob_num):
        #         min_dx = np.sqrt(
        #             np.array([1,0,0]) @ 
        #             model.Q(np.hstack([q_init[i,:], np.zeros(model.nv)])) @ 
        #             np.array([1,0,0]).T
        #         )
        #         min_dy = np.sqrt(
        #             np.array([0,1,0]) @ 
        #             model.Q(np.hstack([q_init[i,:], np.zeros(model.nv)])) @ 
        #             np.array([0,1,0]).T
        #         )
        #         min_dz = np.sqrt(
        #             np.array([0,0,1]) @ 
        #             model.Q(np.hstack([q_init[i,:], np.zeros(model.nv)])) @ 
        #             np.array([0,0,1]).T
        #         )
        #         dx = np.random.uniform(min_dx, model.max_width)
        #         dy = np.random.uniform(min_dy, model.max_length)
        #         dz = np.random.uniform(min_dz, model.max_height)
        #         box_min_values[i, :] = np.array([dx, dy, dz])
        #         dx = np.random.uniform(min_dx, model.max_width)
        #         dy = np.random.uniform(min_dy, model.max_length)
        #         dz = np.random.uniform(min_dz, model.max_height)
        #         box_max_values[i, :] = np.array([dx, dy, dz])

        # --- Random seeds, one per problem ---
        randomSeeds = [random.randint(0, params.prob_num) 
                       for _ in range(params.prob_num)
        ]

        # --- Accumulators for results across all batches ---
        #all_x_0, all_x_t, all_u_t, all_b_m, all_b_M, all_status, all_d_list = \
        #[],[],[],[],[],[],[]
        all_x_0, all_x_t, all_u_t, all_n_final, all_status = [], [], [], [], []
        all_failed_q_init = [] # <--- NUOVA LISTA PER I FALLIMENTI

        # Split the problems into sub-batches to allow intermediate saves
        if params.check:
            sub_batch = 1
        else:
            sub_batch = 100
        n_batch = int(params.prob_num/sub_batch)

        # # --- Per-batch storage (overwritten at each save) ---
        # x_data, x_traj, u_traj, b_min, b_max = [], [], [], [], []
        # solved = 0

        # print('Start data generation')
        # for nb in range(n_batch):  
        #     with Pool(params.cpu_num) as p:
        #         res = p.starmap(
        #             compute_data_on_border, 
        #             [(q0, N, N_increment, vboc_repeat, box_min, box_max, 
        #               randomSeeds) for q0, box_min, box_max, randomSeeds in 
        #               zip(q_init[(nb*sub_batch):((nb+1)*sub_batch)], 
        #                   box_min_values[(nb*sub_batch):((nb+1)*sub_batch)], 
        #                   box_max_values[(nb*sub_batch):((nb+1)*sub_batch)], 
        #                   randomSeeds[(nb*sub_batch):((nb+1)*sub_batch)])]
        #         )
        

            # --- Unpack parallel results ---
            # x_0, x_t, u_t, b_m, b_M, status, d_list = zip(*res)
            # all_x_0.extend(x_0)
            # all_x_t.extend(x_t)
            # all_u_t.extend(u_t)
            # all_b_m.extend(b_m)
            # all_b_M.extend(b_M)
            # all_status.extend(status)
            # all_d_list.extend(d_list)

            # # Warn and skip if no feasible solution was found in this batch
            # if all(item is None for item in x_0):
            #     warnings.warn(f'No solution found for any problem in batch' \
            #                   '{nb}. Skipping this batch.', RuntimeWarning)
            #     print(status)
            #     continue
            # # Abort if no feasible solution has been found across all batches 
            # # so far
            # if all(item is None for item in all_x_0):
            #     warnings.warn('No solution found for any problem. ' \
            #     'Exiting the program.', RuntimeWarning)
            #     print(status)
            #     exit()

        #in 2D
        ref_box = np.array([1.0, 1.0, 1.0, 1.0]) # Box di riferimento 1:1



        print('Start data generation')
        for nb in range(n_batch):  
            with Pool(params.cpu_num) as p:
                res = p.starmap(
                    compute_data_on_border, 
                    [(q0, ref_box, box_guess, N, N_increment, vboc_repeat) 
                     for q0 in q_init[(nb*sub_batch):((nb+1)*sub_batch)]]
                )

            # --- Unpack parallel results (Ora sono 5, non più 7!) ---
            x_0, x_t, u_t, n_final_list, status = zip(*res)
            all_x_0.extend(x_0)
            all_x_t.extend(x_t)
            all_u_t.extend(u_t)
            all_n_final.extend(n_final_list)
            all_status.extend(status)

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

            # # --- Intermediate save: filter out failed problems ---
            # x_data = np.vstack([i for i in all_x_0 if i is not None])
            # x_traj = [i for i in all_x_t if i is not None]
            # u_traj = [i for i in all_u_t if i is not None]
            # b_min = list(all_b_m)
            # b_max = list(all_b_M)
            # d = list(all_d_list)
            # status = list(all_status)
            # b_combined = np.vstack([np.hstack((b_min[i], b_max[i])) 
            #                         for i in range(len(b_min))]
            # )
            # np.save(f'{params.DATA_DIR}{robotic_system}_d_vboc', d)
            # np.save(
            #     f'{params.DATA_DIR}{robotic_system}_b_all_vboc', b_combined
            # )
            # np.save(f'{params.DATA_DIR}{robotic_system}_status_vboc', status)
            # b_min_succ = [all_b_m[i] for i in range(len(all_b_m)) 
            #               if all_x_0[i] is not None
            # ]
            # b_max_succ = [all_b_M[i] for i in range(len(all_b_M)) 
            #               if all_x_0[i] is not None
            # ]
            # b_combined_succ = np.vstack(
            #     [np.hstack((b_min_succ[i], b_max_succ[i])) 
            #      for i in range(len(b_min_succ))
            #     ]
            # )
            # solved = len(x_data)
            # print(f'Batch {nb}: Total number of points saved until now: %d' 
            #       % solved
            # )
            # np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc', x_data)
            # np.save(
            #     f'{params.DATA_DIR}{robotic_system}_b_vboc', b_combined_succ
            # )

            x_data = np.vstack([i for i in all_x_0 if i is not None])
            x_traj = [i for i in all_x_t if i is not None]
            u_traj = [i for i in all_u_t if i is not None]
            n_data = np.array([all_n_final[i] for i in range(len(all_n_final)) if all_x_0[i] is not None])  # Filtriamo N_final usando la stessa esatta logica di x_data per mantenere l'allineamento
            status_list = list(all_status)
            
            # Il "box" ottimizzato è solo un fattore di scala!
            b_optimized = x_data[:, 10].reshape(-1, 1)

            np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc', x_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_b_vboc', b_optimized)
            np.save(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc', n_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_status_vboc', status_list)
            # === SALVATAGGIO DEI FALLIMENTI ===
            np.save(f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc', np.array(all_failed_q_init))
            
            solved = len(x_data)
            print(f'Batch {nb}: Total number of points saved until now: {solved}')

        print('Total number of points solved: %d' % solved)


 # =========================================================================
    # PLOT
# =========================================================================
    # versione precedente
        # # --- Plot generated trajectories ---
        # if params.plot:

        #     # Labels and titles for pose/velocity subplots
        #     extended_pose_title = ['Position', 'Orientation', 'Inclination']
        #     velocities_title = ['Linear velocity', 'Angular velocity']
        #     pose_label = [
        #         'x [m]', 'y [m]', 'z [m]', 'r [deg]', 'p [deg]', 'y [deg]'
        #     ]
        #     vel_label = [
        #         'v$_x$ [m/s]', 'v$_y$ [m/s]', 'v$_z$ [m/s]',
        #           '$\omega_x$ [deg/s]', '$\omega_y$ [deg/s]', 
        #           '$\omega_z$ [deg/s]'
        #     ]
        #     y_lab_pose = ['Pos. [m]', 'Orient. [deg]', 'Incl. [deg]']
        #     y_lab_vel = ['v [m/s]', '$\omega$ [deg/s]']

        #     # Create (or recreate) output subdirectories
        #     traj_dir = os.path.join(plots_dir, 'trajectories')
        #     pose_dir = os.path.join(plots_dir, 'poses')
        #     velocity_dir = os.path.join(plots_dir, 'velocities')
        #     input_dir = os.path.join(plots_dir, 'inputs')
        #     threeD_dir = os.path.join(plots_dir, '3D')
        #     plots_subdirs = [
        #         traj_dir, pose_dir, velocity_dir, input_dir, threeD_dir
        #     ]
        #     for subdir in plots_subdirs:
        #         ensure_clean_dir(subdir)

        #     # Unit normals used to project the occupancy ellipsoid onto each 
        #     # axis
        #     normals = [
        #         np.array([1,0,0]),
        #         np.array([0,1,0]),
        #         np.array([0,0,1]) 
        #     ]

        #     # Plot every trajectory, or 1 in 10 outside check mode
        #     if params.check:
        #         sub_plot = 1
        #     else:
        #         sub_plot = params.prob_num / 10

        #     for k in range(len(x_traj)):
        #         if k % sub_plot == 0:
        #             horizon_ = x_traj[k].shape[0]
        #             colors = np.linspace(0, 1, horizon_)
        #             t = np.linspace(0, horizon_ * params.dt, horizon_)

        #             traj_xlim_min = (-b_min[k]).tolist() + \
        #                 [-np.rad2deg(max_phi), -np.rad2deg(max_phi), -180.0]
        #             traj_xlim_max = b_max[k].tolist() + \
        #                 [np.rad2deg(max_phi), np.rad2deg(max_phi), 180.0]

        #             # Phase-plane plot: position vs velocity for each DOF
        #             fig, ax = plt.subplots(2, 3)
        #             ax = ax.reshape(-1)
        #             for i in range(nq):
        #                 ax[i].grid(True, linewidth=0.5)
        #                 if i < model.npos:
        #                     ax[i].scatter(
        #                         x_traj[k][:, i], x_traj[k][:, nq + i],
        #                         c=colors, 
        #                         cmap='coolwarm', 
        #                         s=1
        #                     )
        #                 else:
        #                     ax[i].scatter(
        #                         np.rad2deg(x_traj[k][:, i]), 
        #                         np.rad2deg(x_traj[k][:, nq + i]), 
        #                         c=colors, 
        #                         cmap='coolwarm', 
        #                         s=1
        #                     )
        #                 ax[i].set_xlim([traj_xlim_min[i], traj_xlim_max[i]])
        #                 ax[i].set_xlabel(f'{pose_label[i]}')
        #                 ax[i].set_ylabel(f'{vel_label[i]}')
        #             plt.suptitle(f'Trajectory {k + 1}, d {all_d_list[k][:3]}')
        #             plt.tight_layout()
        #             plt.savefig(os.path.join(traj_dir, f'traj_{k + 1}.png'))
        #             plt.close(fig)

        #             # Pose over time, with occupancy ellipsoid bounds for 
        #             # position DOFs
        #             fig, ax = plt.subplots(3, 1)
        #             ax = ax.reshape(-1)
        #             j = 0
        #             for i in range(nq):
        #                 if i == model.npos:
        #                     j += 1
        #                 ax[j].grid(True)
        #                 ax[j].set_title(f'{extended_pose_title[j]}')
        #                 if i < model.npos:
        #                     line, = ax[j].plot(
        #                         t, x_traj[k][:, i], label=f'{pose_label[i]}'
        #                     )
        #                     ellips_r = []
        #                     for h in range(len(t)):
        #                         ellips_r.append(
        #                             np.sqrt(normals[i].T @ 
        #                                     model.Q(x_traj[k][h, :]) @ 
        #                                     normals[i])
        #                         )
        #                     ax[j].plot(
        #                         t, 
        #                         x_traj[k][:, i] + ellips_r, 
        #                         color=line.get_color(), 
        #                         linestyle='--', 
        #                         linewidth=0.8
        #                     )
        #                     ax[j].plot(
        #                         t, 
        #                         x_traj[k][:, i] - ellips_r, 
        #                         color=line.get_color(), 
        #                         linestyle='--', 
        #                         linewidth=0.8
        #                     )
        #                 else:
        #                     line, = ax[j].plot(
        #                         t,
        #                         np.rad2deg(x_traj[k][:, i]), 
        #                         label=f'{pose_label[i]}'
        #                     )
        #                 # ax[j].axhline(
        #                 # traj_xlim_min[i], 
        #                 # color=line.get_color(), 
        #                 # linestyle='--', 
        #                 # linewidth=0.8)
        #                 # ax[j].axhline(
        #                 # traj_xlim_max[i], 
        #                 # color=line.get_color(), 
        #                 # linestyle='--', 
        #                 # linewidth=0.8)
        #                 ax[j].set_xlabel('Time [s]')
        #                 ax[j].set_ylabel(y_lab_pose[j])
        #                 ax[j].legend()
        #             j += 1
        #             # Last subplot: total inclination angle with safety 
        #             # thresholds
        #             ax[j].grid(True)
        #             ax[j].set_title(f'{extended_pose_title[j]}')
        #             line, = ax[j].plot(
        #                 t, 
        #                 np.rad2deg(np.sqrt(np.square(x_traj[k][:, 3]) + 
        #                                    np.square(x_traj[k][:, 4]))),
        #                 label=f'{pose_label[i]}'
        #             )
        #             # ax[j].axhline(
        #             # min_phi, 
        #             # color=line.get_color(), 
        #             # linestyle='--', 
        #             # linewidth=0.8)
        #             ax[j].axhline(
        #                 np.rad2deg(max_phi), 
        #                 color=line.get_color(), 
        #                 linestyle='--', 
        #                 linewidth=0.8
        #             )
        #             ax[j].axhline(
        #                 np.rad2deg(model.phi_hovering_max), 
        #                 color='r', 
        #                 linestyle='--', 
        #                 linewidth=0.8
        #             )
        #             ax[j].set_xlabel('Time [s]')
        #             ax[j].set_ylabel(y_lab_pose[j])
        #             ax[j].legend()

        #             plt.suptitle(f'Trajectory {k + 1}')
        #             plt.tight_layout()
        #             plt.savefig(os.path.join(pose_dir, f'pose_{k + 1}.png'))
        #             plt.close(fig)

        #             # Linear and angular velocity over time
        #             fig, ax = plt.subplots(2, 1)
        #             ax = ax.reshape(-1)
        #             j = 0
        #             for i in range(nq):
        #                 if i == model.npos:
        #                     j += 1
        #                 ax[j].grid(True)
        #                 ax[j].set_title(f'{velocities_title[j]}')
        #                 if i < model.npos:
        #                     line, = ax[j].plot(
        #                         t, 
        #                         x_traj[k][:, i + nq], 
        #                         label=f'{vel_label[i]}'
        #                     )
        #                 else:
        #                     line, = ax[j].plot(
        #                         t, 
        #                         np.rad2deg(x_traj[k][:, i + nq]), 
        #                         label=f'{vel_label[i]}'
        #                     )
        #                 ax[j].set_xlabel('Time [s]')
        #                 ax[j].set_ylabel(y_lab_vel[j])
        #                 ax[j].legend()
        #             plt.suptitle(f'Trajectory {k + 1}')
        #             plt.tight_layout()
        #             plt.savefig(os.path.join(velocity_dir, f'vel_{k + 1}.png'))
        #             plt.close(fig)

        #             # Control inputs over time
        #             offset = 200
        #             fig, ax = plt.subplots()
        #             for i in range(nu):
        #                 ax.grid(True)
        #                 ax.plot(t, u_traj[k][:, i], label=f'u_{i + 1}')
        #                 ax.set_title('Inputs')
        #                 ax.axhline(
        #                     model.u_bar, 
        #                     color='r', 
        #                     linestyle='--', 
        #                     lw=0.8
        #                 )
        #                 ax.set_xlabel('Time [s]')
        #                 ax.set_ylabel('$u^2$ [(Hz/s)$^2$]')
        #                 ax.set_ylim([0.0 - offset, model.u_bar+offset])
        #                 ax.legend()
        #             plt.suptitle(f'Trajectory {k + 1}')
        #             plt.tight_layout()
        #             plt.savefig(os.path.join(input_dir, f'input_{k + 1}.png'))
        #             plt.close(fig)

        #             # 3D position trajectory with body-frame axes every 10 
        #             # steps
        #             fig = plt.figure()
        #             ax = fig.add_subplot(111, projection='3d')
        #             sc = ax.scatter(
        #                 x_traj[k][:, 0], 
        #                 x_traj[k][:, 1], 
        #                 x_traj[k][:, 2], 
        #                 c=colors, 
        #                 cmap='coolwarm', 
        #                 s=10
        #             )
                    
        #             # Overlay body-frame arrows to visualise orientation along 
        #             # the path
        #             for i in range(0, len(x_traj[k]), 10):
        #                 roll, pitch, yaw = x_traj[k][i,3:6]
        #                 rotation_matrix = Rot.from_euler(
        #                     'xyz', [roll, pitch, yaw]).as_matrix()
        #                 x_arrow = rotation_matrix[:, 0] * model.min_width
        #                 y_arrow = rotation_matrix[:, 1] * model.min_length
        #                 z_arrow = rotation_matrix[:, 2] * model.min_height
        #                 ax.quiver(
        #                     x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
        #                     x_arrow[0], x_arrow[1], x_arrow[2], color='b', 
        #                     label='X-axis' if i == 0 else ""
        #                 )
        #                 ax.quiver(
        #                     x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
        #                     y_arrow[0], y_arrow[1], y_arrow[2], color='r', 
        #                     label='Y-axis' if i == 0 else ""
        #                 )
        #                 ax.quiver(
        #                     x_traj[k][i,0], x_traj[k][i,1], x_traj[k][i,2],
        #                     z_arrow[0], z_arrow[1], z_arrow[2], color='g', 
        #                     label='Z-axis' if i == 0 else ""
        #                 )
                                
        #             ax.set_xlabel('X [m]')
        #             ax.set_ylabel('Y [m]')
        #             ax.set_zlabel('Z [m]')
        #             ax.set_xlim(traj_xlim_min[0], traj_xlim_max[0])
        #             ax.set_ylim(traj_xlim_min[1], traj_xlim_max[1])
        #             ax.set_zlim(traj_xlim_min[2], traj_xlim_max[2])
        #             ax.set_title(f'3D Position Trajectory {k + 1}')
        #             set_axes_equal(ax)
        #             # plt.colorbar(sc, ax=ax, label='Time progression')
        #             plt.tight_layout()
        #             plt.savefig(
        #                 os.path.join(threeD_dir, f'3D_traj_{k + 1}.png')
        #             )
        #             plt.close(fig)


    #in 2D
        # --- Plot generated trajectories ---
        if params.plot:

            # Labels and titles for pose/velocity subplots
            pose_label = ['x [m]', 'z [m]', 'theta [deg]']
            vel_label = ['v$_x$ [m/s]', 'v$_z$ [m/s]', '$\omega_y$ [deg/s]']

            # Create (or recreate) output subdirectories
            traj_dir = os.path.join(plots_dir, 'trajectories')
            pose_dir = os.path.join(plots_dir, 'poses')
            velocity_dir = os.path.join(plots_dir, 'velocities')
            input_dir = os.path.join(plots_dir, 'inputs')
            planar_dir = os.path.join(plots_dir, 'planar_2D') # Sostituisce la cartella 3D
            
            plots_subdirs = [traj_dir, pose_dir, velocity_dir, input_dir, planar_dir]
            for subdir in plots_subdirs:
                ensure_clean_dir(subdir)

            # Normali per gli assi X e Z usate per l'ingombro del drone
            normals = [np.array([1, 0, 0]), np.array([0, 0, 1])]

            if params.check:
                sub_plot = 1
            else:
                sub_plot = max(1, params.prob_num // 10)

            import matplotlib.patches as patches

            for k in range(len(x_traj)):
                if k % sub_plot == 0:
                    horizon_ = x_traj[k].shape[0]
                    colors = np.linspace(0, 1, horizon_)
                    t = np.linspace(0, horizon_ * params.dt, horizon_)

                    # Estraiamo lo scaling e ricostruiamo le dimensioni del box
                    scale = x_data[k, 10]
                    # box = [X_max, Z_max, X_min_dist, Z_min_dist] (tutti pari a scale * 1.0)
                    box = np.array([scale, scale, scale, scale])
                    
                    # Limiti spaziali per i plot (Pos_X_min, Pos_Z_min, Theta_min)
                    traj_xlim_min = [-box[2], -box[3], -np.rad2deg(max_phi)]
                    traj_xlim_max = [ box[0],  box[1],  np.rad2deg(max_phi)]

                    # ---------------------------------------------------------
                    # 1. Phase-plane plot: Posizione vs Velocità (Sanity Check)
                    # ---------------------------------------------------------
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax = ax.reshape(-1)
                    for i in range(nq):
                        ax[i].grid(True, linewidth=0.5)
                        if i < model.npos:
                            ax[i].scatter(x_traj[k][:, i], x_traj[k][:, nq + i], c=colors, cmap='coolwarm', s=1)
                        else:
                            ax[i].scatter(np.rad2deg(x_traj[k][:, i]), np.rad2deg(x_traj[k][:, nq + i]), c=colors, cmap='coolwarm', s=1)
                        
                        ax[i].set_xlim([traj_xlim_min[i], traj_xlim_max[i]])
                        ax[i].set_xlabel(f'{pose_label[i]}')
                        ax[i].set_ylabel(f'{vel_label[i]}')
                    plt.suptitle(f'Phase-Plane Trajectory {k + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(traj_dir, f'traj_{k + 1}.png'))
                    plt.close(fig)

                    # ---------------------------------------------------------
                    # 2. Posizioni nel tempo con ingombro e limiti del BOX
                    # ---------------------------------------------------------
                    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
                    for i in range(nq):
                        ax[i].grid(True)
                        if i < model.npos:
                            line, = ax[i].plot(t, x_traj[k][:, i], label=f'{pose_label[i]}')
                            
                            # Calcolo dell'ingombro ruotato del drone
                            ellips_r = []
                            for h in range(len(t)):
                                # Lo stato in x_traj ha già esattamente 10 elementi (6 di cinematica + 4 di box)
                                full_x = x_traj[k][h, :]
                                ellips_r.append(np.sqrt(normals[i].T @ model.Q(full_x).full() @ normals[i]))
                            
                            ax[i].plot(t, x_traj[k][:, i] + ellips_r, color=line.get_color(), linestyle='--', linewidth=0.8)
                            ax[i].plot(t, x_traj[k][:, i] - ellips_r, color=line.get_color(), linestyle='--', linewidth=0.8)
                            
                            # Limiti orizzontali del BOX
                            ax[i].axhline(traj_xlim_max[i], color='r', linestyle=':', linewidth=1.5, label='Box Max')
                            ax[i].axhline(traj_xlim_min[i], color='r', linestyle=':', linewidth=1.5, label='Box Min')
                        else:
                            line, = ax[i].plot(t, np.rad2deg(x_traj[k][:, i]), label=f'{pose_label[i]}')
                            ax[i].axhline(np.rad2deg(max_phi), color='r', linestyle=':', linewidth=1.5, label='Max Tilt')
                            ax[i].axhline(-np.rad2deg(max_phi), color='r', linestyle=':', linewidth=1.5)
                        
                        ax[i].set_xlabel('Time [s]')
                        ax[i].set_ylabel(pose_label[i])
                        ax[i].legend(loc='upper right')
                        
                    plt.suptitle(f'Poses Trajectory {k + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(pose_dir, f'pose_{k + 1}.png'))
                    plt.close(fig)

                    # ---------------------------------------------------------
                    # 3. Velocità nel tempo
                    # ---------------------------------------------------------
                    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
                    for i in range(nq):
                        ax[i].grid(True)
                        if i < model.npos:
                            ax[i].plot(t, x_traj[k][:, nq + i], label=f'{vel_label[i]}')
                        else:
                            ax[i].plot(t, np.rad2deg(x_traj[k][:, nq + i]), label=f'{vel_label[i]}')
                        ax[i].set_xlabel('Time [s]')
                        ax[i].set_ylabel(vel_label[i])
                        ax[i].legend()
                    plt.suptitle(f'Velocities Trajectory {k + 1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(velocity_dir, f'vel_{k + 1}.png'))
                    plt.close(fig)

                    # ---------------------------------------------------------
                    # 4. Input di controllo (Motori)
                    # ---------------------------------------------------------
                    offset = 200
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for i in range(nu):
                        ax.grid(True)
                        ax.plot(t, u_traj[k][:, i], label=f'u_{i + 1}')
                    ax.set_title(f'Inputs Trajectory {k + 1}')
                    ax.axhline(model.u_bar, color='r', linestyle='--', lw=1, label='u_max')
                    ax.set_xlabel('Time [s]')
                    ax.set_ylabel('$u^2$ [(rad/s)$^2$]')
                    ax.set_ylim([0.0 - offset, model.u_bar + offset])
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(input_dir, f'input_{k + 1}.png'))
                    plt.close(fig)

                    # ---------------------------------------------------------
                    # 5. GRAFICO PLANARE 2D (Visualizzazione Fisica Reale)
                    # ---------------------------------------------------------
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Disegna la stanza (Box) ottimizzata in rosso tratteggiato
                    rect = patches.Rectangle((-box[2], -box[3]), box[0] + box[2], box[1] + box[3], 
                                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Viability Box')
                    ax.add_patch(rect)
                    
                    # Traccia il volo del drone nel piano X-Z
                    sc = ax.scatter(x_traj[k][:, 0], x_traj[k][:, 1], c=colors, cmap='coolwarm', s=10, label='Drone CM')
                    
                    # Disegna l'inclinazione del drone lungo il percorso
                    step = max(1, len(x_traj[k]) // 10)
                    for i in range(0, len(x_traj[k]), step):
                        x_pos, z_pos, theta = x_traj[k][i, 0], x_traj[k][i, 1], x_traj[k][i, 2]
                        
                        # Vettori orientamento basati sulla rotazione Pitch
                        dx_body = np.cos(theta) * (model.min_width/2)
                        dz_body = -np.sin(theta) * (model.min_width/2)
                        upx_body = np.sin(theta) * (model.min_height/2)
                        upz_body = np.cos(theta) * (model.min_height/2)
                        
                        # Freccia blu: asse orizzontale del drone / Freccia verde: direzione spinta motori
                        ax.quiver(x_pos, z_pos, dx_body, dz_body, angles='xy', scale_units='xy', scale=1, color='b', width=0.005)
                        ax.quiver(x_pos, z_pos, upx_body, upz_body, angles='xy', scale_units='xy', scale=1, color='g', width=0.005)

                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Z [m]')
                    # Margine per far respirare il grafico
                    margin = max(1.0, np.max(box) * 0.2)
                    ax.set_xlim(-box[2] - margin, box[0] + margin)
                    ax.set_ylim(-box[3] - margin, box[1] + margin)
                    ax.set_aspect('equal', adjustable='box')
                    ax.grid(True)
                    ax.set_title(f'Real World 2D Trajectory {k + 1}')
                    ax.legend(loc='upper right')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(planar_dir, f'planar_traj_{k + 1}.png'))
                    plt.close(fig)

    
    # =========================================================================
    # NEURAL NETWORK TRAINING
    # =========================================================================
    if params.training: 

        # --- Load data ---
        x_data = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc.npy')
        b_data = np.load(f'{params.DATA_DIR}{robotic_system}_b_vboc.npy')
        b_all_data = np.load(params.DATA_DIR + 'sth_b_all_vboc.npy')
        d_data = np.load(params.DATA_DIR + 'sth_d_vboc.npy')
        status_data = np.load(params.DATA_DIR + 'sth_status_vboc.npy')
        
        # --- Histograms of raw data distributions ---
        # if params.plot:
        #     hist_dir = os.path.join(plots_dir, 'histograms')
        #     ensure_clean_dir(hist_dir)
        #     b_all_data = np.load(
        #         f'{params.DATA_DIR}{robotic_system}_b_all_vboc.npy'
        #     )
        #     d_data = np.load(f'{params.DATA_DIR}{robotic_system}_d_vboc.npy')
        #     status_data = np.load(
        #         f'{params.DATA_DIR}{robotic_system}_status_vboc.npy'
        #     )
        #     plot_histogram(
        #         x_data[:,:6], 
        #         title="x[0:6]", 
        #         xlabel="Value",
        #         ylabel="Frequency", 
        #         bins=50, 
        #         saving_dir=hist_dir
        #     )
        #     plot_histogram(
        #         x_data[:,6:], 
        #         title="x[6:12]", 
        #         xlabel="Value", 
        #         ylabel="Frequency", 
        #         bins=50, 
        #         saving_dir=hist_dir
        #     )
        #     plot_histogram(
        #         b_data, 
        #         title="b", 
        #         xlabel="Value", 
        #         ylabel="Frequency", 
        #         bins=50, 
        #         saving_dir=hist_dir)
        #     plot_histogram(
        #         b_all_data, 
        #         title="b_all", 
        #         xlabel="Value", 
        #         ylabel="Frequency", 
        #         bins=50, 
        #         saving_dir=hist_dir
        #     )
        #     plot_histogram(
        #         -d_data, 
        #         title="d", 
        #         xlabel="Value", 
        #         ylabel="Frequency", 
        #         bins=50, 
        #         saving_dir=hist_dir
        #     )
        #     plot_histogram(
        #         status_data, 
        #         title="status", 
        #         xlabel="Value", 
        #         ylabel="Frequency", 
        #         bins=10, 
        #         saving_dir=hist_dir
        #     )    

        # --- Histograms of raw data distributions ---
        if params.plot:
            hist_dir = os.path.join(plots_dir, 'histograms')
            ensure_clean_dir(hist_dir)
            
            # 1. Istogramma degli Input: theta, vx, vz, wy (Indici 2, 3, 4, 5 di x_data)
            plot_histogram(
                #in 2D
                x_data[:, 2:6],
                title="Inputs_Angles_and_Velocities", 
                xlabel="Value",
                ylabel="Frequency", 
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
                bins=10, 
                saving_dir=hist_dir,
                xticks=[0, 2, 4]
            )
            # 4. Istogramma degli Orizzonti di Convergenza (N)
            # Carichiamo il file appena salvato
            n_data = np.load(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc.npy')
        
            plot_histogram(
                n_data, 
                title="Distribution_of_Converged_Horizons_N", 
                xlabel="Horizon Length (N steps)", 
                ylabel="Frequency", 
                bins=15, # 10 colonne vanno bene per step discreti (20, 25, 30...)
                saving_dir=hist_dir
            )
            # 5. Istogramma dei casi FALLITI
            failed_file = f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc.npy'
            if os.path.exists(failed_file):
                failed_data = np.load(failed_file)
                if len(failed_data) > 0:
                    # In 2D estraiamo theta, vx, vz, wy (indici da 2 a 5 compresi)
                    plot_histogram(
                        failed_data[:, 2:6],
                        title="Failed_Cases_Initial_Conditions", 
                        xlabel="Value",
                        ylabel="Frequency", 
                        bins=20, 
                        saving_dir=hist_dir
                    )

        # Drop position columns and prepend box dimensions as input features
        #x_data = np.hstack((b_data, x_data[:, model.npos:]))
        #in 2D
        dataset = np.hstack(( x_data[:, 2:6], b_data))
        np.random.shuffle(dataset)

        # Dividiamo in Input (x_data) e Target (y_data)
        #in 2D
        x_data = dataset[:, :4]
        y_data = dataset[:, 4:]

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
        # mean = np.mean(x_train[:, :nbori])
        # std = np.std(x_train[:, :nbori])
        # for x_input in [x_train, x_val, x_test]:
        #     x_input[:, :nbori] = (x_input[:, :nbori] - mean) / std
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        for x_input in [x_train, x_val, x_test]:
            x_input[:] = (x_input - mean) / std

        # Normalize velocity components by their L2 norm (output = the norm 
        # itself)
        # y_data = np.linalg.norm(x_data[:, nbori:], axis=1).reshape(n, 1)
        # for k in range(n):
        #     if y_data[k] != 0.: 
        #         x_data[k, nbori:] /= y_data[k] 

        # --- Split outputs ---
        y_train = y_data[:train_size]
        y_val = y_data[train_size:train_size + val_size]
        y_test = y_data[train_size + val_size:]
        
        # --- Build model, loss, and optimiser ---
        #nx_train = nbori+model.nv
        #in 2D
        nx_train = 4
    

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

        # --- Train ---
        print('***START TRAINING***\n')
        train_val_dir = os.path.join(plots_dir, 'training_validation')
        ensure_clean_dir(train_val_dir)

        train_evol, val_evol = regressor.training(
            x_train, 
            y_train, 
            x_val, 
            y_val, 
            args['epochs']
        )
        print('***TRAINING COMPLETED***\n')

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
        loss_dir = os.path.join(plots_dir, 'loss_evolution')
        ensure_clean_dir(loss_dir)
        fig = plt.figure()
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
    # VIABILITY KERNEL PLOTTING 2D
    # =========================================================================
    if params.plot and not params.generation: 
        
        # --- Load trained network ---
        device = torch.device("cpu")
        nx_train = 4
        
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
        x_data_plot = x_data_raw[:, 2:6]

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
        # 2. PLOT DELL'ANALISI DI SENSIBILITA' (Velocità vs Dimensione Stanza)
    # =========================================================================
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Nel 2D, gli indici in x_data_raw sono: vx=3, vz=4, scale=6
        v_x = x_data_raw[:, 3]
        v_z = x_data_raw[:, 4]
        scale = x_data_raw[:, 10]
        
        # v_x vs Scaling
        ax[0].scatter(v_x, scale, alpha=0.5, s=5, c='blue')
        ax[0].set_xlabel('v_x [m/s]', fontsize=12)
        ax[0].set_ylabel('Optimized Scale [m]', fontsize=12)
        ax[0].set_title('Dipendenza da v_x')
        ax[0].grid(True)

        # v_z vs Scaling
        ax[1].scatter(v_z, scale, alpha=0.5, s=5, c='red')
        ax[1].set_xlabel('v_z [m/s]', fontsize=12)
        ax[1].set_title('Dipendenza da v_z')
        ax[1].grid(True)

        plt.suptitle('Dimensioni della stanza in funzione delle velocità iniziali (2D)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(brs_dir, 'Velocities_vs_Scale_2D.png'))
        plt.close(fig)