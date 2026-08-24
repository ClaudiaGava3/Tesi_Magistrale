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
from src.VBOC.learning_copy import NeuralNetwork, NovelNeuralNetwork, RegressionNN, plot_brs
from src.VBOC.parser import Parameters, parse_args

# Global font size settings
plt.rcParams.update({
    'axes.titlesize': 28,     # Title size
    'axes.labelsize': 24,     # Axis label size (X and Y)
    'xtick.labelsize': 12,    # X-axis tick label size
    'ytick.labelsize': 12,    # Y-axis tick label size
    'legend.fontsize': 22,    # Legend font size
    'font.size': 22           # Base font size for everything else
}) 

install()

progress_var = Value('i', 0)
np.set_printoptions(linewidth=np.inf)


def plot_histogram(
   data: np.ndarray,
    title: str = "Histogram",
    xlabel: str | list = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
    saving_dir: str = "plots/histograms/",
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
    q_init: np.ndarray, # MUST CONSIST OF POSITIONS AND VELOCITIES
    ref_box: np.ndarray,
    box_guess: float,
    N_guess: int,
    N_increment: int,
    vboc_repeat: int,

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


    # --- Initial guess: stationary at q_init with gravity compensation ---
    #in 2D
    x_guess = np.zeros((N_guess, model.nx))
    # Keep the position flat and the scaling constant for the entire guess
    x_guess[:, :2] = q_init[:2]  
    # Initial guess for the 4 sides (we imagine they start as 1.0 proportional squares)
    x_guess[:, 6:10] = ref_box
    x_guess[:, 10] = box_guess 

    # But at the very first time instant (node 0), we tell Acados exactly
    # how the drone starts in reality (including the true velocities and angles)
    x_guess[0, :6] = q_init


    # 1. Define a flat initial-state guess
    x_flat = np.zeros(11)
    x_flat[:2] = q_init[:2]  # Copy only the positions (X, Z)
    x_flat[10] = box_guess    # Copy the scaling (Index 6 in 2D)

    # 2. Build the allocation matrix for 2D: a 6x2 matrix.
    allocation_matrix = np.vstack((model.F, model.M))

    # 3. Physical objective: [Fx, Fy, Fz, Mx, My, Mz]
    # Balance gravity on Z and cancel torques
    wrench_hover = np.array([0.0, 0.0, model.mass * model.g, 0.0, 0.0, 0.0])
    
    # 4. Balanced thrust for the 2 motors
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
    nn_filename = f'{params.NN_DIR}{robotic_system}_{params.act}_randB.pt'
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
            max_phi = np.pi/2
        else:
            max_phi = np.pi/2
        
        if params.check:
            orient_init = np.zeros((params.prob_num, model.nori))
            vel_init = np.zeros((params.prob_num, model.nv))
        else:
            # Generate a random pitch
            # in 2D
            orient_init = np.random.uniform(-max_phi, max_phi, params.prob_num).reshape(-1, 1)

            # Generate random VELOCITIES (vx, vz, wy)
            vel_init = np.random.uniform(-1.0, 1.0, (params.prob_num, model.nv))

        # Create the initial state vector of 6 elements: [x, z, theta, vx, vz, wy]
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
        all_failed_q_init = [] # <--- NEW LIST FOR FAILURES
        all_test_dataset = []

        # Split the problems into sub-batches to allow intermediate saves
        if params.check:
            sub_batch = 1
        else:
            sub_batch = 100
        n_batch = int(params.prob_num/sub_batch)


        # in 2D
        ref_box = np.array([1.0, 1.0, 1.0, 1.0]) # Reference box 1:1


        print('Start data generation')
        for nb in range(n_batch):  
            with Pool(params.cpu_num) as p:
                res = p.starmap(
                    compute_data_on_border, 
                    [(q0, b0, box_guess, N, N_increment, vboc_repeat) 
                     for q0, b0 in zip(q_init[(nb*sub_batch):((nb+1)*sub_batch)],
                                       b_init[(nb*sub_batch):((nb+1)*sub_batch)])]
                )

            # --- Unpack parallel results (Now 5 items, no longer 7!) ---
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
                row = np.hstack([q0_batch[i][2:6], b0_batch[i], status[i], scale_val])
                all_test_dataset.append(row)
            # ------------------------------------------------

            # === NEW CODE: CAPTURE FAILED CASES ===
            q0_batch = q_init[(nb*sub_batch):((nb+1)*sub_batch)]
            for i in range(len(x_0)):
                if x_0[i] is None:
                    # If it failed, save its initial condition
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
            n_data = np.array([all_n_final[i] for i in range(len(all_n_final)) if all_x_0[i] is not None])  # Filter N_final using the same exact logic as x_data to keep alignment
            status_list = list(all_status)
            
            # The optimized "box" is just a scaling factor!
            b_optimized = x_data[:, 10].reshape(-1, 1)

            actual_boxes = x_data[:, 6:10] * b_optimized

            traj_kinematics = np.vstack([traj[:, 2:6] for traj in x_traj])

            np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc_randB', x_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_b_vboc_randB', b_optimized)
            np.save(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc_randB', n_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_status_vboc_randB', status_list)
            # === SAVE FAILURES ===
            np.save(f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc_randB', np.array(all_failed_q_init))

            np.save(f'{params.DATA_DIR}{robotic_system}_actual_boxes_vboc_randB', actual_boxes)
            np.save(f'{params.DATA_DIR}{robotic_system}_traj_kinematics_vboc_randB', traj_kinematics)
            np.save(f'{params.DATA_DIR}{robotic_system}_u_traj_vboc_randB', np.vstack(u_traj))

            np.save(f'{params.DATA_DIR}{robotic_system}_TEST_dataset_classification', np.array(all_test_dataset))
            
            solved = len(x_data)
            print(f'Batch {nb}: Total number of points saved until now: {solved}')

        print('Total number of points solved: %d' % solved)


 # =========================================================================
    # PLOT
# =========================================================================
    #in 2D
        # --- Plot generated trajectories ---
        if params.plot:

            # Labels and titles for pose/velocity subplots
            pose_label = ['x [m]', 'z [m]', 'theta [deg]']
            vel_label = ['v$_x$ [m/s]', 'v$_z$ [m/s]', '$\omega_y$ [deg/s]']

            # Create (or recreate) output subdirectories
            traj_dir = os.path.join(plots_dir, 'trajectories500')
            pose_dir = os.path.join(plots_dir, 'poses500')
            velocity_dir = os.path.join(plots_dir, 'velocities500')
            input_dir = os.path.join(plots_dir, 'inputs500')
            planar_dir = os.path.join(plots_dir, 'planar_2D500') # Sostituisce la cartella 3D
            
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

                    # Extract scaling and free sides to reconstruct the asymmetric box
                    scale = x_data[k, 10]
                    l_x_plus = x_data[k, 6]
                    l_z_plus = x_data[k, 7]
                    l_x_minus = x_data[k, 8]
                    l_z_minus = x_data[k, 9]

                    # box = [X_max, Z_max, X_min_dist, Z_min_dist]
                    box = np.array([
                        scale * l_x_plus,
                        scale * l_z_plus,
                        scale * l_x_minus,
                        scale * l_z_minus
                    ])

                    # Spatial limits for the plots (Pos_X_min, Pos_Z_min, Theta_min)
                    traj_xlim_min = [-box[2], -box[3], -np.rad2deg(max_phi)]
                    traj_xlim_max = [ box[0],  box[1],  np.rad2deg(max_phi)]

                    # ---------------------------------------------------------
                    # 1. Phase-plane plot: Position vs Velocity (Sanity Check)
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
                    # 2. Positions over time with occupancy and BOX limits
                    # ---------------------------------------------------------
                    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
                    for i in range(nq):
                        ax[i].grid(True)
                        if i < model.npos:
                            line, = ax[i].plot(t, x_traj[k][:, i], label=f'{pose_label[i]}')
                            
                            # Compute the rotated occupancy of the drone
                            ellips_r = []
                            for h in range(len(t)):
                                # The state in x_traj already has exactly 10 elements (6 kinematics + 4 box)
                                full_x = x_traj[k][h, :]
                                ellips_r.append(np.sqrt(normals[i].T @ model.Q(full_x).full() @ normals[i]))
                            
                            ax[i].plot(t, x_traj[k][:, i] + ellips_r, color=line.get_color(), linestyle='--', linewidth=0.8)
                            ax[i].plot(t, x_traj[k][:, i] - ellips_r, color=line.get_color(), linestyle='--', linewidth=0.8)
                            
                            # Horizontal BOX limits
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
                    # 3. Velocities over time
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
                    # 4. Control inputs (Motors)
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
                    # 5. 2D PLANAR PLOT (Real-World Visualization)
                    # ---------------------------------------------------------
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    # Draw the optimized room (Box) in dashed red
                    rect = patches.Rectangle((-box[2], -box[3]), box[0] + box[2], box[1] + box[3], 
                                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Viability Box')
                    ax.add_patch(rect)
                    
                    # Trace the drone flight in the X-Z plane
                    sc = ax.scatter(x_traj[k][:, 0], x_traj[k][:, 1], c=colors, cmap='coolwarm', s=10, label='Drone CM')
                    
                    # Draw the drone tilt along the path
                    step = max(1, len(x_traj[k]) // 10)
                    for i in range(0, len(x_traj[k]), step):
                        x_pos, z_pos, theta = x_traj[k][i, 0], x_traj[k][i, 1], x_traj[k][i, 2]
                        
                        # Orientation vectors based on Pitch rotation
                        dx_body = np.cos(theta) * (model.min_width/2)
                        dz_body = -np.sin(theta) * (model.min_width/2)
                        upx_body = np.sin(theta) * (model.min_height/2)
                        upz_body = np.cos(theta) * (model.min_height/2)
                        
                        # Blue arrow: drone horizontal axis / Green arrow: motor thrust direction
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
        x_data = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc_randB2000.npy')
        b_data = np.load(f'{params.DATA_DIR}{robotic_system}_b_vboc_randB2000.npy')
        b_all_data = np.load(params.DATA_DIR + 'sth_b_all_vboc.npy')
        d_data = np.load(params.DATA_DIR + 'sth_d_vboc.npy')
        status_data = np.load(params.DATA_DIR + 'sth_status_vboc_randB2000.npy')

        actual_boxes = np.load(f'{params.DATA_DIR}{robotic_system}_actual_boxes_randB2000.npy')
        traj_kinematics = np.load(f'{params.DATA_DIR}{robotic_system}_traj_kinematics_randB2000.npy')
        u_traj = np.load(f'{params.DATA_DIR}{robotic_system}_u_traj_randB2000.npy')
        n_data = np.load(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc_randB2000.npy')
        
          

        # --- Histograms of raw data distributions ---
        if params.plot:
            hist_dir = os.path.join(plots_dir, 'histograms_rand2000')
            ensure_clean_dir(hist_dir)
            
            # 1. Histogram of Inputs: theta, vx, vz, wy (Indices 2, 3, 4, 5 of x_data)
            plot_histogram(
                #in 2D
                x_data[:, 2:6],
                title="Initial_Conditions", 
                xlabel="Value",
                ylabel="Frequency",
                subplot_titles=["$\\theta_0$", "$v_x0$", "$v_z0$", "$\omega_y0$"],
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
            
            
            # 5. Dimensioni del Box Normalizzato (Indici 6, 7, 8, 9 di x_data)
            plot_histogram(
                x_data[:, 6:10],
                title="5_Normalized_Box_Dimensions", 
                xlabel="Normalized Value",
                ylabel="Frequency",
                subplot_titles=["$X_{max}$ norm", "$Z_{max}$ norm", "$X_{min}$ norm", "$Z_{min}$ norm"],
                bins=50, 
                saving_dir=hist_dir
            )
        
            plot_histogram(
                n_data, 
                title="Distribution_of_Converged_Horizons_N", 
                xlabel="Horizon Length (N steps)", 
                ylabel="Frequency", 
                bins=np.arange(19, 34, 2),
                saving_dir=hist_dir,
                xticks=np.arange(20, 33, 2)
            )

            # 6. Dimensioni del Box Effettivo (Lati calcolati in metri)
            plot_histogram(
                actual_boxes,
                title="Actual_Box_Dimensions", 
                xlabel="Length [m]",
                ylabel="Frequency",
                subplot_titles=["$X_{max}$ [m]", "$Z_{max}$ [m]", "$X_{min}$ [m]", "$Z_{min}$ [m]"],
                bins=50, 
                saving_dir=hist_dir
            )

            # 5. Histogram of FAILED cases
            failed_file = f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc_randB2000.npy'
            if os.path.exists(failed_file):
                failed_data = np.load(failed_file)
                if len(failed_data) > 0:
                    # In 2D extract theta, vx, vz, wy (indices 2 through 5 inclusive)
                    plot_histogram(
                        failed_data[:, 2:6],
                        title="Failed_Cases_Initial_Conditions", 
                        xlabel="Value",
                        ylabel="Frequency", 
                        bins=20, 
                        saving_dir=hist_dir
                    )

            # 7. Traiettorie: Angolo, vx, vz e Motori
            # traj_kinematics[:, 0:3] sono theta, vx, vz.
            # Aggiungiamo u_traj[:, 0] (il motore 1) per avere esattamente 4 grafici come hai chiesto.
            traj_data_to_plot = np.hstack((
                traj_kinematics[:, 0:4],          # Angolo e 3 velocità
                u_traj                           # Comando motore (u_1,u_2)
            ))

            plot_histogram(
                traj_data_to_plot,
                title="Trajectory_Kinematics_and_Motors", 
                xlabel="Value",
                ylabel="Frequency",
                subplot_titles=["Traj $\\theta$", "Traj $v_x$", "Traj $v_z$", "Traj $\omega_y$", "Motor $u_1$", "Motor $u_2$"], 
                bins=50, 
                saving_dir=hist_dir
            )

        # Drop position columns and prepend box dimensions as input features
        #in 2D
        dataset = np.hstack(( x_data[:, 2:6],x_data[:, 6:10], b_data))
        np.random.shuffle(dataset)

        # Split into Input (x_data) and Target (y_data)
        # in 2D
        x_data = dataset[:, :8]
        y_data = dataset[:, 8:]

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


        # --- Split outputs ---
        y_train = y_data[:train_size]
        y_val = y_data[train_size:train_size + val_size]
        y_test = y_data[train_size + val_size:]
        
        # --- Build model, loss, and optimiser ---
        #nx_train = nbori+model.nv
        #in 2D
        nx_train = 8
    

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

        # Reload the simulated data for the scatter plot
        x_data_raw = np.load(f'{params.DATA_DIR}{robotic_system}_x_vboc.npy')
        # Extract only the 4 inputs [theta, vx, vz, wy] for the BRS plot
        x_data_plot = x_data_raw[:, 2:6]

        # Clean call
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
        # 2. SENSITIVITY ANALYSIS PLOT (Speed vs Room Size)
    # =========================================================================
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # In 2D, the indices in x_data_raw are: vx=3, vz=4, scale=6
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