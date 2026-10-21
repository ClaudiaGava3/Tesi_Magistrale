
# ==========================================================================================================================================================
"""Generate an OCP-based ground-truth dataset for safety validation.

The standard VBOC optimal control problem is solved for sampled drone states
and box dimensions. Its feasibility outcomes and optimized box scales provide
ground truth for comparing safety validation with the VBOC method and
DeepReach.
"""
# ==========================================================================================================================================================


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
from RBOC.abstract_dataset_test_deepreach import Model
from RBOC.controller_dataset_test_deepreach import ViabilityController
from src.RBOC.learning import NeuralNetwork, NovelNeuralNetwork, RegressionNN, plot_brs
from src.RBOC.parser import Parameters, parse_args

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





def compute_data_on_border(
    q_init: np.ndarray, # Initial state: positions, orientation, and velocities.
    ref_box: np.ndarray,
    box_guess: float,
    N_guess: int,
    N_increment: int,
    vboc_repeat: int,

) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute a single data point on the border of the viability kernel.

    Solves the VBOC OCP from an initial drone state and box-side parameters,
    extending the prediction horizon iteratively if needed.

    Parameters
    ----------
    q_init : np.ndarray
        Initial drone state ``[x, z, theta, vx, vz, wy]`` of shape (6,).
    ref_box : np.ndarray
        Parameters for the four box sides in the order
        ``[x_plus, z_plus, x_minus, z_minus]``.
    box_guess : float
        Initial guess for the common scale applied to the box-side parameters.
    N_guess : int
        Initial prediction horizon length.
    N_increment : int
        Number of time steps added per VBOC iteration.
    vboc_repeat : int
        Maximum number of VBOC solve repetitions before declaring failure.
    Returns
    -------
    x0 : np.ndarray or None
        First state of the optimal trajectory; None if infeasible.
    x_star : np.ndarray or None
        Full optimal state trajectory of shape (N, nx); None if infeasible.
    u_star : np.ndarray or None
        Optimal control sequence of shape (N, nu); None if infeasible.
    N_final : int or None
        Final prediction horizon; None if the solve fails.
    status : int
        Solver status returned by the final solve attempt (0 or 2 is accepted).
    """
    global progress_var
    
    controller.resetHorizon(N_guess)


    # --- Initial state and hover-control guesses for the planar OCP ---
    x_guess = np.zeros((N_guess, model.nx))
    # Keep position and box variables constant across the initial guess.
    x_guess[:, :2] = q_init[:2]  
    # Initialize the four box-side parameters sampled for this case.
    x_guess[:, 6:10] = ref_box
    x_guess[:, 10] = box_guess 

    # But at the very first time instant (node 0), we tell Acados exactly
    # how the drone starts in reality (including the true velocities and angles)
    x_guess[0, :6] = q_init


    # Legacy flat-state vector; it is not passed to the controller.
    x_flat = np.zeros(11)
    x_flat[:2] = q_init[:2]  # Copy only the positions (X, Z)
    x_flat[10] = box_guess    # The scale is state element 10.

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

def ensure_clean_dir(path: str) -> None:
    """
    Ensure that a directory exists and is empty.

    If the directory exists, regular files directly inside it are deleted.
    Subdirectories and their contents are left unchanged.
    If it does not exist, it is created (including any missing parent 
    directories).

    Parameters
    ----------
    path : str
        Path to the directory to clean or create.
    """
    if os.path.exists(path):
        # Remove regular files directly inside the directory.
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(path)



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

        # In check mode, use one fixed drone initial state; box dimensions
        # are still sampled below.
        if params.check:
            params.prob_num = 1
        
        # --- Initial position: origin for all problems ---
        pos_init = np.zeros((params.prob_num, model.npos))

        # --- Initial orientation: sampled within the allowed inclination range ---
        if(params.orient_g_rej):
            max_phi = np.pi/2
        else:
            max_phi = np.pi/2
        
        if params.check:
            orient_init = np.zeros((params.prob_num, model.nori))
            vel_init = np.zeros((params.prob_num, model.nv))
        else:
            # Generate a random pitch in the planar model.
            orient_init = np.random.uniform(-max_phi, max_phi, params.prob_num).reshape(-1, 1)

            # Generate random velocities (vx, vz, wy).
            vel_init = np.random.uniform(-1.0, 1.0, (params.prob_num, model.nv))

        # Create the 6-element drone state: [x, z, theta, vx, vz, wy].
        q_init = np.hstack([pos_init, orient_init, vel_init])
        b_init = np.random.uniform(0.15, 4.0, (params.prob_num, model.nbox))

        # --- Initial guess for the common box scale ---
        box_guess=1.0


        # Generate legacy random seeds; they are not passed to the worker calls.
        randomSeeds = [random.randint(0, params.prob_num) 
                       for _ in range(params.prob_num)
        ]

        # --- Accumulators for results across all batches ---
        all_x_0, all_x_t, all_u_t, all_n_final, all_status = [], [], [], [], []
        all_failed_q_init = [] # <--- NEW LIST FOR FAILURES
        all_test_dataset = []

        # Split the problems into sub-batches to allow intermediate saves
        if params.check:
            sub_batch = 1
        else:
            sub_batch = 100
        n_batch = int(params.prob_num/sub_batch)


        # Nominal equal-side parameters; worker calls use sampled b_init instead.
        ref_box = np.array([1.0, 1.0, 1.0, 1.0])


        print('Start data generation')
        for nb in range(n_batch):  
            with Pool(params.cpu_num) as p:
                res = p.starmap(
                    compute_data_on_border, 
                    [(q0, b0, box_guess, N, N_increment, vboc_repeat) 
                     for q0, b0 in zip(q_init[(nb*sub_batch):((nb+1)*sub_batch)],
                                       b_init[(nb*sub_batch):((nb+1)*sub_batch)])]
                )

            # --- Unpack parallel results ---
            x_0, x_t, u_t, n_final_list, status = zip(*res)
            all_x_0.extend(x_0)
            all_x_t.extend(x_t)
            all_u_t.extend(u_t)
            all_n_final.extend(n_final_list)
            all_status.extend(status)

            # --- Build one classification-dataset row per attempted solve ---
            q0_batch = q_init[(nb*sub_batch):((nb+1)*sub_batch)]
            b0_batch = b_init[(nb*sub_batch):((nb+1)*sub_batch)]
            
            for i in range(len(x_0)):
                # Read the optimized scale at state index 10; use -1000.0 on failure.
                scale_val = x_0[i][10] if x_0[i] is not None else -1000.0

                # Mark success only when the solver status is accepted and a solution exists.
                is_success = 1.0 if (status[i] in [0, 2] and x_0[i] is not None) else 0.0
                
                # Store [theta, vx, vz, wy], sampled box-side parameters, status,
                # optimized scale, and binary success label.
                row = np.hstack([q0_batch[i][2:6], b0_batch[i], status[i], scale_val, is_success])
                all_test_dataset.append(row)
            # ------------------------------------------------

            # Collect the initial drone states for failed solves.
            q0_batch = q_init[(nb*sub_batch):((nb+1)*sub_batch)]
            for i in range(len(x_0)):
                if x_0[i] is None:
                    # If it failed, save its initial condition
                    all_failed_q_init.append(q0_batch[i])

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
            
            # The optimized common scale multiplies the four sampled side parameters.
            b_optimized = x_data[:, 10].reshape(-1, 1)

            actual_boxes = x_data[:, 6:10] * b_optimized

            traj_kinematics = np.vstack([traj[:, 2:6] for traj in x_traj])

            np.save(f'{params.DATA_DIR}{robotic_system}_x_vboc_dataset4m', x_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_b_vboc_dataset4m', b_optimized)
            np.save(f'{params.DATA_DIR}{robotic_system}_n_horizons_vboc_dataset4m', n_data)
            np.save(f'{params.DATA_DIR}{robotic_system}_status_vboc_dataset4m', status_list)
            # === SAVE FAILURES ===
            np.save(f'{params.DATA_DIR}{robotic_system}_failed_q_init_vboc_dataset4m', np.array(all_failed_q_init))

            np.save(f'{params.DATA_DIR}{robotic_system}_actual_boxes_vboc_dataset4m', actual_boxes)
            np.save(f'{params.DATA_DIR}{robotic_system}_traj_kinematics_vboc_dataset4m', traj_kinematics)
            np.save(f'{params.DATA_DIR}{robotic_system}_u_traj_vboc_dataset4m', np.vstack(u_traj))

            np.save(f'{params.DATA_DIR}{robotic_system}_TEST_dataset_classification4m', np.array(all_test_dataset))
            
            solved = len(x_data)
            print(f'Batch {nb}: Total number of points saved until now: {solved}')

        print('Total number of points solved: %d' % solved)


