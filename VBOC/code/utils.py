import os
import yaml
import numpy as np
import torch    
import argparse
import matplotlib.pyplot as plt
from urdf_parser_py.urdf import URDF


def dof_value(value):
    try:
        dof = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")
    if not 1 <= dof <= 6:
        raise argparse.ArgumentTypeError(f"{dof} is not a valid number of degrees of freedom. Must be between 1 and 6")
    return dof

def horizon_value(value):
    try:
        horizon = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")
    if not horizon > 0:
        raise argparse.ArgumentTypeError(f"{horizon} is not a valid horizon. Must be greater than 0")
    return horizon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--system', type=str, default='z1',
                        help='Systems to test. Available: pendulum, double_pendulum, ur5, z1')
    parser.add_argument('-d', '--dofs', type=dof_value, default=4,
                        help='Number of desired degrees of freedom of the system')
    parser.add_argument('-b', '--build', action='store_true',
                        help='Build the code of the embedded controller')
    parser.add_argument('--horizon', type=horizon_value, default=45,
                        help='Horizon of the optimal control problem')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the approximated viability kernel')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of epochs for training the neural network')
    parser.add_argument('-a', '--activation', type=str, default='gelu',
                        help='Activation function for the neural network')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Alpha parameter for the NN safety factor')
    return vars(parser.parse_args())


class Parameters:
    def __init__(self, urdf_name):
        self.urdf_name = urdf_name
        # Define all the useful paths
        self.PKG_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.join(self.PKG_DIR, '../..')
        self.CONF_DIR = os.path.join(self.ROOT_DIR, 'config/')
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data/')
        self.GEN_DIR = os.path.join(self.ROOT_DIR, 'generated/')
        self.NN_DIR = os.path.join(self.ROOT_DIR, 'nn_models/' + urdf_name + '/')
        self.ROBOTS_DIR = os.path.join(self.ROOT_DIR, 'robots/')
        # Create directories if they do not exist
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
        if not os.path.exists(self.NN_DIR):
            os.makedirs(self.NN_DIR)
        self.robot_urdf = f'{self.ROBOTS_DIR}/{urdf_name}_description/urdf/{urdf_name}.urdf'

        parameters = yaml.load(open(self.ROOT_DIR + '/config.yaml'), Loader=yaml.FullLoader)

        self.prob_num = int(parameters['prob_num'])
        self.n_steps = int(parameters['n_steps'])
        self.cpu_num = int(parameters['cpu_num'])
        self.build = False
        
        self.N = int(parameters['N'])
        self.dt = float(parameters['dt'])
        self.alpha = int(parameters['alpha'])

        self.solver_type = 'SQP'
        self.solver_mode = parameters['solver_mode']
        self.nlp_max_iter = int(parameters['nlp_max_iter'])
        self.qp_max_iter = int(parameters['qp_max_iter'])
        self.alpha_reduction = float(parameters['alpha_reduction'])
        self.alpha_min = float(parameters['alpha_min'])
        self.levenberg_marquardt = float(parameters['levenberg_marquardt'])

        self.state_tol = float(parameters['state_tol'])
        self.cost_tol = float(parameters['cost_tol'])
        self.globalization = 'MERIT_BACKTRACKING'

        self.learning_rate = float(parameters['learning_rate'])
        self.weight_decay = float(parameters['weight_decay'])
        self.batch_size = int(parameters['batch_size'])
        self.beta = float(parameters['beta'])
        self.train_ratio = float(parameters['train_ratio'])
        self.val_ratio = float(parameters['val_ratio'])
        self.hidden_size = int(parameters['hidden_size'])
        self.hidden_layers = int(parameters['hidden_layers'])

        # For cartesian constraint
        self.obs_flag = bool(parameters['obs_flag'])
        self.frame_name = 'gripperMover'            # only for z1


class Sine(torch.nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sin(self.alpha * x)
    

args = parse_args()
model_name = args['system']
params = Parameters(model_name)
robot = URDF.from_xml_file(params.robot_urdf)
links = [robot.links[i].name for i in range(len(robot.links))]
joints = [robot.joints[i] for i in range(len(robot.joints))]


def align_vectors(a, b):
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):
        return -np.eye(3, dtype=np.float64)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

def create_moving_capsule(name: str,link_name:str,link_axis: int,radius: float,length: float,spatial_offset,rotation_offset=None,color=[1,0,0,0.3]):
    capsule=dict()
    capsule['type'] = 'moving'
    capsule['axis'] = link_axis
    capsule['radius'] = radius
    capsule['length'] = length
    capsule['name'] = name
    capsule['link_name'] = link_name
    # first point defined by offset from link origin, second length offset from the first one 
    # capsule['end_points'] = [np.hstack((spatial_offset,np.ones(1))), np.hstack((spatial_offset,np.ones(1)))]
    capsule['end_points'] = [np.hstack((np.zeros(3),np.ones(1))), np.hstack((np.zeros(3),np.ones(1)))]
    capsule['direction'] = np.sign( joints[links.index(capsule['link_name'])].origin.xyz[link_axis])
    capsule['end_points'][1][link_axis] += capsule['direction']*capsule['length']
    capsule['end_points_T_fun'] = [None]
    capsule['end_points_fk'] = [None,None]
    capsule['rotation_offset'] = rotation_offset
    capsule['spatial_offset'] = spatial_offset
    capsule['end_points_fk_fun'] = [None]
    capsule['color'] = color
    return capsule

def create_fixed_capsule(name,radius: float,fixed_A,fixed_B,color=[1,0,0,0.3]):
    capsule=dict()
    capsule['type'] = 'fixed'
    capsule['name'] = name
    capsule['end_points'] = [fixed_A,fixed_B]
    capsule['length'] = np.linalg.norm(fixed_A-fixed_B)
    capsule['radius'] = radius
    capsule['end_points_fk'] = capsule['end_points']
    capsule['end_points_T_fun'] = align_vectors(np.array([0,1,0]),capsule['end_points'][1]-capsule['end_points'][0])     
    capsule['color'] = color
    return capsule

def assign_pairs(obj1_name,obj2_name,obstacles_list,capsules_list):
    pair=dict()
    pair['elements'] = [None,None]
    pair['type'] = None
    for capsule in capsules_list:
        if obj1_name == capsule['name']:
            pair['elements'][0] = capsule
            break
    for capsule in capsules_list:
        if obj2_name == capsule['name']:
            pair['elements'][1] = capsule
            if pair['elements'][0] != None:
                pair['type'] = 0
            break 
    for obstacle in obstacles_list:
        if obj2_name == obstacle['name']:
            pair['elements'][1] = obstacle
            if obstacle['type'] == 'sphere': pair['type'] = 1
            elif obstacle['type'] == 'box': pair['type'] = 2
            break 
    return pair


# Define the obstacles (for z1)
ee_radius = 0.075
obs = dict()
obs['name'] = 'floor'
obs['type'] = 'box'
obs['dimensions'] = [1.5, 0.75, 1e-3]
obs['color'] = [0, 0, 1, 1]
obs['position'] = np.array([0.75, 0., 0.])
obs['transform'] = np.eye(4)
obs['bounds'] = np.array([ee_radius, 1e6])      # lb , ub
obstacles = [obs]

# obs = dict()
# obs['name'] = 'ball'
# obs['type'] = 'sphere'
# obs['radius'] = 0.12
# obs['color'] = [0, 1, 1, 1]
# obs['position'] = np.array([0.6, 0., 0.12])
# T_ball = np.eye(4)
# T_ball[:3, 3] = obs['position']
# obs['transform'] = T_ball
# obs['bounds'] = np.array([(ee_radius + obs['radius']) ** 2, 1e6])     
# obstacles.append(obs)

obs = dict()
obs['name'] = 'ball1'
obs['type'] = 'sphere'
obs['radius'] = 0.05
obs['color'] = [0, 1, 0, 1]
obs['position'] = np.array([0.6, 0.2, 0.12])
T_ball = np.eye(4)
T_ball[:3, 3] = obs['position']
obs['transform'] = T_ball
obs['bounds'] = np.array([(ee_radius + obs['radius']) ** 2, 1e6]) 
obstacles.append(obs)

obs = dict()
obs['name'] = 'ball2'
obs['type'] = 'sphere'
obs['radius'] = 0.05
obs['color'] = [0, 1, 0, 1]
obs['position'] = np.array([0.6, -0.2, 0.12])
T_ball = np.eye(4)
T_ball[:3, 3] = obs['position']
obs['transform'] = T_ball
obs['bounds'] = np.array([(ee_radius + obs['radius']) ** 2, 1e6]) 
obstacles.append(obs)


### CAPSULES ###
capsules = []

capsules.append(create_moving_capsule('arm','link02',0,0.05,0.35,[0,0,0],rotation_offset=[0,0,0]))
capsules.append(create_moving_capsule('ee','link05',0,0.055,0.22,[0,0,0],None))
#create_capsule('moving','link03',0.05,0.1,capsules,-0.0,color=[0,0,1,0.3],rotation_offset=-np.pi/6)
capsules.append(create_moving_capsule('forearm','link03',0,0.05,0.16,[0.06,0,0.05],rotation_offset=[0,-np.pi/50,0],color=[0,0,1,0.3]))
# capsules.append(create_fixed_capsule('fixed1',0.025,np.array([0.5, 0.175, 0.]),np.array([0.5, 0.175, 0.225]),color=[0, 1, 1, 1]))
# capsules.append(create_fixed_capsule('fixed2',0.025,np.array([0.5, -0.175, 0.]),np.array([0.5, -0.175, 0.225]),color=[0, 1, 1, 1]))
# capsules.append(create_fixed_capsule('fixed3',0.025,np.array([0.5, -0.175, 0.225]),np.array([0.5, 0.175, 0.225]),color=[0, 1, 1, 1]))

### Pairs: CAPSULE-CAPSULE -> type 0, CAPSULE-BALL -> type 1, CAPSULE-FLOOR type 2. For now insert
# elements in the pairs in this order                                                       ###
capsule_pairs = []
capsule_pairs.append(assign_pairs('arm','ee',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','floor',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','floor',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','ball1',obstacles,capsules))
capsule_pairs.append(assign_pairs('forearm','ball2',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','ball1',obstacles,capsules))
capsule_pairs.append(assign_pairs('ee','ball2',obstacles,capsules))
# capsule_pairs.append(assign_pairs('forearm','fixed1',obstacles,capsules))
# capsule_pairs.append(assign_pairs('forearm','fixed2',obstacles,capsules))
# capsule_pairs.append(assign_pairs('forearm','fixed3',obstacles,capsules))
# capsule_pairs.append(assign_pairs('ee','fixed1',obstacles,capsules))
# capsule_pairs.append(assign_pairs('ee','fixed2',obstacles,capsules))
# capsule_pairs.append(assign_pairs('ee','fixed3',obstacles,capsules))


nls = {
    'relu': torch.nn.ReLU(),
    'elu': torch.nn.ELU(),
    'tanh': torch.nn.Tanh(),
    'sine': Sine(),
    'gelu': torch.nn.GELU(approximate='tanh'),
    'silu': torch.nn.SiLU()
}

CUSTOM_PARAMS = {
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 6,
    'lines.markersize': 12,
    'patch.linewidth': 2,
    'axes.grid': True,
    'axes.labelsize': 35,
    'font.family': 'serif',
    'font.size': 30,
    'text.usetex': True,
    'legend.fontsize': 20,
    'legend.loc': 'best',
    'figure.figsize': (10, 7),
    'figure.facecolor': 'white',
    'grid.linestyle': '-',
    'grid.alpha': 0.7,
    'savefig.format': 'pdf'
}

def apply_rc_params():
    plt.rcParams.update(CUSTOM_PARAMS)