
# ==========================================================================================================================================================
"""Define the tridimensional drone model and the base feedback-MPC formulation.

The model builds the symbolic rigid-body dynamics, rotor mappings, and state
and input dimensions used by Acados. The abstract controller configures the
tracking cost, wall-clearance path constraint, state and motor bounds, solver
options, and warm-start storage used by specialized MPC controllers.
"""
# ==========================================================================================================================================================


from __future__ import annotations
import re
import numpy as np
from casadi import (
    MX, DM, horzcat, vertcat, dot, Function,
    sin, cos, tan, cross, fabs, sqrt, diag
)
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

class Model:
    """Multirotor rigid-body model with CasADi symbolic expressions for acados.

    Builds the symbolic planar state-space representation of a Star-shaped
    Tilted Hexarotor (STH), including rotation matrices, thrust/torque
    allocation matrices, explicit dynamics, and the geometric representation
    of the drone footprint.

    Parameters
    ----------
    params : Parameters
        Configuration object containing all physical and solver parameters.
    """
    def __init__(self, params: object) -> None:
        self.params = params
        
        # --- Physical parameters ---
        self.mass = params.mass
        self.J = params.J
        self.l = params.l 
        self.cf = params.cf
        self.ct = params.ct
        self.r = self.cf / self.ct * self.l # Effective moment arm ratio
        self.g = 9.81   # Gravitational acceleration [m/s²]
        self.u_bar = params.u_bar 
        self.alpha_tilt = params.alpha_tilt
        self.eps = params.state_tol
        self.vboc_repeat = params.vboc_repeat

        # --- Environment and drone size bounds ---
        self.min_width = params.min_width
        self.min_length = params.min_length
        self.min_height = params.min_height
        self.max_width = params.max_width
        self.max_length = params.max_length
        self.max_height = params.max_height
        self.v_min = params.v_min
        self.v_max = params.v_max
        

        # --- State and input dimensions 3D---
        nq = 6  # Pose: 3 position + 3 orientation (Euler angles)
        nv = 6
        nu = 4  # Input: 4 squared rotor spinning rates
        npos = 3    # Position sub-space dimension
        nori = 3    # Orientation sub-space dimension
        #nbox = 6    # Obstacle box constraint dimension
        #nscale = 1  # FATTORE DI SCALA (Sostituisce nbox)

      # --- PARAMETERS MPC ---
        self.alpha_real = MX.sym('alpha_real', 1)   # Lo spazio letto dai sensori
        self.x_ref = MX.sym('x_ref', nq+nv)       # Il target da raggiungere (6 elementi)
        
        self.p = vertcat(self.alpha_real, self.x_ref)
        
        # --- Acados model registration ---
        self.acados_model = AcadosModel()
        self.acados_model.name = params.robot_name
        
        self.acados_model.p = self.p

        

        # --- CasADi symbolic variables ---
        self.x = MX.sym("x", nq + nv)    # Full state [q; q_dot]
        self.x_dot = MX.sym("x_dot", nq + nv)
        self.u = MX.sym("u", nu)    # Control input
        #self.p = MX.sym("p", nbox)    # OCP parameter x fixing box ratio

        # --- Acados model registration ---
        self.acados_model.x = self.x
        self.acados_model.u = self.u
            
        # --- Rotation matrix (ZYX Euler convention) ---
        euler_angles = self.x[npos:npos+nori] 
        roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

        R_x = vertcat(
            horzcat(1,          0,          0   ),
            horzcat(0,  cos(roll),  -sin(roll)  ),
            horzcat(0,  sin(roll),   cos(roll)   )
        )
        R_y = vertcat(
            horzcat( cos(pitch),     0,     sin(pitch)),
            horzcat(          0,     1,              0),
            horzcat(-sin(pitch),     0,     cos(pitch))
        )
        R_z = vertcat(
            horzcat(cos(yaw), -sin(yaw), 0),
            horzcat(sin(yaw),  cos(yaw), 0),
            horzcat(       0,         0, 1)
        )

        # Total rotation matrix: world ← body
        R_tot = R_z @ R_y @ R_x
        self.R = Function('R', [self.x], [R_tot])

        # --- Thrust and torque allocation matrices ---
        sin_a = np.sin(self.alpha_tilt)
        cos_a = np.cos(self.alpha_tilt)
        tan_a = np.tan(self.alpha_tilt)

        # F maps squared rotor speeds to body-frame force components [3 × nu]
        # in 2D
        self.F=self.cf*np.array([
            [0,0,0,0],
            [0,0,0,0],
            [1,1,1,1]
        ])


        # M maps squared rotor speeds to body-frame torque components [3 × nu]
        # in 2D
        self.M = np.array([
            [0,      self.l*self.cf,  0,      -self.l*self.cf], # Roll
            [-self.l*self.cf, 0,      self.l*self.cf,  0     ], # Pitch
            [self.ct, -self.ct,      self.ct, -self.ct       ]  # Yaw
        ])


        # CasADi functions for net control force and torque
        self.fc = Function(
            'fc', 
            [self.x, self.u], 
            [self.R(self.x) @ self.F @ self.u]
        )
        self.tc = Function(
            'tc', 
            [self.u], 
            [self.M @ self.u]
        )

        # --- Euler-rate to angular-velocity transformation (inverse) ---
        Tinv_expr = vertcat(
            horzcat(1, sin(roll)*tan(pitch),  cos(roll)*tan(pitch)),
            horzcat(0,            cos(roll),            -sin(roll)),
            horzcat(0, sin(roll)/cos(pitch),  cos(roll)/cos(pitch))
        )
        self.Tinv = Function('Tinv', [self.x], [Tinv_expr])

        omega_3d = self.x[9:12]

        euler_rates_3d = self.Tinv(self.x) @ omega_3d
        accel_3d = -self.g*np.array([[0], [0], [1]]) + self.fc(self.x, self.u) / self.mass
        alpha_3d = np.linalg.inv(self.J) @ (-cross(omega_3d, self.J @ omega_3d)) + np.linalg.inv(self.J) @ self.tc(self.u)

        # --- Explicit continuous-time dynamics: x_dot = f(x, u) ---
# 3D
        self.f_expl = vertcat(
            self.x[6:9],         # x_dot, y_dot, z_dot
            euler_rates_3d,      # phi_dot, theta_dot, psi_dot
            accel_3d,            # vx_dot, vy_dot, vz_dot
            alpha_3d             # wx_dot, wy_dot, wz_dot
            #0.0, 0.0, 0.0, 0.0,   # Derivate dei 4 parametri del box (costanti)
            #0.0     # Derivata dello scaling (costante)
            
        )
       

         # --- Acados model registration ---
        self.acados_model.f_expl_expr = self.f_expl


        # --- Input bounds ---
        self.u_max = np.full(nu, self.u_bar)
        self.u_min = np.zeros((nu,))

        self.phi_max = np.pi/2

        # --- Ellipsoidal obstacle-avoidance constraint ---
        # Q(x) = R(x) · diag(half-axes²) · R(x)ᵀ encodes drone occupancy as
        # a rotation-aware ellipsoid; constraint: nᵢᵀ p + √(nᵢᵀ Q nᵢ) ≤ bound_i
        D = diag(vertcat(
            self.min_width**2, 
            self.min_length**2, 
            self.min_height**2
        ))
        self.Q = Function(
            'Q', 
            [self.x], 
            [self.R(self.x) @ D @ self.R(self.x).T]
        )

        # Outward normals of the six faces of the axis-aligned bounding box
        box_normals = [
            DM([1.0,0.0,0.0]),
            DM([0.0,1.0,0.0]),
            DM([0.0,0.0,1.0]),
            DM([-1.0,0.0,0.0]),
            DM([0.0,-1.0,0.0]),
            DM([0.0,0.0,-1.0])
        ]


        # Build one scalar constraint per face
        self.con_h_expr_list = []
        
        
        # Signed environment extents used as constraint bounds [m]
        self.env_dimensions = np.array([
            -self.max_width, -self.max_length, -self.max_height,
             self.max_width, self.max_length, self.max_height
        ]) 
        
        # Signed drone half-extents (same sign convention) [m]
        self.drone_occupancy = np.array([
            -self.min_width, -self.min_length, -self.min_height,
             self.min_width, self.min_length, self.min_height
        ]) 

        # --- Acados model registration ---
        self.amodel = AcadosModel()
        self.amodel.name = params.robot_name
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.f_expl_expr = self.f_expl
        self.amodel.p = self.p
        
        # Derived dimension attributes exposed to controllers
        self.nx = self.amodel.x.size()[0]   # Full state dimension
        self.nu = self.amodel.u.size()[0]   # Input dimension
        self.ny = self.nx + self.nu         # Output dimension (state + input)
        self.nq = nq                        # Generalised coordinate dimension
        self.nv = nq                        # Velocity dimension (= nq for 
                                            # this model)
        self.npos = npos                    # Position sub-space dimension
        self.nori = nori                    # Orientation sub-space dimension
        #self.nbox = self.con_h_expr.size()[0]  # Number of box-face constraints

class AbstractController:
    """Base class for acados-based OCP controllers.

    Builds and compiles a shared AcadosOcp object from the supplied model and
    parameter set. Subclasses implement specific solve strategies on top of
    the common solver infrastructure created here.

    Parameters
    ----------
    model : Model
        Symbolic robot model providing dynamics, constraints, and dimensions.
    """

    def __init__(self, model: object) -> None:
        # Derive a human-readable OCP name from the concrete subclass name
        self.ocp_name = "".join(
            re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model

        self.N = self.params.N
        self.ocp = AcadosOcp()
        self.vboc_repeat = self.params.vboc_repeat
        self.tol=self.params.state_tol

        # --- Horizon and time discretisation ---
        self.ocp.solver_options.tf = self.params.N * self.params.dt
        self.ocp.dims.N = self.N

        # --- Symbolic model ---
        self.ocp.model = self.model.amodel

        from scipy.linalg import block_diag # Make sure to import it at the top of the file!

        # --- External stage and terminal costs for target tracking ---
        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.cost.cost_type_e = 'EXTERNAL'
        
        # Weights (Q_cost for state, R_cost for motors)
        # State order: [x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz]
        Q_cost = diag(vertcat(
                100.0, 100.0, 100.0, 20.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        R_cost = diag(vertcat(0.0001, 0.0001, 0.0001, 0.0001))
        
        
        err_x = self.model.x - self.model.x_ref
        
        u_hover_val = (self.model.mass * self.model.g) / (4.0 * self.model.cf)
        err_u = self.model.u - u_hover_val
        

        cost_step = err_x.T @ Q_cost @ err_x + err_u.T @ R_cost @ err_u
        # At the terminal node I do not penalize motor effort
        cost_terminal = err_x.T @ Q_cost @ err_x * 20
        
        # Assegno le espressioni ad Acados
        self.ocp.model.cost_expr_ext_cost = cost_step
        self.ocp.model.cost_expr_ext_cost_e = cost_terminal

        
        
        # Default parameters: wall position 0 and zero target state.
        self.ocp.parameter_values = np.zeros(1 + self.model.nx)
        self.ocp.parameter_values[0] = 1.0


        # --- Initial shooting node: the complete twelve-element state is fixed ---
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nq + self.model.nv) 
        self.ocp.constraints.lbx_0 = np.zeros(self.model.nq + self.model.nv)
        self.ocp.constraints.ubx_0 = np.zeros(self.model.nq + self.model.nv)

        # --- Path constraints ---
        # Simplified constraint only for X
        # Add the physical constraint: the drone must not cross the wall
        # obs_x is self.model.p[0], drone x is self.model.x[0]
        self.ocp.model.con_h_expr = self.model.p[0] - self.model.x[0]
        self.ocp.constraints.lh = np.array([0.0])   # Distance to the wall >= 0
        self.ocp.constraints.uh = np.array([1e5])   # Up to infinity

        # Bound orientation and the three velocity components at each stage.
        self.ocp.constraints.idxbx = np.arange(self.model.npos, self.model.nx) 
        self.ocp.constraints.lbx = np.hstack([
            np.full(self.model.nori, -np.pi), 
            np.full(self.model.nv, -1e1)
        ])
        self.ocp.constraints.ubx = np.hstack([
            np.full(self.model.nori, np.pi),  
            np.full(self.model.nv, 1e1)
        ])

        # --- Terminal constraints for NN ---
        self.ocp.constraints.idxbx_e = np.arange(self.model.npos, self.model.nx)
        self.ocp.constraints.lbx_e = np.hstack([np.full(self.model.nori, -np.pi/2), np.full(self.model.nv, -1)
        ])
        self.ocp.constraints.ubx_e = np.hstack([np.full(self.model.nori, np.pi/2),  np.full(self.model.nv, 1)
        ])



        # --- Input bounds: rotor speeds non-negative and below saturation ---
        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)

        # --- Solver options ---
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0   # Constraint Hessian 
                                                        # approximated
        self.ocp.solver_options.exact_hess_dyn = 0  # Dynamics Hessian 
                                                    # approximated
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.globalization_alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.globalization_alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt
        self.ocp.solver_options.tol = self.params.state_tol
        self.ocp.solver_options.print_level = 0  # Suppress solver output

        # --- Code generation and solver compilation ---
        # gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        # self.ocp.code_export_directory = gen_name
        # self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # --- Warm-start storage ---
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        #self.tol = self.params.cost_tol

    def setGuess(
        self,
        x_guess: np.ndarray,
        u_guess: np.ndarray,
    ) -> None:
        """Store a new warm-start trajectory for the next solver call.

        Parameters
        ----------
        x_guess : np.ndarray
            State trajectory used as initial guess, shape (N, nx).
        u_guess : np.ndarray
            Input trajectory used as initial guess, shape (N, nu).
        """
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a copy of the current warm-start trajectory.

        Returns
        -------
        x_guess : np.ndarray
            Copy of the stored state trajectory, shape (N, nx).
        u_guess : np.ndarray
            Copy of the stored input trajectory, shape (N, nu).
        """
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N: int) -> None:
        """Update the solver horizon length and re-condition the QP.

        Parameters
        ----------
        N : int
            New prediction horizon length (number of shooting intervals).
        """
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)