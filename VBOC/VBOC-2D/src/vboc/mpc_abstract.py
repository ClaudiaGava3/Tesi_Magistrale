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

    Builds the full symbolic state-space representation of a Star-shaped
    Tilted Hexarotor (STH), including rotation matrices, thrust/torque
    allocation matrices, explicit dynamics, state/input bounds, and the
    nonlinear obstacle-avoidance constraints required by the OCP.

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
        

        # --- State and input dimensions 2D---
        nq = 3  # Pose: 2 position + 1 orientation (Euler angles)
        nv = 3
        nu = 2  # Input: 2 squared rotor spinning rates
        npos = 2    # Position sub-space dimension
        nori = 1    # Orientation sub-space dimension
        #nbox = 4    # Obstacle box constraint dimension
        #nscale = 1  # FATTORE DI SCALA (Sostituisce nbox)

      # --- NOVITÀ MPC: I Parametri ---
        self.alpha_real = MX.sym('alpha_real', 1)   # Lo spazio letto dai sensori
        self.x_ref = MX.sym('x_ref', nq+nv)       # Il target da raggiungere (6 elementi)
        
        # Uniamo tutto nel vettore dei parametri 'p'
        self.p = vertcat(self.alpha_real, self.x_ref)
        
        # --- Acados model registration ---
        self.acados_model = AcadosModel()
        self.acados_model.name = params.robot_name
        
        # ASSEGNAZIONE CORRETTA: lo passiamo ad acados_model, non a ocp!
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
        roll, pitch, yaw = 0.0, euler_angles[0], 0.0

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
            [0,0],
            [0,0],
            [1,1]
        ])


        # exacopter in 3D
        # self.F = self.cf * np.array([
        #     [0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a, 
        #      0, np.sqrt(3)/2 * sin_a, -np.sqrt(3)/2 * sin_a],
        #     [sin_a, -1/2 * sin_a, -1/2 * sin_a, 
        #      sin_a, -1/2 * sin_a, -1/2 * sin_a],
        #     [cos_a, cos_a, cos_a, 
        #      cos_a, cos_a, cos_a]
        # ])

        # M maps squared rotor speeds to body-frame torque components [3 × nu]
        # in 2D
        self.M= self.cf * np.array([
            [0.0,0.0],
            [-self.l,self.l],
            [0.0,0.0]
        ])


        # exacopter in 3D
        # self.M = self.ct * np.array([
        #     [0, 
        #      np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 
        #      np.sqrt(3)/2 * self.r * cos_a - np.sqrt(3)/2 * sin_a, 
        #      0, 
        #      -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a, 
        #      -np.sqrt(3)/2 * self.r * cos_a + np.sqrt(3)/2 * sin_a],
        #     [-self.r * cos_a + sin_a, 
        #      -1/2 * self.r * cos_a + 1/2 * sin_a, 
        #       1/2 * self.r * cos_a - 1/2 * sin_a, 
        #       self.r * cos_a - sin_a, 
        #       1/2 * self.r * cos_a - 1/2 * sin_a,
        #      -1/2 * self.r * cos_a + 1/2 * sin_a],
        #     [self.r * sin_a + cos_a, 
        #      -self.r * sin_a - cos_a, 
        #       self.r * sin_a + cos_a, 
        #      -self.r * sin_a - cos_a, 
        #       self.r * sin_a + cos_a, 
        #      -self.r * sin_a - cos_a]
        # ])

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

        omega_3d = vertcat(0.0, self.x[5], 0.0)

        euler_rates_3d = self.Tinv(self.x) @ omega_3d
        accel_3d = -self.g*np.array([[0], [0], [1]]) + self.fc(self.x, self.u) / self.mass
        alpha_3d = np.linalg.inv(self.J) @ (-cross(omega_3d, self.J @ omega_3d)) + np.linalg.inv(self.J) @ self.tc(self.u)

        # --- Explicit continuous-time dynamics: x_dot = f(x, u) ---

        # 2D
        self.f_expl = vertcat(
            self.x[3:5],         # x_dot, z_dot
            euler_rates_3d[1],   # theta_dot (derivato dal tuo Tinv!)
            accel_3d[0],         # vx_dot
            accel_3d[2],         # vz_dot
            alpha_3d[1],         # wy_dot (derivato dalla tua equazione con cross!)
            #0.0, 0.0, 0.0, 0.0,   # Derivate dei 4 parametri del box (costanti)
            #0.0     # Derivata dello scaling (costante)
            
        )
       

         # --- Acados model registration ---
        self.acados_model.f_expl_expr = self.f_expl

        # exacopter in 3D
        # self.f_expl = vertcat(
        #     self.x[nq:nq+npos],
        #     self.Tinv(self.x)@self.x[nq+npos:],
        #     -self.g*np.array([[0], [0], [1]]) + self.fc(self.x, self.u) /
        #     self.mass, 
        #     np.linalg.inv(self.J) @ (
        #         -cross(self.x[nq+npos:], self.J @ self.x[nq+npos:])) + 
        #         np.linalg.inv(self.J) @ self.tc(self.u),
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0
        # )

        # --- Input bounds ---
        self.u_max = np.full(nu, self.u_bar)
        self.u_min = np.zeros((nu,))

        # --- Feasible orientation bounds (derived from actuator saturation) 
        # ---
        # Three regimes depending on the ratio between u_bar and hovering thrust
        # if self.u_bar >= (self.mass*self.g)/(2*self.cf*cos_a):
        #     ri = abs(-self.mass*self.g/2 * tan_a)
        #     ro = self.mass*self.g * tan_a
        # elif (self.mass*self.g) / (4*self.cf*cos_a) <= self.u_bar \
        #       < (self.mass*self.g) / (2*self.cf*cos_a):
        #     ri = min(
        #         abs(-self.mass*self.g/2 * tan_a), 
        #         abs(3*self.cf*self.u_bar*sin_a -self.mass*self.g/2 * tan_a)
        #     )   
        #     ro = np.linalg.norm(np.array([
        #         np.sqrt(3)*self.cf*self.u_bar*sin_a, 
        #         self.mass*self.g * tan_a - 3*self.cf*self.u_bar*sin_a
        #     ]))
        # else:
        #     ri = 3*self.cf*self.u_bar*sin_a - self.mass*self.g/2 * tan_a
        #     ro = self.mass * self.g * tan_a - 6* self.cf * self.u_bar * sin_a

        # # Maximum tilt angles admissible for hovering and full actuation
        # self.phi_hovering = np.arctan2(ri, self.mass * self.g)
        # self.phi_hovering_max = np.arctan2(ro, self.mass * self.g) 
        # self.phi_max = np.arccos(
        #     (self.mass*self.g) / 
        #     (self.cf * 6 * np.cos(self.alpha_tilt)*self.u_bar)
        # )
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
        #in 2D
        box_normals = [
            DM([1.0,0.0,0.0]),
            DM([0.0,0.0,1.0]),
            DM([-1.0,0.0,0.0]),
            DM([0.0,0.0,-1.0])
        ]


        # Build one scalar constraint per face
        self.con_h_expr_list = []
        # Gli indici delle dimensioni del box in self.x sono 6, 7, 8, 9
        #box_vars_indices = [6, 7, 8, 9]
        
        #TENTATIVO 1 CON SOLO DIM BOX
        # for i,n in enumerate(box_normals):
        #     box_dim = self.x[box_vars_indices[i]]
        #     expr = sqrt(n.T @ self.Q(self.x) @ n) - box_dim
        #     self.con_h_expr_list.append(expr)

        # self.con_h_expr = vertcat(*self.con_h_expr_list)

        #TENTATIVO 2 CON PROIEZIONE
        # for i, n in enumerate(box_normals):
        #     # Dimensione = fattore_di_scala (x[6]) * dimensione_base_fissa (p[i])
        #     #in 2D
        #     box_dim = self.x[10] * self.x[6+i]
        #     pos_proj = n[0] * self.x[0] + n[2] * self.x[1]
            
        #     expr = pos_proj + sqrt(n.T @ self.Q(self.x) @ n) - box_dim
        #     self.con_h_expr_list.append(expr)

        # self.con_h_expr = vertcat(*self.con_h_expr_list)

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

        # --- Stage cost: minimize box size ---
        # Costo allo stadio iniziale
        from scipy.linalg import block_diag # Assicurati di importarlo all'inizio del file!

        # --- Stage cost: Target tracking ---
        # --- Stage cost: Inseguimento Target (EXTERNAL) ---
        self.ocp.cost.cost_type = 'EXTERNAL'
        self.ocp.cost.cost_type_e = 'EXTERNAL'
        
        # Definiamo i pesi (Q_cost per lo stato, R_cost per i motori)
        # Ordine stato: [x, z, theta, vx, vz, wy]
        Q_cost = diag(vertcat(1000.0, 1000.0, 20.0, 50.0, 50.0, 10.0))
        R_cost = diag(vertcat(0.00001, 0.00001))
        
        # Calcoliamo l'errore tra dove siamo (x) e dove vogliamo andare (x_ref)
        err_x = self.model.x - self.model.x_ref
        # Calcoliamo la spinta teorica per contrastare la gravità
        # u_hover = (m * g) / (2 * cf)
        u_hover_val = (self.model.mass * self.model.g) / (2.0 * self.model.cf)
        err_u = self.model.u - u_hover_val
        
        
        # Costruiamo le espressioni simboliche (Esattamente come chiedevi tu!)
        cost_step = err_x.T @ Q_cost @ err_x + err_u.T @ R_cost @ err_u
        # Al nodo finale non si usano i motori
        cost_terminal = err_x.T @ Q_cost @ err_x * 20
        
        # Assegniamo le espressioni ad Acados
        self.ocp.model.cost_expr_ext_cost = cost_step
        self.ocp.model.cost_expr_ext_cost_e = cost_terminal

        # #in 2D
        # self.ocp.model.cost_expr_ext_cost_0 = 1.0 * self.model.x[10] # Costo = Scaling
        
        # Inizializziamo i parametri con dei valori di default (1 per alpha, 0 per x_ref)
        self.ocp.parameter_values = np.zeros(1 + self.model.nx)
        self.ocp.parameter_values[0] = 1.0


        # --- Initial shooting node: position and velocity fixed ---
        # Fissiamo x, z, theta, vx, vz, wy (Primi 6 indici: 0,1,2,3,4,5)
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nq + self.model.nv) 
        self.ocp.constraints.lbx_0 = np.zeros(self.model.nq + self.model.nv)
        self.ocp.constraints.ubx_0 = np.zeros(self.model.nq + self.model.nv)

        # --- Path constraints ---
        # Nonlinear obstacle constraint (one per box face)
       # self.ocp.model.con_h_expr = self.model.con_h_expr
        # self.ocp.constraints.uh = np.full(self.model.nbox, 0.0)
        # self.ocp.constraints.lh = np.full(self.model.nbox, -1e5)

        # State box: theta (1), velocity (3), box (4)scale (1) = 9 elementi totali
        self.ocp.constraints.idxbx = np.arange(self.model.npos, self.model.nx) 
        self.ocp.constraints.lbx = np.hstack([
            np.full(self.model.nori, -np.pi), 
            np.full(self.model.nv, -2),
            # np.full(self.model.nbox, 0.1),             # <--- Minimo per i lati (es. 10% della scala)     
            # np.array([0.0])                   # scale min
        ])
        self.ocp.constraints.ubx = np.hstack([
            np.full(self.model.nori, np.pi),  
            np.full(self.model.nv, 2),
            # np.full(self.model.nbox, 1.0),             # <--- MAX per i lati imposto a 1.0!      
            # np.array([1e5])                    # scale max
        ])

        # --- Terminal constraints ---
        # non lo obbligo a fermarsi!!!!!
        self.ocp.constraints.idxbx_e = np.arange(self.model.npos, self.model.nx)
        self.ocp.constraints.lbx_e = np.hstack([np.full(self.model.nori, -np.pi/2), np.full(self.model.nv, -1)
        ])
        self.ocp.constraints.ubx_e = np.hstack([np.full(self.model.nori, np.pi/2),  np.full(self.model.nv, 1)
        ])


        # Terminal obstacle constraint (same expression as path)
       # self.ocp.model.con_h_expr_e = self.model.con_h_expr 
        # self.ocp.constraints.uh_e = np.full(self.model.nbox, 0.0)
        # self.ocp.constraints.lh_e = np.full(self.model.nbox, -1e5)


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