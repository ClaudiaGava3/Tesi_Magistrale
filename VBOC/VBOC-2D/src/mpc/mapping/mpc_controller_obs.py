from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from mpc_abstract_obs import AbstractController
from learning import NeuralNetwork


class MpcController(AbstractController):
    """MPC controller using a Neural Network as a terminal constraint."""

    def __init__(self, model) -> None:
        super().__init__(model)
        
        # 2. LOAD NEURAL NETWORK
        nn_filename = f"{self.params.NN_DIR}{self.params.robot_name}_{self.params.act}.pt"
        print(f"--- Loading neural network from {nn_filename} ---")
        
        checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
        self.mean_X = checkpoint['mean']
        self.std_X = checkpoint['std']
        
        # Neural network initialization (4D input for 2D: theta, vx, vz, wy)
        # NOTE: make sure to use the same activation as training (e.g. GELU)
        net = NeuralNetwork(
            input_size=4, 
            hidden_size=self.params.hidden_size, 
            output_size=1, 
            number_hidden=self.params.hidden_layers, 
            activation=torch.nn.GELU(approximate='tanh'), 
            ub=1
        )
        net.load_state_dict(checkpoint['model'])
        net.eval()
        
        # 3. L4CASADI AND TERMINAL CONSTRAINT
        # In the MPC state [x, z, theta, vx, vz, wy], the network uses indices [2, 3, 4, 5]
        theta_sym = self.model.x[2]
        vx_sym = self.model.x[3]
        vz_sym = self.model.x[4]
        wy_sym = self.model.x[5]
        x_nn_sym = cs.vertcat(theta_sym, vx_sym, vz_sym, wy_sym)
        
        # Normalize the input for the network
        x_norm = (x_nn_sym - cs.DM(self.mean_X)) / cs.DM(self.std_X)
        
        # L4CasADi function
        self.l4c_model = l4c.L4CasADi(net, name="drone_viability_net")
        
        # Network output: predicted braking margin (Alpha)
        alpha_pred = self.l4c_model(x_norm.T)


        # Extract the drone's future position and box limits from parameters
        x_drone = self.model.x[0]
        z_drone = self.model.x[1]
        x_min_box = self.model.p[0]
        x_max_box = self.model.p[1]
        z_min_box = self.model.p[2]
        z_max_box = self.model.p[3]

        # --- PATH CONSTRAINT (4 ASYMMETRIC SIDES) ---
        # All 4 distances to the walls must be >= 0
        h_expr_path = cs.vertcat(
            x_drone - x_min_box,  # Left wall
            x_max_box - x_drone,  # Right wall
            z_drone - z_min_box,  # Floor
            z_max_box - z_drone   # Ceiling
        )
        self.ocp.model.con_h_expr = h_expr_path
        # Imponiamo che tutti e 4 i valori siano compresi tra 0 e infinito
        self.ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0])
        self.ocp.constraints.uh = np.array([1e5, 1e5, 1e5, 1e5])


        # --- TERMINAL CONSTRAINT (4 ASYMMETRIC SIDES) ---
        # All 4 distances to the walls minus the predicted alpha must be >= 0
        h_expr_terminal = cs.vertcat(
            x_drone - x_min_box - alpha_pred,  # Left wall
            x_max_box - x_drone - alpha_pred,  # Right wall
            z_drone - z_min_box - alpha_pred,  # Floor
            z_max_box - z_drone - alpha_pred   # Ceiling
        )

        self.ocp.model.con_h_expr_e = h_expr_terminal
        # Imponiamo che tutti e 4 i valori siano compresi tra 0 e infinito
        self.ocp.constraints.lh_e = np.array([0.0, 0.0, 0.0, 0.0])
        self.ocp.constraints.uh_e = np.array([1e5, 1e5, 1e5, 1e5])

        # if h_expr_terminal<0: print("⚠️ WARNING: TERMINAL CONSTRAINT IS NEGATIVE!")

        
        # SOLVER COMPILATION
        gen_name = self.params.GEN_DIR + 'ocp_mpc_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name

        self.ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name

        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)


    def solve_step(
        self,
        current_x: np.ndarray,
        x_ref: np.ndarray,
        box_abs: np.ndarray # Array di 4 elementi [x_min, x_max, z_min, z_max]
        #obs_x: float
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Performs a single MPC optimization step.
        Returns: (state_trajectory, input_trajectory, predicted_alpha, status)
        """
        self.ocp_solver.reset()

        # Initial state set to current sensor measurements
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)


        # 2. Parameter vector: [alpha_real_min(1), x_ref(6)]
        p_val = np.hstack([box_abs, x_ref])

        
        # 2. WARM-START AND PARAMETERS
        for i in range(self.N):
            if i == 0:
                self.ocp_solver.set(i, 'x', current_x)
            else:
                self.ocp_solver.set(i, 'x', self.x_guess[i])
                
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', p_val)



        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p_val)

        
        
        # Solver
        status = self.ocp_solver.solve()

        # --- DETAILED DEBUG ANALYSIS BLOCK ---
        # x_terminal = self.ocp_solver.get(self.N, 'x')
        # x_norm = (x_terminal[2:6] - self.mean_X) / self.std_X
        # with torch.no_grad():
        #     alpha_pred_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()

        # # Compute the 4 terminal distances to the current walls
        # dist_sx = x_terminal[0] - box_abs[0]
        # dist_dx = box_abs[1] - x_terminal[0]
        # dist_pav = x_terminal[1] - box_abs[2]
        # dist_sof = box_abs[3] - x_terminal[1]

        # print(f"\n--- DEBUG NODE N (Status {status}) ---")
        # print(f"Alpha Required (Network): {alpha_pred_val:.3f}m")
        # print(f"Available Distances: LEFT: {dist_sx:.3f} | RIGHT: {dist_dx:.3f} | FLOOR: {dist_pav:.3f} | CEILING: {dist_sof:.3f}")
        
        # # See which constraint is violated
        # violations = []
        # if dist_sx < alpha_pred_val: violations.append("LEFT WALL")
        # if dist_dx < alpha_pred_val: violations.append("RIGHT WALL")
        # if dist_pav < alpha_pred_val: violations.append("FLOOR")
        # if dist_sof < alpha_pred_val: violations.append("CEILING")
        
        # if violations:
        #     print(f"❌ CONSTRAINTS VIOLATED: {', '.join(violations)}")
        # else:
        #     print(f"✅ ALL CONSTRAINTS SATISFIED")
        # # -------------------------------------------

        # Query the network to return the correct value even if it fails
        x_terminal = self.ocp_solver.get(self.N, 'x')
        x_norm = (x_terminal[2:6] - self.mean_X) / self.std_X
        with torch.no_grad():
            alpha_pred_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()

        # --- THE CORRECT DEBUG PRINT GOES HERE ---
        # Compute the REAL DISTANCE AT THE TERMINAL NODE
        dist_x_min = x_terminal[0] - box_abs[0]
        dist_x_max = box_abs[1] - x_terminal[0]
        dist_z_min = x_terminal[1] - box_abs[2]
        dist_z_max = box_abs[3] - x_terminal[1]
        min_dist_to_wall = min(dist_x_min, dist_x_max, dist_z_min, dist_z_max)
        
        print(f" --> (Status {status}, alpha_pred {alpha_pred_val:.2f}, min_dist_muro {min_dist_to_wall:.2f})")
        # ------------------------------------------
        
        if status in [0, 2]: # Success or acceptable suboptimal
            x_sol = np.empty((self.N + 1, self.model.nx))
            u_sol = np.empty((self.N, self.model.nu))
            
            for i in range(self.N):
                x_sol[i] = self.ocp_solver.get(i, 'x')
                u_sol[i] = self.ocp_solver.get(i, 'u')
            x_sol[self.N] = self.ocp_solver.get(self.N, 'x')
            


            # Update warm-start
            new_x_guess = np.vstack([x_sol[1:], x_sol[-1]])
            new_u_guess = np.vstack([u_sol[1:], u_sol[-1]])
            self.setGuess(new_x_guess, new_u_guess)
            
            return x_sol, u_sol, alpha_pred_val, status
        else:
            return None, None, alpha_pred_val, status