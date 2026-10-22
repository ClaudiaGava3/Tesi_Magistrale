
"""Add mapped-box constraints and neural-network evaluation to feedback MPC.

The controller loads the trained network and embeds it through L4CasADi. At
each MPC stage it constrains the drone center to the four mapped box limits;
the terminal constraint currently also enforces box containment, while the
predicted network margin is evaluated and returned for diagnostics but is not
subtracted from the active terminal constraint.
"""

from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from mpc_abstract_obs import AbstractController
from learning import NeuralNetwork


class MpcController(AbstractController):
    """MPC controller using mapped bounds and a trained viability network."""

    def __init__(self, model) -> None:
        super().__init__(model)
        
        # 2. LOAD NEURAL NETWORK
        nn_filename = f"{self.params.NN_DIR}{self.params.robot_name}_{self.params.act}.pt"
        print(f"--- Caricamento Rete Neurale da {nn_filename} ---")
        
        checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
        self.mean_X = checkpoint['mean']
        self.std_X = checkpoint['std']
        
        # Neural network initialization (4D input for 2D: theta, vx, vz, wy)
        net = NeuralNetwork(
            input_size=9, 
            hidden_size=self.params.hidden_size, 
            output_size=1, 
            number_hidden=self.params.hidden_layers, 
            activation=torch.nn.GELU(approximate='tanh'), 
            ub=1
        )
        net.load_state_dict(checkpoint['model'])
        net.eval()
        
        
        x_nn_sym = self.model.x[3:12]
        
        # Input normalization
        x_norm = (x_nn_sym - cs.DM(self.mean_X)) / cs.DM(self.std_X)
        
        # Function L4CasADi
        self.l4c_model = l4c.L4CasADi(net, name="drone_viability_net")
        
        # NN output: predicted alpha
        alpha_pred = self.l4c_model(x_norm.T)
        

        # Use the stage position and mapped box limits supplied as OCP parameters.
        x_drone = self.model.x[0]
        y_drone = self.model.x[1]
        z_drone = self.model.x[2]
        
        x_min_box = self.model.p[0]
        x_max_box = self.model.p[1]
        y_min_box = self.model.p[2]
        y_max_box = self.model.p[3]
        z_min_box = self.model.p[4]
        z_max_box = self.model.p[5]


        # --- PATH CONSTRAINT ---
        h_expr_path = cs.vertcat(
            x_drone - x_min_box,  # Wall SX
            x_max_box - x_drone,  # Wall DX
            y_drone - y_min_box,  # Wall FRONT
            y_max_box - y_drone,  # Wall BACK
            z_drone - z_min_box,  # Floor
            z_max_box - z_drone   # Roof
        )
        self.ocp.model.con_h_expr = h_expr_path

        self.ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ocp.constraints.uh = np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5])


        # --- TERMINAL CONSTRAINT ---
        h_expr_terminal = cs.vertcat(
            x_drone - x_min_box - alpha_pred,
            x_max_box - x_drone - alpha_pred,
            y_drone - y_min_box - alpha_pred,
            y_max_box - y_drone - alpha_pred,
            z_drone - z_min_box - alpha_pred,
            z_max_box - z_drone - alpha_pred
        )

        self.ocp.model.con_h_expr_e = h_expr_terminal
        self.ocp.constraints.lh_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ocp.constraints.uh_e = np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5])

                
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
        box_abs: np.ndarray
        #obs_x: float
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Performs a single MPC optimization step.
        Returns the state trajectory, input sequence, NN margin evaluated at
        the terminal solver iterate, and solver status. The margin is reported
        but is not an active terminal clearance constraint; after a failed
        solve, the terminal iterate may not be feasible.
        """
        self.ocp_solver.reset()

        # Fix the initial node to the current measured or estimated state.
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)


        # 2. Parameters vector
        p_val = np.hstack([box_abs, x_ref])

        # Limiti del box per la posizione
        lbx_dynamic = np.hstack([box_abs[0], box_abs[2], box_abs[4], np.full(self.model.nori, -np.pi), np.full(self.model.nv, -1e1)])
        ubx_dynamic = np.hstack([box_abs[1], box_abs[3], box_abs[5], np.full(self.model.nori, np.pi), np.full(self.model.nv, 1e1)])
        
        # 2. WARM-START AND PARAMETERS
        for i in range(self.N):
            if i == 0:
                self.ocp_solver.set(i, 'x', current_x)
            else:
                self.ocp_solver.set(i, 'x', self.x_guess[i])
                
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', p_val)


        # # Warm-start terminal stage
        lbx_dynamic_e = np.hstack([box_abs[0], box_abs[2], box_abs[4], np.full(self.model.nori, -np.pi), np.full(self.model.nv, -1e1)])
        ubx_dynamic_e = np.hstack([box_abs[1], box_abs[3], box_abs[5], np.full(self.model.nori, np.pi), np.full(self.model.nv, 1e1)])

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p_val)

                
        
        # Solver
        status = self.ocp_solver.solve()

        # --- BLOCCO ANALISI DI DEBUG DETTAGLIATA ---
        # x_terminal = self.ocp_solver.get(self.N, 'x')
        # x_norm = (x_terminal[2:6] - self.mean_X) / self.std_X
        # with torch.no_grad():
        #     alpha_pred_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()

        # # Calcoliamo le 4 distanze terminali rispetto ai muri attuali
        # dist_sx = x_terminal[0] - box_abs[0]
        # dist_dx = box_abs[1] - x_terminal[0]
        # dist_pav = x_terminal[1] - box_abs[2]
        # dist_sof = box_abs[3] - x_terminal[1]

        # print(f"\n--- DEBUG NODO N (Status {status}) ---")
        # print(f"Alpha Richiesto (Rete): {alpha_pred_val:.3f}m")
        # print(f"Distanze Disponibili: SX: {dist_sx:.3f} | DX: {dist_dx:.3f} | PAV: {dist_pav:.3f} | SOF: {dist_sof:.3f}")
        
        # # Vediamo quale vincolo è violato
        # violations = []
        # if dist_sx < alpha_pred_val: violations.append("MURO SX")
        # if dist_dx < alpha_pred_val: violations.append("MURO DX")
        # if dist_pav < alpha_pred_val: violations.append("PAVIMENTO")
        # if dist_sof < alpha_pred_val: violations.append("SOFFITTO")
        
        # if violations:
        #     print(f"❌ VINCOLI VIOLATI: {', '.join(violations)}")
        # else:
        #     print(f"✅ TUTTI I VINCOLI RISPETTATI")
        # # -------------------------------------------

        
        x_terminal = self.ocp_solver.get(self.N, 'x')
        x_norm = (x_terminal[3:12] - self.mean_X) / self.std_X  
        with torch.no_grad():
            alpha_pred_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()

        
        dist_x_min = x_terminal[0] - box_abs[0]
        dist_x_max = box_abs[1] - x_terminal[0]
        dist_y_min = x_terminal[1] - box_abs[2]
        dist_y_max = box_abs[3] - x_terminal[1]
        dist_z_min = x_terminal[2] - box_abs[4]
        dist_z_max = box_abs[5] - x_terminal[2]
        min_dist_to_wall = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)
        
        print(f" --> (Status {status}, alpha_pred {alpha_pred_val:.2f}, min_dist_muro {min_dist_to_wall:.2f})")

    
        # ------------------------------------------
        
        if status in [0, 2]: # Success of sub-optimal
            x_sol = np.empty((self.N + 1, self.model.nx))
            u_sol = np.empty((self.N, self.model.nu))
            
            for i in range(self.N):
                x_sol[i] = self.ocp_solver.get(i, 'x')
                u_sol[i] = self.ocp_solver.get(i, 'u')
            x_sol[self.N] = self.ocp_solver.get(self.N, 'x')
            

            # Warm-start update
            new_x_guess = np.vstack([x_sol[1:], x_sol[-1]])
            new_u_guess = np.vstack([u_sol[1:], u_sol[-1]])
            self.setGuess(new_x_guess, new_u_guess)
            
            return x_sol, u_sol, alpha_pred_val, status
        else:
            return None, None, alpha_pred_val, status
        