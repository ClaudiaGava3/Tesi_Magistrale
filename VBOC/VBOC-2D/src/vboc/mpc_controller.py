from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from mpc_abstract import AbstractController
from learning import NeuralNetwork


class MpcController(AbstractController):
    """Controller MPC che usa una Rete Neurale come vincolo terminale"""

    def __init__(self, model) -> None:
        super().__init__(model)
        # Projection matrix used to enforce the velocity constraint orthogonal to d
        
        # 2. CARICAMENTO RETE NEURALE
        nn_filename = f"{self.params.NN_DIR}{self.params.robot_name}_{self.params.act}.pt"
        print(f"--- Caricamento Rete Neurale da {nn_filename} ---")
        
        checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
        self.mean_X = checkpoint['mean']
        self.std_X = checkpoint['std']
        
        # Inizializziamo la rete (Input 4D per il 2D: theta, vx, vz, wy)
        # NOTA: assicurati di usare la stessa activation del training (es. GELU)
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
        
        # 3. L4CASADI E VINCOLO TERMINALE
        # Nello stato dell'MPC [x, z, theta, vx, vz, wy], la rete usa gli indici [2, 3, 4, 5]
        theta_sym = self.model.x[2]
        vx_sym = self.model.x[3]
        vz_sym = self.model.x[4]
        wy_sym = self.model.x[5]
        x_nn_sym = cs.vertcat(theta_sym, vx_sym, vz_sym, wy_sym)
        
        # Normalizziamo gli input per la rete
        x_norm = (x_nn_sym - cs.DM(self.mean_X)) / cs.DM(self.std_X)
        
        # Creiamo la funzione L4CasADi
        self.l4c_model = l4c.L4CasADi(net, name="drone_viability_net")
        
        # Output della rete: lo spazio di frenata predetto (Alpha)
        alpha_pred = self.l4c_model(x_norm.T)
        
        # IL VINCOLO: Alpha Predetto - Alpha Reale <= 0
        # self.model.p[0] è l'alpha_real che abbiamo definito nei parametri
        self.ocp.model.con_h_expr_e = alpha_pred - self.model.p[0]
        self.ocp.constraints.uh_e = np.array([0.0])
        self.ocp.constraints.lh_e = np.array([-1e5]) # Limite inferiore largo
        
        # 4. COMPILAZIONE DEL SOLVER
        # Ora che abbiamo aggiunto il vincolo di L4CasADi, possiamo compilare!
        gen_name = self.params.GEN_DIR + 'ocp_mpc_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name

        # ---> AGGIUNGI QUESTE DUE RIGHE PER L4CASADI <---
        self.ocp.solver_options.model_external_shared_lib_dir = self.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.l4c_model.name

        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)


    def solve_step(
        self,
        current_x: np.ndarray,
        x_ref: np.ndarray,
        alpha_real: float
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Esegue un singolo step di ottimizzazione MPC.
        Restituisce: (traiettoria_stati, traiettoria_input, alpha_predetto, status)
        """
        self.ocp_solver.reset()

        # 1. Fissa lo stato iniziale ai valori correnti misurati dai sensori
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)

        # 2. Crea il vettore dei parametri per questo step: [alpha_real, x_ref(6)]
        p_val = np.hstack([alpha_real, x_ref])
        
        for i in range(self.N):
            # Warm-start primal variables and set the direction parameter
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', p_val)

        #     if i != 0:
        #         # Box constraint on the obstacle (upper bound encodes both min 
        #         # and max)
        #         # self.ocp_solver.constraints_set(
        #         #      i, "uh", 
        #         #      np.hstack([box_min_values, box_max_values])
        #         # )
        #         # Orientation bounded in [-π, π]; velocity loosely bounded for 
        #         # feasibility
        #         self.ocp_solver.constraints_set(
        #             i, 
        #             "lbx", 
        #             np.hstack([
        #                 np.full(self.model.nori, -np.pi), 
        #                 np.full(self.model.nv, -1e2),
        #                 # np.full(self.model.nbox, 0.1),
        #                 # np.array([0.0])    # Limite inferiore box
        #             ])
        #         )
        #         self.ocp_solver.constraints_set(
        #             i, 
        #             "ubx", 
        #             np.hstack([
        #                 np.full(self.model.nori, np.pi), 
        #                 np.full(self.model.nv, 1e2),
        #                 # np.full(self.model.nbox, 1.0),
        #                 # np.array([1e5])    # Limite superiore box
        #             ])
        #         )

        # # Warm-start terminal stage
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p_val)

        # # Shared bounds reused for stages N-1 and N
        # # zero_vel = np.zeros(self.model.nv)
        # # ori_0 = np.full(self.model.nori, 0.0) #cambiato vincolo a 0 da -np.pi
        # # box_min = np.full(self.model.nbox, 0.1)
        # # box_max = np.full(self.model.nbox, 1.0)

        # # Zero-velocity and angles terminal constraint at stage N-1
        # self.ocp_solver.constraints_set(
        #     self.N - 1, "lbx", np.hstack([
        #         np.full(self.model.nori, -np.pi), 
        #         np.full(self.model.nv, -1e2),
        #         # np.full(self.model.nbox, 0.1),
        #         # np.array([0.0])    # Limite inferiore box
        #     ])
        # )
        # self.ocp_solver.constraints_set(
        #     self.N - 1, "ubx", np.hstack([
        #         np.full(self.model.nori, np.pi), 
        #         np.full(self.model.nv, 1e2),
        #         # np.full(self.model.nbox, 1.0),
        #         # np.array([1e5])    # Limite superiore box
        #     ])
        # )

        # # Zero-velocity and angles terminal constraint at stage N
        # self.ocp_solver.constraints_set(
        #     self.N, "lbx", np.hstack([
        #         np.full(self.model.nori, -np.pi), 
        #         np.full(self.model.nv, -1e2),
        #         # np.full(self.model.nbox, 0.1),
        #         # np.array([0.0])    # Limite inferiore box
        #     ])
        # )
        # self.ocp_solver.constraints_set(
        #     self.N, "ubx", np.hstack([
        #         np.full(self.model.nori, np.pi), 
        #         np.full(self.model.nv, 1e2),
        #         # np.full(self.model.nbox, 1.0),
        #         # np.array([1e5])    # Limite superiore box
        #     ])
        # )

        # self.ocp_solver.constraints_set(
        #     self.N, "uh",
        #     np.hstack([box_min_values, box_max_values])
        # )

        # Build the projection matrix C = [0 | I - d·dᵀ] to constrain
        # the initial velocity to lie in the hyperplane orthogonal to d
        # d_arr = np.array([d.tolist()])
        # self.C[:, self.model.nq:] = (
        #      np.eye(self.model.nv) - np.matmul(d_arr.T, d_arr)
        # )
        # self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # Fix initial position and velocity; box dimensions remain free to be optimized

        # Ensure we pass 6 elements (pos + vel) to match idxbx_0
        # self.ocp_solver.constraints_set(0, "lbx", q_init)
        # self.ocp_solver.constraints_set(0, "ubx", q_init)

        # 4. Risolvi
        status = self.ocp_solver.solve()
        
        if status in [0, 2]: # Successo o sub-ottimo accettabile
            x_sol = np.empty((self.N + 1, self.model.nx))
            u_sol = np.empty((self.N, self.model.nu))
            
            for i in range(self.N):
                x_sol[i] = self.ocp_solver.get(i, 'x')
                u_sol[i] = self.ocp_solver.get(i, 'u')
            x_sol[self.N] = self.ocp_solver.get(self.N, 'x')
            
          # --- CALCOLO DI ALPHA CORRENTE ---
            # Prendiamo lo stato finale della previsione (nodo N)
            x_terminal = x_sol[self.N]
            # Estraiamo theta, vx, vz, wy (indici 2, 3, 4, 5)
            x_nn_input = x_terminal[2:6]
            # Normalizziamo
            x_norm = (x_nn_input - self.mean_X) / self.std_X

            # Interroghiamo la rete neurale (Python) per avere il valore da plottare
            with torch.no_grad():
                # self.l4c_model.model è la rete torch interna
                alpha_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()



            # Aggiorna il warm-start shiftandolo in avanti di 1 passo per la prossima iterazione
            new_x_guess = np.vstack([x_sol[1:], x_sol[-1]])
            new_u_guess = np.vstack([u_sol[1:], u_sol[-1]])
            self.setGuess(new_x_guess, new_u_guess)
            
            return x_sol, u_sol, alpha_val, status
        else:
            return None, None, 0.0, status