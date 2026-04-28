from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from mpc.mpc_abstract import AbstractController
from learning import NeuralNetwork


class MpcController(AbstractController):
    """Controller MPC che usa una Rete Neurale come vincolo terminale"""

    def __init__(self, model) -> None:
        super().__init__(model)
        
        # 2. CARICAMENTO RETE NEURALE
        nn_filename = f"{self.params.NN_DIR}{self.params.robot_name}_{self.params.act}.pt"
        print(f"--- Caricamento Rete Neurale da {nn_filename} ---")
        
        checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
        self.mean_X = checkpoint['mean']
        self.std_X = checkpoint['std']
        
        # Inizializzazione rete neurale (Input 4D per il 2D: theta, vx, vz, wy)
        # NOTA: assicurati di usare la stessa activation del training (es. GELU)
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
        
        # 3. L4CASADI E VINCOLO TERMINALE
        # Nello stato 3D [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r], la rete usa gli indici da 3 a 12 esclusi
        x_nn_sym = self.model.x[3:12]
        
        # Normalizzazione input per la rete
        x_norm = (x_nn_sym - cs.DM(self.mean_X)) / cs.DM(self.std_X)
        
        # Funzione L4CasADi
        self.l4c_model = l4c.L4CasADi(net, name="drone_viability_net")
        
        # Output della rete: spazio di frenata predetto (Alpha)
        alpha_pred = self.l4c_model(x_norm.T)
        
        # VINCOLO: Alpha Predetto - Alpha Reale <= 0
        self.ocp.model.con_h_expr_e = alpha_pred - self.model.p[0]
        self.ocp.constraints.uh_e = np.array([0.0])
        self.ocp.constraints.lh_e = np.array([-1e5]) # Limite inferiore largo
        
        # COMPILAZIONE DEL SOLVER
        gen_name = self.params.GEN_DIR + 'ocp_mpc_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name

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

        # Stato iniziale ai valori correnti misurati dai sensori
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)

        # Vettore dei parametri per questo step: [alpha_real, x_ref(6)]
        p_val = np.hstack([alpha_real, x_ref])
        
        for i in range(self.N):
            # Warm-start
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

        # Solver
        status = self.ocp_solver.solve()
        
        if status in [0, 2]: # Successo o sub-ottimo accettabile
            x_sol = np.empty((self.N + 1, self.model.nx))
            u_sol = np.empty((self.N, self.model.nu))
            
            for i in range(self.N):
                x_sol[i] = self.ocp_solver.get(i, 'x')
                u_sol[i] = self.ocp_solver.get(i, 'u')
            x_sol[self.N] = self.ocp_solver.get(self.N, 'x')
            
          # --- CALCOLO DI ALPHA CORRENTE ---
            # Considero stato finale della previsione (nodo N)
            x_terminal = x_sol[self.N]
            # Estraggo angoli e velocità (indici da 3 a 11 compresi)
            x_nn_input = x_terminal[3:12]
            # Normalizzo
            x_norm = (x_nn_input - self.mean_X) / self.std_X

            # Interrogo NN per avere alpha da plottare
            with torch.no_grad():
                alpha_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()



            # Aggiorno warm-start
            new_x_guess = np.vstack([x_sol[1:], x_sol[-1]])
            new_u_guess = np.vstack([u_sol[1:], u_sol[-1]])
            self.setGuess(new_x_guess, new_u_guess)
            
            return x_sol, u_sol, alpha_val, status
        else:
            return None, None, 0.0, status