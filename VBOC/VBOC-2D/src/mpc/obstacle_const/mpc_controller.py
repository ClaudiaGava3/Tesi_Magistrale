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
        
        # 2. CARICAMENTO RETE NEURALE
        nn_filename = f"{self.params.NN_DIR}{self.params.robot_name}_{self.params.act}.pt"
        print(f"--- Caricamento Rete Neurale da {nn_filename} ---")
        
        checkpoint = torch.load(nn_filename, map_location=torch.device('cpu'), weights_only=False)
        self.mean_X = checkpoint['mean']
        self.std_X = checkpoint['std']
        
        # Inizializzazione rete neurale (Input 4D per il 2D: theta, vx, vz, wy)
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
        
        # Normalizzazione input per la rete
        x_norm = (x_nn_sym - cs.DM(self.mean_X)) / cs.DM(self.std_X)
        
        # Funzione L4CasADi
        self.l4c_model = l4c.L4CasADi(net, name="drone_viability_net")
        
        # Output della rete: spazio di frenata predetto (Alpha)
        alpha_pred = self.l4c_model(x_norm.T)
     
        
        
        # --- 2. VINCOLO TERMINALE ---
        # La distanza dal muro (p[0] - x[0]) deve essere maggiore o uguale ad alpha_pred
        self.ocp.model.con_h_expr_e = self.model.p[0] - self.model.x[0] - alpha_pred
        self.ocp.constraints.lh_e = np.array([0.0])   # Deve essere >= 0
        self.ocp.constraints.uh_e = np.array([1e5])
        
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
        obs_x: float
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
        p_val = np.hstack([obs_x, x_ref])
        
        for i in range(self.N):
            # Warm-start
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', p_val)

              

        # # Warm-start terminal stage
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p_val)

        
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
            # Estraggo theta, vx, vz, wy (indici 2, 3, 4, 5)
            x_nn_input = x_terminal[2:6]
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