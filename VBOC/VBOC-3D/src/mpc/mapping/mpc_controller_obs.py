from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from mpc_abstract_obs import AbstractController
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
        

        # DISTANZA DINAMICA: Posizione Muro fissa (p[0]) - Posizione drone futura (x[0])
        #dist_dinamica = self.model.p[0] - self.model.x[0]

        # Estraiamo la posizione futura del drone e i limiti del box dai parametri
        x_drone = self.model.x[0]
        y_drone = self.model.x[1]
        z_drone = self.model.x[2]
        
        x_min_box = self.model.p[0]
        x_max_box = self.model.p[1]
        y_min_box = self.model.p[2]
        y_max_box = self.model.p[3]
        z_min_box = self.model.p[4]
        z_max_box = self.model.p[5]


        # --- PATH CONSTRAINT (6 LATI ASIMMETRICI) ---
        # Tutte e 6 le distanze dai muri devono essere >= 0
        h_expr_path = cs.vertcat(
            x_drone - x_min_box,  # Muro SX
            x_max_box - x_drone,  # Muro DX
            y_drone - y_min_box,  # Muro FRONT
            y_max_box - y_drone,  # Muro BACK
            z_drone - z_min_box,  # Pavimento
            z_max_box - z_drone   # Soffitto
        )
        self.ocp.model.con_h_expr = h_expr_path
        # Imponiamo che tutti e 6 i valori siano compresi tra 0 e infinito
        self.ocp.constraints.lh = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ocp.constraints.uh = np.array([1e5, 1e5, 1e5, 1e5, 1e5, 1e5])


        # --- VINCOLO TERMINALE ( 6 LATI ASIMMETRICI) ---
        # Tutte e 6 le distanze dai muri meno l'alpha predetto devono essere >= 0
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

        # if h_expr_terminal<0: print("⚠️ ATTENZIONE: VINCOLO TERMINALE CON VALORE NEGATIVO!")


        
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
        box_abs: np.ndarray # Array di 4 elementi [x_min, x_max, z_min, z_max]
        #obs_x: float
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Esegue un singolo step di ottimizzazione MPC.
        Restituisce: (traiettoria_stati, traiettoria_input, alpha_predetto, status)
        """
        self.ocp_solver.reset()

        # Stato iniziale ai valori correnti misurati dai sensori
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)


        # 2. Vettore dei parametri: [alpha_real_min(1), x_ref(6)]
        p_val = np.hstack([box_abs, x_ref])

        # Limiti del box per la posizione
        lbx_dynamic = np.hstack([box_abs[0], box_abs[2], box_abs[4], np.full(self.model.nori, -np.pi), np.full(self.model.nv, -1e1)])
        ubx_dynamic = np.hstack([box_abs[1], box_abs[3], box_abs[5], np.full(self.model.nori, np.pi), np.full(self.model.nv, 1e1)])
        
        # 2. WARM-START E PARAMETRI
        for i in range(self.N):
            if i == 0:
                self.ocp_solver.set(i, 'x', current_x)
            else:
                self.ocp_solver.set(i, 'x', self.x_guess[i])
                
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', p_val)

            # self.ocp_solver.constraints_set(i, "lbx", lbx_dynamic)
            # self.ocp_solver.constraints_set(i, "ubx", ubx_dynamic)


        # # Warm-start terminal stage
        lbx_dynamic_e = np.hstack([box_abs[0], box_abs[2], box_abs[4], np.full(self.model.nori, -np.pi), np.full(self.model.nv, -1e1)])
        ubx_dynamic_e = np.hstack([box_abs[1], box_abs[3], box_abs[5], np.full(self.model.nori, np.pi), np.full(self.model.nv, 1e1)])

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p_val)

        # self.ocp_solver.constraints_set(self.N, "lbx", lbx_dynamic_e)
        # self.ocp_solver.constraints_set(self.N, "ubx", ubx_dynamic_e)

        
        
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

        # Interroga la rete per restituire il valore corretto anche se fallisce
        x_terminal = self.ocp_solver.get(self.N, 'x')
        x_norm = (x_terminal[3:12] - self.mean_X) / self.std_X  
        with torch.no_grad():
            alpha_pred_val = self.l4c_model.model(torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)).item()

        # --- IL PRINT DI DEBUG CORRETTO VA QUI ---
        # Calcoliamo la distanza REALE AL NODO TERMINALE
        dist_x_min = x_terminal[0] - box_abs[0]
        dist_x_max = box_abs[1] - x_terminal[0]
        dist_y_min = x_terminal[1] - box_abs[2]
        dist_y_max = box_abs[3] - x_terminal[1]
        dist_z_min = x_terminal[2] - box_abs[4]
        dist_z_max = box_abs[5] - x_terminal[2]
        min_dist_to_wall = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)
        
        print(f" --> (Status {status}, alpha_pred {alpha_pred_val:.2f}, min_dist_muro {min_dist_to_wall:.2f})")

    
        # ------------------------------------------
        
        if status in [0, 2]: # Successo o sub-ottimo accettabile
            x_sol = np.empty((self.N + 1, self.model.nx))
            u_sol = np.empty((self.N, self.model.nu))
            
            for i in range(self.N):
                x_sol[i] = self.ocp_solver.get(i, 'x')
                u_sol[i] = self.ocp_solver.get(i, 'u')
            x_sol[self.N] = self.ocp_solver.get(self.N, 'x')
            


            # Aggiorno warm-start
            new_x_guess = np.vstack([x_sol[1:], x_sol[-1]])
            new_u_guess = np.vstack([u_sol[1:], u_sol[-1]])
            self.setGuess(new_x_guess, new_u_guess)
            
            return x_sol, u_sol, alpha_pred_val, status
        else:
            return None, None, alpha_pred_val, status
        