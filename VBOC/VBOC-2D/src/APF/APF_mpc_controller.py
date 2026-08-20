from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from src.MPC.mapping.mpc_abstract_obs import AbstractController
from src.MPC.mapping.learning import NeuralNetwork


class MpcController(AbstractController):
    """MPC controller using a Neural Network as a terminal constraint."""

    def __init__(self, model) -> None:
        super().__init__(model)
               
        # SOLVER COMPILATION
        gen_name = self.params.GEN_DIR + 'ocp_mpc_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name


        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)


    def solve_step(
        self,
        current_x: np.ndarray,
        x_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Performs a single MPC optimization step.
        Returns: (state_trajectory, input_trajectory, status)
        """
        self.ocp_solver.reset()

        # Initial state set to current sensor measurements
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)


        # 2. Parameter vector: [alpha_real_min(1), x_ref(6)]
        p_val = x_ref

        
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
            
            return x_sol, u_sol, status
        else:
            return None, None, status