
# ==========================================================================================================================================================
"""Solve feedback MPC steps for the APF-guided planar drone simulation.

At each step, the controller fixes the measured/simulated current state,
tracks the reference produced by the APF layer, and returns the optimized
state trajectory, motor-input sequence, and solver status. Successful
solutions are shifted forward to warm-start the next feedback step.
"""
# ==========================================================================================================================================================

from __future__ import annotations
import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
from acados_template import AcadosOcpSolver

from src.MPC.mapping.mpc_abstract_obs import AbstractController
from src.MPC.mapping.learning import NeuralNetwork


class MpcController(AbstractController):
    """Feedback MPC controller that tracks an APF-generated state reference."""

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
        """Solve one MPC step from the current state toward ``x_ref``.

        Returns the predicted state trajectory, optimized motor-input
        sequence, and solver status. The trajectories are ``None`` if the
        solver does not return an accepted solution.
        """
        self.ocp_solver.reset()

        # Fix the initial shooting node to the current measured/simulated state.
        self.ocp_solver.constraints_set(0, "lbx", current_x)
        self.ocp_solver.constraints_set(0, "ubx", current_x)


        # The six-element target state is the OCP parameter at each stage.
        p_val = x_ref

        
        # Warm-start primal variables and set the target parameter per stage.
        for i in range(self.N):
            if i == 0:
                self.ocp_solver.set(i, 'x', current_x)
            else:
                self.ocp_solver.set(i, 'x', self.x_guess[i])
                
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', p_val)



        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', p_val)

        
        
        # Solve the finite-horizon feedback OCP.
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