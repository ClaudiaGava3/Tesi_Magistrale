
# =========================================================================================================================================================
"""Solve the asymmetric-parallelepiped RBOC OCP for neural-network training data.

The controller fixes each sampled initial drone state and four box-side
proportions. It solves for their common scale, imposes zero pitch and velocity
at the final shooting nodes, and extends the horizon after unsuccessful
attempts.
"""
# ==========================================================================================================================================================

from __future__ import annotations
import numpy as np
from casadi import dot, horzcat
from .abstract_cubev2 import AbstractController


class ViabilityController(AbstractController):
    """Solve the asymmetric-box RBOC problem and retry with a longer horizon.

    Parameters
    ----------
    model : object
        Robot model exposing state/input dimensions and constraint metadata.
    """

    def __init__(self, model) -> None:
        super().__init__(model)
        # Legacy projection matrix; the current solve methods do not use it.
        self.C = np.zeros((self.model.nv, self.model.nx))

    def solve(
        self,
        q_init: np.ndarray,
        ref_box: np.ndarray,
    ) -> int:
        """Warm-start and solve one OCP for a sampled initial drone state.

        The initial kinematic state and four box-side proportions are fixed.
        Zero pitch and velocity are imposed at the final shooting nodes.

        Parameters
        ----------
        q_init : np.ndarray
            Initial state ``[x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz]``, shape (12,).
        ref_box : np.ndarray
            Six initial box-side proportions, shape (6,).

        Returns
        -------
        status : int
            Solver status code (0 = success, 2 = acceptable solution).
        """
        self.ocp_solver.reset()
        
        for i in range(self.N):
            # Warm-start state/input variables and set the box parameter.
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', ref_box)

            if i != 0:
                
                self.ocp_solver.constraints_set(
                     i, 
                     "lbx", 
                     np.hstack([
                          np.full(self.model.nori, -np.pi), 
                          np.full(self.model.nv, -1e2),
                          np.full(self.model.nbox, 0.01),
                          np.array([0.0])
                     ])
                )
                self.ocp_solver.constraints_set(
                     i, 
                     "ubx", 
                     np.hstack([
                          np.full(self.model.nori, np.pi), 
                          np.full(self.model.nv, 1e2),
                          np.full(self.model.nbox, 1.0),
                          np.array([1e5])
                    ])
                )

        # Warm-start terminal stage
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', ref_box)

        # Shared bounds reused for stages N-1 and N
        zero_vel = np.zeros(self.model.nv)
        ori_0 = np.full(self.model.nori, 0.0)
        box_min = np.full(self.model.nbox, 0.01)
        box_max = np.full(self.model.nbox, 1.0)


        # Zero-velocity and angles terminal constraint at stage N-1
        self.ocp_solver.constraints_set(
            self.N - 1, "lbx", np.hstack([ori_0, zero_vel, box_min,np.array([0.0])])
        )
        self.ocp_solver.constraints_set(
            self.N - 1, "ubx", np.hstack([ori_0, zero_vel, box_max,np.array([1e5])])
        )

        # Zero-velocity terminal constraint at final stage N
        self.ocp_solver.constraints_set(
            self.N, "lbx", np.hstack([ori_0, zero_vel, box_min,np.array([0.0])])
        )
        self.ocp_solver.constraints_set(
            self.N, "ubx", np.hstack([ori_0, zero_vel, box_max, np.array([1e5])])
        )
       
        # Fix the six initial kinematic components and four box proportions.
        state_box_init = np.hstack([q_init, ref_box])
        self.ocp_solver.constraints_set(0, "lbx", state_box_init)
        self.ocp_solver.constraints_set(0, "ubx", state_box_init)

        return self.ocp_solver.solve()
    
        
    def solve_rboc(
        self,
        q_init: np.ndarray,
        ref_box: np.ndarray,
        N_start: int,
        n: int = 1,
        repeat: int = 10
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, int]:
        """Retry the OCP with a longer horizon after unsuccessful solves.

        Each failed attempt extends the warm-start by ``n`` intervals and
        increases the solver horizon. The retry limit is ``self.rboc_repeat``.

        Parameters
        ----------
        q_init : np.ndarray
            Initial state ``[x, y, z, phi, theta, psi, vx, vy, vz, wx, wy, wz]``, shape (12,).
        ref_box : np.ndarray
            Six reference box-side proportions passed to each OCP solve.
        N_start : int
            Initial prediction horizon length.
        n : int, optional
            Number of steps by which the horizon is incremented each iteration.
            Default is 1.
        repeat : int, optional
            Documented retry argument; the loop uses ``self.rboc_repeat``.

        Returns
        -------
        x_sol : np.ndarray or None
            State trajectory; returned only when the final status is 0.
        u_sol : np.ndarray or None
            Input trajectory; returned only when the final status is 0.
        N : int or None
            Final horizon length; returned only when the final status is 0.
        status : int
            Final OCP status. Status 2 stops retries but returns no trajectories
            because this wrapper returns data only for status 0.
        """
        N = N_start
        gamma = float('inf')   # Legacy variable; unused in the current retry logic.
        x_sol, u_sol = None, None

        repeat = self.rboc_repeat
        for r in range(repeat):
            status = self.solve(q_init, ref_box)

            if status == 0 or status == 2:
                # Read the initial-stage state for diagnostic scale reporting.
                x0 = self.ocp_solver.get(0, "x")
                
                gamma_new = x0[18] # State element 18 stores the common scale.


                print(f"Iteration {r}: SUCCESS! Scale = {gamma_new:.4f}, Horizon N = {N}")

                

                # Read the optimized state and input trajectories from Acados.
                x_sol = np.empty((N, self.model.nx))
                u_sol = np.empty((N, self.model.nu)) 
                for i in range(N):
                    x_sol[i] = self.ocp_solver.get(i, 'x')
                    u_sol[i] = self.ocp_solver.get(i, 'u')

                break


            else:     
                # If the solver fails, we don't give up!
                print(f"Iteration {r}: Solver failed with N={N}. Increasing horizon...")
                
                # Extend the warm-start trajectories before retrying.
                new_x_guess = np.empty((N + n, self.model.nx))
                new_u_guess = np.empty((N + n, self.model.nu)) 
                
                # Copy the existing guesses for the current N intervals.
                new_x_guess[:N] = self.x_guess
                new_u_guess[:N] = self.u_guess
                
                # Repeat the last guess over the newly added intervals.
                new_x_guess[N:] = self.x_guess[-1]
                
                new_u_guess[N:] = self.u_guess[-1] 
                
                # Store the extended warm start.
                self.setGuess(new_x_guess, new_u_guess)

                # Increase the horizon and update the solver discretization.
                N += n
                self.resetHorizon(N)
            
        if status == 0:
            return x_sol, u_sol, N, status
        else:
            return None, None, None, status