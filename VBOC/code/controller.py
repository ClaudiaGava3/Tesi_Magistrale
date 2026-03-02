import numpy as np
from casadi import dot
from .abstract import AbstractController


class ViabilityController(AbstractController):
    def __init__(self, model, obstacles=None, capsule=None, capsule_pairs=None):
        super().__init__(model, obstacles, capsule, capsule_pairs)
        self.C = np.zeros((self.model.nv, self.model.nx))

    def addCost(self):
        # Maximize initial velocity
        self.ocp.cost.cost_type_0 = 'EXTERNAL'
        self.ocp.model.cost_expr_ext_cost_0 = dot(self.model.p, self.model.x[self.model.nq:])
        self.ocp.parameter_values = np.zeros(self.model.nv)

    def addConstraint(self):
        q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        self.ocp.constraints.C = np.zeros((self.model.nv, self.model.nx))
        self.ocp.constraints.D = np.zeros((self.model.nv, self.model.nu))
        self.ocp.constraints.lg = np.zeros((self.model.nv,))
        self.ocp.constraints.ug = np.zeros((self.model.nv,))
    
    def solve(self, q_init, d):
        self.ocp_solver.reset()
        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.set(i, 'p', d)
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(self.N, 'p', d)

        # Set the initial constraint
        d_arr = np.array([d.tolist()])
        self.C[:, self.model.nq:] = np.eye(self.model.nv) - np.matmul(d_arr.T, d_arr)
        self.ocp_solver.constraints_set(0, "C", self.C, api='new')

        # Set initial bounds -> x0_pos = q_init, x0_vel free; (final bounds already set)
        q_init_lb = np.hstack([q_init, self.model.x_min[self.model.nq:]])
        q_init_ub = np.hstack([q_init, self.model.x_max[self.model.nq:]])
        self.ocp_solver.constraints_set(0, "lbx", q_init_lb)
        self.ocp_solver.constraints_set(0, "ubx", q_init_ub)

        # Solve the OCP
        return self.ocp_solver.solve()
    
    def solveVBOC(self, q, d, N_start, n=1, repeat=10):
        N = N_start
        gamma = 0
        x_sol, u_sol = None, None
        for _ in range(repeat):
            # Solve the OCP
            status = self.solve(q, d)
            if status == 0 or status == 2:
                # Compare the current cost with the previous one:
                x0 = self.ocp_solver.get(0, "x")
                gamma_new = np.linalg.norm(x0[self.model.nq:])

                if gamma_new < gamma + self.tol and status == 0:
                    break
                gamma = gamma_new

                # Rollout the solution
                x_sol = np.empty((N + n, self.model.nx))
                u_sol = np.empty((N + n, self.model.nu))    # last control is not used
                for i in range(N):
                    x_sol[i] = self.ocp_solver.get(i, 'x')
                    u_sol[i] = self.ocp_solver.get(i, 'u')
                x_sol[N:] = self.ocp_solver.get(N, 'x')
                u_sol[N:] = np.zeros((n, self.model.nu))

                # Reset the initial guess with the previous solution
                self.setGuess(x_sol, u_sol)
                # Increase the horizon
                N += n
                self.resetHorizon(N)
            else:
                return None, status
        if status == 0:
            return x_sol, status
        else:
            return None, status


class SafeBackupController(AbstractController):
    """ Equilibrium condition as terminal constraint """
    def __init__(self, model, obstacles=None, capsule=None, capsule_pairs=None):
        super().__init__(model, obstacles, capsule, capsule_pairs)

    def addCost(self):
        self.ocp.cost.cost_type = 'LINEAR_LS'        
        self.ocp.cost.cost_type_e = 'LINEAR_LS'
        self.ocp.parameter_values = np.zeros(self.model.nv)

    def addConstraint(self):
        x_min_e = np.hstack((self.model.x_min[:self.model.nq], np.zeros(self.model.nv)))
        x_max_e = np.hstack((self.model.x_max[:self.model.nq], np.zeros(self.model.nv)))

        self.ocp.constraints.lbx_e = x_min_e
        self.ocp.constraints.ubx_e = x_max_e
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])

        self.ocp_solver.set(self.N, 'x', self.x_guess[self.N])

        # Solve the OCP
        status = self.ocp_solver.solve()

        if status == 0:
            x_sol = np.empty((self.N + 1, self.model.nx))
            u_sol = np.empty((self.N, self.model.nu))    
            for i in range(self.N):
                x_sol[i] = self.ocp_solver.get(i, 'x')
                u_sol[i] = self.ocp_solver.get(i, 'u')
            x_sol[-1] = self.ocp_solver.get(self.N, 'x')
            return x_sol, u_sol
        else:
            print(f'Solver failed with status {status}')
            print(self.ocp_solver.get_stats('time_tot'))
            return None, None