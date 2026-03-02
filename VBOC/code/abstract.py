import re
import numpy as np
import casadi as cs
from copy import deepcopy
from casadi import MX, vertcat, dot, Function
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


class AdamModel:
    def __init__(self, params, n_dofs=False):
        self.params = params
        robot = URDF.from_xml_file(params.robot_urdf)
        try:
            n_dofs = n_dofs if n_dofs else len(robot.joints)
            if n_dofs > len(robot.joints) or n_dofs < 1:
                raise ValueError
        except ValueError:
            print(f'\nInvalid number of degrees of freedom! Must be > 1 and <= {len(robot.joints)}\n')
            exit()
        robot_joints = robot.joints[1:n_dofs + 1] if params.urdf_name == 'z1' else robot.joints[:n_dofs]
        joint_names = [joint.name for joint in robot_joints]
        self.kin_dyn = KinDynComputations(params.robot_urdf, joint_names, robot.get_root())
        self.kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
        self.H_b = np.eye(4)                                            # Base roto-translation matrix
        self.mass = self.kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = self.kin_dyn.bias_force_fun()                            # Nonlinear effects 
        self.fk = self.kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics
        nq = len(joint_names)

        self.amodel = AcadosModel()
        self.amodel.name = params.urdf_name
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        self.p = MX.sym("p", nq)
        # Double-integrator 
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.disc_dyn_expr = self.f_disc
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = nq
        self.nv = nq

        # Real dynamics
        self.tau = self.mass(self.H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias(self.H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:] 
        
        # EE position (global frame)
        T_ee = self.fk(np.eye(4), self.x[:nq])
        self.t_loc = np.array([0.035, 0., 0.])
        self.t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ self.t_loc
        self.ee_fun = Function('ee_fun', [self.x], [self.t_glob])

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints])
        if params.urdf_name == 'z1':
            joint_effort = np.array([2., 23., 10., 4., 4., 4.])
            joint_effort = joint_effort[:nq]
        else:
            joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 


        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])
        self.eps = params.state_tol
    
    def jointToEE(self, x):
        return np.array(self.ee_fun(x))

    # def checkPositionBounds(self, q):
    #     return np.logical_or(np.any(q < self.x_min[:self.nq] + self.eps), np.any(q > self.x_max[:self.nq] - self.eps))

    # def checkVelocityBounds(self, v):
    #     return np.logical_or(np.any(v < self.x_min[self.nq:] + self.eps), np.any(v > self.x_max[self.nq:] - self.eps))

    # def checkStateBounds(self, x):
    #     return np.logical_or(np.any(x < self.x_min + self.eps), np.any(x > self.x_max - self.eps))


    def casadi_segment_dist(self,A_s,B_s,C_s,D_s):
        # ab_a = cs.MX.sym('ab_a',3,1)
        # ab_b = cs.MX.sym('ab_b',3,1)
        # cd_c = cs.MX.sym('cd_a',3,1)
        # cd_d = cs.MX.sym('cd_b',3,1)

        R = cs.sum1((B_s-A_s)*(D_s-C_s))
        S1 = cs.sum1((B_s-A_s)*(C_s-A_s))
        D1 = cs.sum1((B_s-A_s)**2)
        S2 = cs.sum1((D_s-C_s)*(C_s-A_s))
        D2 = cs.sum1((D_s-C_s)**2)

        t = (S1*D2 - S2*R)/(D1*D2 - (R**2+1e-5))
        t = cs.fmax(cs.fmin(t,1),0)
        #u = -(S2*D1 - S1*R)/(D1*D2 - R**2)
        u = (t*R - S2)/D2
        u = cs.fmax(cs.fmin(u,1),0)

        t = (u*R + S1) / D1
        t = cs.fmax(cs.fmin(t,1),0)

        constr_expr = cs.sum1(((B_s-A_s)*t - (D_s-C_s)*u - (C_s-A_s))**2)

        return constr_expr
    
    def np_segment_dist(self,A_s,B_s,C_s,D_s):
        R = np.sum((B_s-A_s)*(D_s-C_s),axis=1)
        S1 = np.sum((B_s-A_s)*(C_s-A_s),axis=1)
        D1 = np.sum(((B_s-A_s)**2),axis=1)
        S2 = np.sum((D_s-C_s)*(C_s-A_s),axis=1)
        D2 = np.sum(((D_s-C_s)**2),axis=1)

        t = (np.multiply(S1,D2) - np.multiply(S2,R))/(np.multiply(D1,D2) - (R**2+1e-5))
        t = np.fmax(np.fmin(t,1),0)
        #u = -(S2*D1 - S1*R)/(D1*D2 - R**2)
        u = (t*R - S2)/D2
        u = np.fmax(np.fmin(u,1),0)

        t = (u*R + S1) / D1
        t = np.fmax(np.fmin(t,1),0)

        return np.sum(((B_s-A_s)*t[:,np.newaxis] - (D_s-C_s)*u[:,np.newaxis] - (C_s-A_s))**2,axis=1)
    
    def ball_segment_dist(self,A_s,B_s,capsule_length,obs_pos):
        obst_pos = cs.MX(obs_pos)
        t = cs.fmin(cs.fmax(cs.dot((obst_pos-A_s),(B_s-A_s)) / (capsule_length**2),0),1)
        d = cs.sum1((obst_pos-(A_s+(B_s-A_s)*t))**2) 
        return d
    
    def np_ball_segment_dist(self,A_s,B_s,capsule_length,obs_pos):
        t = np.fmin(np.fmax(np.dot((obs_pos-A_s),(B_s-A_s)) / (capsule_length**2),0),1)
        d = np.sum((obs_pos-(A_s+(B_s-A_s)*t))**2) 
        return d
    


class AbstractController:
    def __init__(self, model, obstacles=None, capsules=None, capsule_pairs=None):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.params = model.params
        self.model = model
        self.obstacles = obstacles  
        self.capsules = capsules
        self.capsule_pairs = capsule_pairs

        self.N = self.params.N
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.N * self.params.dt
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.addCost()

        # Capsules end-points forward kinematics
        n_cap=0
        for capsule in self.capsules:
            capsule['index']=n_cap
            if capsule['type'] == 'moving':
                rot_mat=np.eye(4)
                if capsule['rotation_offset'] != None:
                    th_off=capsule['rotation_offset']
                    rot_mat_x = np.array([[1,0,0,0],
                                          [0,np.cos(th_off[0]),-np.sin(th_off[0]),0],
                                          [0,np.sin(th_off[0]),np.cos(th_off[0]),0],
                                          [0,0,0,1]])
                    
                    rot_mat_y = np.array([[np.cos(th_off[1]),0,np.sin(th_off[1]),0],
                                          [0,1,0,0],
                                          [-np.sin(th_off[1]),0,np.cos(th_off[1]),0],
                                          [0,0,0,1]])
                    
                    rot_mat_z = np.array([[np.cos(th_off[2]),-np.sin(th_off[2]),0,0],
                                          [np.sin(th_off[2]), np.cos(th_off[2]),0,0],
                                          [0,0,1,0],
                                          [0,0,0,1]])
                    rot_mat = rot_mat_x@rot_mat_y@rot_mat_z
                if capsule['spatial_offset'] != None:
                    prism_mat = np.array([[1,0,0,capsule['spatial_offset'][0]],
                                            [0,1,0,capsule['spatial_offset'][1]],
                                            [0,0,1,capsule['spatial_offset'][2]],
                                            [0,0,0,1]])
                    rot_mat = prism_mat@rot_mat  
                fk_capsule_points = self.model.kin_dyn.forward_kinematics_fun(capsule['link_name'])   
                T_capsule_points = fk_capsule_points(np.eye(4), self.model.x[:self.model.nq])@rot_mat
                capsule['end_points_fk'] = deepcopy([(T_capsule_points @ capsule['end_points'][0])[:3],
                                                     (T_capsule_points @ capsule['end_points'][1])[:3]])
                capsule['end_points_T_fun'] = deepcopy(cs.Function(f'fun_T_{n_cap}',[self.model.x],[T_capsule_points]))
                capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.model.x],[(T_capsule_points @ capsule['end_points'][0])[:3],
                                                                                                      (T_capsule_points @ capsule['end_points'][1])[:3]]))
            n_cap += 1

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Nonlinear constraint 
        self.nl_con_0, self.nl_lb_0, self.nl_ub_0 = [], [], []
        self.nl_con, self.nl_lb, self.nl_ub = [], [], []
        self.nl_con_e, self.nl_lb_e, self.nl_ub_e = [], [], []
        
        # --> dynamics (only on running nodes)
        self.nl_con_0.append(self.model.tau)
        self.nl_lb_0.append(self.model.tau_min)
        self.nl_ub_0.append(self.model.tau_max)
        
        self.nl_con.append(self.model.tau)
        self.nl_lb.append(self.model.tau_min)
        self.nl_ub.append(self.model.tau_max)

        for pair in self.capsule_pairs:
            if pair['type'] == 0:
                self.nl_con_0.append(self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']))
                self.nl_con.append(self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']))
                self.nl_con_e.append(self.model.casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']))

                self.nl_lb_0.append((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
                self.nl_ub_0.append(1e6)
                self.nl_lb.append((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
                self.nl_ub.append(1e6)
                self.nl_lb_e.append((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
                self.nl_ub_e.append(1e6)

            elif pair['type'] == 1:
                self.nl_con_0.append(self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']))
                self.nl_con.append(self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']))
                self.nl_con_e.append(self.model.ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']))

                self.nl_lb_0.append((pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2)
                self.nl_ub_0.append(1e6)
                self.nl_lb.append((pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2)
                self.nl_ub.append(1e6)
                self.nl_lb_e.append((pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2)
                self.nl_ub_e.append(1e6)     

            elif pair['type'] == 2:
                for point in pair['elements'][0]['end_points_fk']:
                    self.nl_con_0.append(point[2])
                    self.nl_con.append(point[2])
                    self.nl_con_e.append(point[2])

                    self.nl_lb_0.append(pair['elements'][1]['bounds'][0])
                    self.nl_ub_0.append(pair['elements'][1]['bounds'][1])
                    self.nl_lb.append(pair['elements'][1]['bounds'][0])
                    self.nl_ub.append(pair['elements'][1]['bounds'][1])
                    self.nl_lb_e.append(pair['elements'][1]['bounds'][0])
                    self.nl_ub_e.append(pair['elements'][1]['bounds'][1])

        # # --> collision (both on running and terminal nodes)
        # if obstacles is not None and self.params.obs_flag:
        #     # Collision avoidance with two obstacles
        #     t_glob = self.model.t_glob
        #     for obs in self.obstacles:
        #         if obs['name'] == 'floor':
        #             self.nl_con_0.append(t_glob[2])
        #             self.nl_con.append(t_glob[2])
        #             self.nl_con_e.append(t_glob[2])

        #             self.nl_lb_0.append(obs['bounds'][0])
        #             self.nl_ub_0.append(obs['bounds'][1])
        #             self.nl_lb.append(obs['bounds'][0])
        #             self.nl_ub.append(obs['bounds'][1])
        #             self.nl_lb_e.append(obs['bounds'][0])
        #             self.nl_ub_e.append(obs['bounds'][1])
        #         elif obs['name'] == 'ball':
        #             dist_b = (t_glob - obs['position']).T @ (t_glob - obs['position'])
        #             self.nl_con_0.append(dist_b)
        #             self.nl_con.append(dist_b)
        #             self.nl_con_e.append(dist_b)

        #             self.nl_lb_0.append(obs['bounds'][0])
        #             self.nl_ub_0.append(obs['bounds'][1])
        #             self.nl_lb.append(obs['bounds'][0])
        #             self.nl_ub.append(obs['bounds'][1])
        #             self.nl_lb_e.append(obs['bounds'][0])
        #             self.nl_ub_e.append(obs['bounds'][1])

        # Additional constraints
        self.addConstraint()
        
        self.model.amodel.con_h_expr_0 = vertcat(*self.nl_con_0)   
        self.model.amodel.con_h_expr = vertcat(*self.nl_con)

        self.ocp.constraints.lh_0 = np.hstack(self.nl_lb_0)
        self.ocp.constraints.uh_0 = np.hstack(self.nl_ub_0)
        self.ocp.constraints.lh = np.hstack(self.nl_lb)
        self.ocp.constraints.uh = np.hstack(self.nl_ub)

        if len(self.nl_con_e) > 0:
            self.model.amodel.con_h_expr_e = vertcat(*self.nl_con_e)
            self.ocp.constraints.lh_e = np.array(self.nl_lb_e)
            self.ocp.constraints.uh_e = np.array(self.nl_ub_e)

        # Solver options
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.hessian_approx = "EXACT"
        self.ocp.solver_options.exact_hess_constr = 0
        self.ocp.solver_options.exact_hess_dyn = 0
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.alpha_reduction = self.params.alpha_reduction
        self.ocp.solver_options.alpha_min = self.params.alpha_min
        self.ocp.solver_options.levenberg_marquardt = self.params.levenberg_marquardt
        self.ocp.solver_options.ext_fun_compile_flags = '-O3'

        # Generate OCP solver
        gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', build=self.params.build)

        # Storage
        self.x_guess = np.zeros((self.N, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.tol = self.params.cost_tol

    def addCost(self):
        pass

    def addConstraint(self):
        pass

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def resetHorizon(self, N):
        self.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)

    # def checkCollision(self, x):
    #     if self.obstacles is not None and self.params.obs_flag:
    #         t_glob = self.model.jointToEE(x) 
    #         for obs in self.obstacles:
    #             if obs['name'] == 'floor':
    #                 if t_glob[2] < obs['bounds'][0]:
    #                     return False
    #             elif obs['name'] == 'ball':
    #                 dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2) 
    #                 if dist_b < obs['bounds'][0]:
    #                     return False
    #     return True
    

    def checkCollision(self, x):
        if self.capsule_pairs is None:
            if self.obstacles is not None and self.params.obs_flag:
                t_glob = self.model.jointToEE(x) 
                for obs in self.obstacles:
                    if obs['name'] == 'floor':
                        if t_glob[2] + self.params.tol_obs < obs['bounds'][0]:
                            return False
                    elif obs['name'] == 'ball':
                        dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2)
                        if dist_b + self.params.tol_obs < obs['bounds'][0]:
                            return False
            return True
        else:
            capsules_pos = []
            for capsule in self.capsules:
                if capsule['type'] == 'moving':
                    capsules_pos.append(np.array([capsule['end_points_fk_fun'](x[i] if len(x.shape)>1 else x ) for i in range(x.shape[0] if len(x.shape)>1 else 1)]))
                elif capsule['type'] == 'fixed':
                    capsules_pos.append(np.array(capsule['end_points']).reshape(1,2,3,1))
            for pair in self.capsule_pairs:
                if pair['type'] == 0:
                    # A_s=capsules_pos[pair['elements'][0]['index']][:,0]
                    # B_s =capsules_pos[pair['elements'][0]['index']][:,1]
                    # C_s=capsules_pos[pair['elements'][1]['index']][:,0]
                    # D_s =capsules_pos[pair['elements'][1]['index']][:,1]
                    # dists = np.array([self.model.casadi_segment_dist(A_s[i],B_s[i],C_s[i],D_s[i]) for i in range(A_s.shape[0] if len(A_s.shape)>1 else 1)]) 
                    # if not(dists >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                    #     return False
                    if not(self.model.np_segment_dist(capsules_pos[pair['elements'][0]['index']][:,0],capsules_pos[pair['elements'][0]['index']][:,1],
                        capsules_pos[pair['elements'][1]['index']][:,0],capsules_pos[pair['elements'][1]['index']][:,1]) >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                        return False
                elif pair['type'] == 1:
                    A_s = capsules_pos[pair['elements'][0]['index']][:,0]
                    B_s = capsules_pos[pair['elements'][0]['index']][:,1]
                    dists = np.array([self.model.np_ball_segment_dist(A_s[i].flatten(),B_s[i].flatten(),pair['elements'][0]['length'],pair['elements'][1]['position']) for i in range(A_s.shape[0] if len(A_s.shape)>1 else 1)]) 
                    if not(dists >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                        return False
                elif pair['type'] == 2:
                    if not(capsules_pos[pair['elements'][0]['index']][:,0,2] >=  pair['elements'][1]['bounds'][0]).all(): return False
                    if not(capsules_pos[pair['elements'][0]['index']][:,0,2] <=  pair['elements'][1]['bounds'][1]).all(): return False
                    if not(capsules_pos[pair['elements'][0]['index']][:,1,2] >=  pair['elements'][1]['bounds'][0]).all(): return False
                    if not(capsules_pos[pair['elements'][0]['index']][:,1,2] <=  pair['elements'][1]['bounds'][1]).all(): return False
            return True