# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:41:27 2018

@author: eee

"""
from make_parameter import PreParam
import numpy as np
from pyomo.opt import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.environ import *
import pickle
from copy import deepcopy, copy
from collections import Counter
import multiprocessing  as mp
import time
from solver_single import solve_single
import cvxpy as cp
from itertools import combinations

class OptQcq(PreParam):

    def __init__(self, path_casedata,set_disturb, mode):
        super(OptQcq,self).__init__(path_casedata,set_disturb, mode)

        self.optsolver = SolverFactory('ipopt')
        self.optsolver.set_options('constr_viol_tol=1e-10')
        #self.solver_manager = SolverManagerFactory('pyro')
        
        if   mode == 30: self.opt_qq()
        elif mode == 31: self.opt_qq_sdp_large()
        elif mode == 32: self.opt_qq_sdp_small()
        elif mode == 33: self.opt_qq_sdp_small_admm()
        elif mode == 34: self.opt_qq_sdp_small_admm_fs()  
            

    def define_z_lambda(self):
        '''
        define z and lambda and initialize
        '''
        # define global variables z 
        self.z_m = deepcopy(self.m_0)
        self.z_d = deepcopy(self.d_0)
        self.z_w = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower_union for i_bus in self.i_gen], [w_0_i for ki in self.M_lower_union for w_0_i in self.w_0] ))
        self.z_theta = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower_union for i_bus in self.i_gl], [theta_0_i for ki in self.M_lower_union for theta_0_i in self.theta_gl_0] ))

        # define Lambda for each s \in P
        self.Lambda =  dict()
        for s in self.P:
            self.Lambda[s] =  dict()
            self.Lambda[s]['m'] = dict(zip(self.m_0.keys(), [0 for i in self.m_0.keys()]))
            self.Lambda[s]['d'] = dict(zip(self.d_0.keys(), [0 for i in self.d_0.keys()]))
            self.Lambda[s]['w']     = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower[s] + self.M_upper[s] for i_bus in self.i_gen], [0 for ki in self.M_lower_union + self.M_upper[s] for w_0_i in self.w_0] ))
            self.Lambda[s]['theta'] = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower[s] + self.M_upper[s]for i_bus in self.i_gl], [0 for ki in self.M_lower_union + self.M_upper[s] for theta_0_i in self.theta_gl_0] ))

    def _opt_xs_init(self, s):
        '''
        initialize the optimization model for optimization x_s, for a given s
        '''

        set_k_i_s = self.Xi[s]  # set_k_i for the subproblem s

        opt_sm = ConcreteModel()
        #===================== variables=========================
        # problem variables x_ls
        # index of opt variables and initial values of opt variables
        index_w = list()
        index_theta = list()
        w_0 = dict()
        theta_0 = dict()
        for (k, i) in set_k_i_s:
            for j in range(self.param_dd[k[0]][i][2]+1):
                # differential variables and algebraic variables
                for i_bus in self.i_gen:
                    index_w.append( (k[0], k[1], i, j, i_bus) )
                    w_0[(k[0], k[1], i, j, i_bus)] = self.w_0[i_bus-1]
                for i_bus in self.i_all:
                    index_theta.append( (k[0], k[1], i, j, i_bus) )
                    theta_0[(k[0], k[1], i, j, i_bus)] = self.theta_0[i_bus-1]

        opt_sm.m = Var(self.i_gen, initialize = self.m_0)
        opt_sm.d = Var(self.i_gen, initialize = self.d_0)
        opt_sm.w = Var(index_w, initialize = w_0)
        opt_sm.theta = Var(index_theta, initialize = theta_0)

        opt_sm.con = ConstraintList()
    
        J_ki = dict()
        for (k, i) in set_k_i_s:
            ########### J_ki ###############
            [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, 0)
            nc = self.param_dd[k[0]][i][2]  # order of disturbance k[0] (the same type of disturbances is with the same 
            s_tau= self.param_dd[k[0]][i][3]  # collocation points, tau, of disturbance k[0]
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  # length of time element i for disturbance k[0]

            J_theta = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum( ( opt_sm.theta[(k[0], k[1], i, j1, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, j1, i_bus_t)] ) * ( opt_sm.theta[(k[0], k[1], i, j2, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, j2, i_bus_t)] ) * ratio_B[i_bus_f-1, i_bus_t-1] for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )
                
            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum( ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum( ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( opt_sm.d[i_bus]**2 * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( opt_sm.m[i_bus]**2 * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( opt_sm.m[i_bus]* opt_sm.d[i_bus] * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * ( opt_sm.theta[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.theta[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            J_ki[(k,i)] =  h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]


            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    opt_sm.con.add( sum( opt_sm.theta[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * opt_sm.w[(k[0], k[1], i, r, i_bus)] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    opt_sm.con.add( sum( opt_sm.theta[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] * sin(opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    opt_sm.con.add( sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * (1/ opt_sm.m[i_bus]) * ( - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)]  +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] * sin(opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) for j_bus in self.i_all)) )
            
            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    opt_sm.con.add( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] * sin(opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)

                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] >= self.casedata['freq_band'][k[0]][key_fb][0] - 50 )
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] <= self.casedata['freq_band'][k[0]][key_fb][1] - 50 )
                            break
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        opt_sm.con.add(  ( opt_sm.theta[(k[0], k[1], i, r, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, r, i_bus_t)] ) <= 135/180*np.pi  )
                        opt_sm.con.add(  ( opt_sm.theta[(k[0], k[1], i, r, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, r, i_bus_t)] ) >= -135/180*np.pi  )
                
                    
                        
                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    opt_sm.con.add( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *opt_sm.m[i_bus] * sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    opt_sm.con.add( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *opt_sm.m[i_bus] * sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)] <= self.casedata['gencontrol'][i_gc, 7][0] )


        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            opt_sm.con.add( opt_sm.m[i_bus] >=  self.casedata['gencontrol'][i_gc, 2][0])
            opt_sm.con.add( opt_sm.m[i_bus] <=  self.casedata['gencontrol'][i_gc, 3][0])
            opt_sm.con.add( opt_sm.d[i_bus] >=  self.casedata['gencontrol'][i_gc, 4][0])
            opt_sm.con.add( opt_sm.d[i_bus] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in set_k_i_s:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    opt_sm.con.add( opt_sm.w[(k[0], k[1], i, 0, i_bus)] ==  self.w_0[i_bus-1] )
                for i_bus in self.i_gen + self.i_load +self.i_non:
                    opt_sm.con.add( opt_sm.theta[(k[0], k[1], i, 0, i_bus)] ==  self.theta_0[i_bus-1] )
            elif (k, i - 1) in set_k_i_s: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    opt_sm.con.add( opt_sm.w[(k[0], k[1], i, 0, i_bus)] ==  opt_sm.w[(k[0], k[1], i-1 , nc, i_bus)] )
                for i_bus in self.i_gen + self.i_load:
                    opt_sm.con.add( opt_sm.theta[(k[0], k[1], i, 0, i_bus)] ==  opt_sm.theta[(k[0], k[1], i-1, nc, i_bus)] )

        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in set_k_i_s)
        opt_sm.J = J
        opt_sm.obj = Objective(expr = opt_sm.J, sense=minimize)

        return opt_sm

    def opt_x_init(self):
        '''
        initialize model opt_x_s for all s \in P
        '''
        self.opt_x = dict()
        for s in self.P:
            self.opt_x[s] = self._opt_xs_init(s)

    def _opt_xs_modify(self, s):
        '''
        modify opt_xs, with augmented Lagrangian term added, for given s
        '''
        opt_sm = self.opt_x[s]

        opt_sm.L_1 = (  sum(self.Lambda[s]['m'][i_bus] * opt_sm.m[i_bus] + self.Lambda[s]['d'][i_bus] * opt_sm.d[i_bus] for i_bus in self.i_gen) 
        + sum(self.Lambda[s]['w'][ki[0][0], ki[0][1], ki[1], i_bus] * opt_sm.w[ki[0][0], ki[0][1], ki[1], 0,i_bus] for ki in self.M_lower[s] for i_bus in self.i_gen)
        + sum(self.Lambda[s]['w'][ki[0][0], ki[0][1], ki[1], i_bus] * opt_sm.w[ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], i_bus] for ki in self.M_upper[s] for i_bus in self.i_gen)
        + sum(self.Lambda[s]['theta'][ki[0][0], ki[0][1], ki[1], i_bus] * opt_sm.theta[ki[0][0], ki[0][1], ki[1], 0,i_bus] for ki in self.M_lower[s] for i_bus in self.i_gl)
        + sum(self.Lambda[s]['theta'][ki[0][0], ki[0][1], ki[1], i_bus] * opt_sm.theta[ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], i_bus] for ki in self.M_upper[s] for i_bus in self.i_gl)  )

        opt_sm.L_2 = (self.rho/2.0) * (  sum( (opt_sm.m[i_bus] - self.z_m[i_bus])**2 + (opt_sm.d[i_bus] - self.z_d[i_bus])**2 for i_bus in self.i_gen) 
        + sum((opt_sm.w[ki[0][0], ki[0][1], ki[1], 0,i_bus] - self.z_w[ki[0][0], ki[0][1], ki[1], i_bus])**2 for ki in self.M_lower[s] for i_bus in self.i_gen)
        + sum( (opt_sm.w[ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], i_bus] - self.z_w[ki[0][0], ki[0][1], ki[1] + 1, i_bus])**2 for ki in self.M_upper[s] for i_bus in self.i_gen)
        + sum((opt_sm.theta[ki[0][0], ki[0][1], ki[1], 0,i_bus] - self.z_theta[ki[0][0], ki[0][1], ki[1], i_bus])**2 for ki in self.M_lower[s] for i_bus in self.i_gl)
        + sum( (opt_sm.theta[ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], i_bus] - self.z_theta[ki[0][0], ki[0][1], ki[1] + 1, i_bus])**2 for ki in self.M_upper[s] for i_bus in self.i_gl)  )

        opt_sm.del_component('obj')
        opt_sm.obj = Objective( expr = opt_sm.J + opt_sm.L_1 + opt_sm.L_2, sense=minimize)

        self.opt_x[s] = opt_sm

    def opt_x_modify(self):
        '''
        modify opt_x with augmented Lagrangian term added, for all s \in P
        '''
        for s in self.P:
            self._opt_xs_modify(s)
   
    def opt_x_solve_mp(self):
        '''
        solve opt_x for all s \in P, with multiprocessing library
        '''
        n_cpu = mp.cpu_count()
        pool = mp.Pool(processes= n_cpu) 
        t3 = time.time()
        opt_x = [self.opt_x[s] for s in self.P]

        t4 = time.time()
        res_opt_x = pool.map(solve_single, opt_x)
        t5 = time.time()
        for s in self.P: self.opt_x[s] = res_opt_x.pop(0)
        t6 = time.time()
        
        print(t4-t3, t5 - t4, t6-t5)


    def opt_x_solve_pyomomp(self):
        '''
        solve opt_x for all s \in P, with parallel function provided by pyomo
        '''
        t3 = time.time()
        action_handle_map = {} # maps action handles to instances
        for s in self.P:
            action_handle = self.solver_manager.queue(self.opt_x[s], opt=self.optsolver,  tee=False)
            action_handle_map[action_handle] = s

        for s in self.P:
            this_action_handle = self.solver_manager.wait_any()
            self.solver_manager.get_results(this_action_handle)
        t4 = time.time()
        print(t4 - t3)  

    def opt_x_solve(self):
        '''
        solve opt_x for all s \in P, without multiprocess
        '''
        t3 = time.time()
        solver = SolverFactory('ipopt')
        solver.set_options('constr_viol_tol=1e-10')
        for s in self.P:
            solver.solve(self.opt_x[s], tee=False)
        t4 = time.time()
        print(t4 - t3)    

    def _find_s(self, ki):
        '''
        return s which ki in self.Xi[s]
        '''
        for s in self.P:
            if ki in self.Xi[s]:
                return s
                break


    def update_z(self):
        '''
        update self.z_ for each iteration
        '''
        # store the previous z before update for compute dual residual s
        self.z_p_m     = deepcopy(self.z_m)
        self.z_p_d     = deepcopy(self.z_d)
        self.z_p_w     = deepcopy(self.z_w)
        self.z_p_theta = deepcopy(self.z_theta)

        # update 
        for i_bus in self.i_gen:
            self.z_m[i_bus] = (1.0/self.N_s) * sum(self.opt_x[s].m[i_bus]() for s in self.P)
            self.z_d[i_bus] = (1.0/self.N_s) * sum(self.opt_x[s].d[i_bus]() for s in self.P)

        for ki in self.M_lower_union:
            s_ki = self._find_s(ki)    # ki in Xi[s_ki]
            s_ki_1 = self._find_s( ((ki[0][0], ki[0][1]), ki[1] - 1) )  # (k,i-1) in Xi[s_ki_1]
            nc_1 = self.param_dd[ki[0][0]][ki[1] - 1][2]  # n_c  for (k, i - 1)
            nc = self.param_dd[ki[0][0]][ki[1]][2]  # n_c for (k,i)
            for i_bus in self.i_gen:
                self.z_w[(ki[0][0], ki[0][1], ki[1], i_bus)] = 0.5 * ( self.opt_x[s_ki_1].w[ki[0][0], ki[0][1], ki[1] - 1, nc_1, i_bus]() + self.opt_x[s_ki].w[ki[0][0], ki[0][1], ki[1], nc, i_bus]() )
            for i_bus in self.i_gl:
                self.z_theta[(ki[0][0], ki[0][1], ki[1], i_bus)] = 0.5 * ( self.opt_x[s_ki_1].theta[ki[0][0], ki[0][1], ki[1] - 1, nc_1, i_bus]() + self.opt_x[s_ki].theta[ki[0][0], ki[0][1], ki[1], nc, i_bus]() )


    def update_lambda(self):
        '''
        update self.Lambda_ for each iteration
        '''
        # store the previous Lambda before update for compute dual residual r, unlike s, it not neessary
        self.Lambda_p = deepcopy(self.Lambda)
        #update
        for s in self.P:
            for i_bus in self.i_gen:
                self.Lambda[s]['m'][i_bus] = self.Lambda[s]['m'][i_bus] + self.rho * (self.opt_x[s].m[i_bus]() - self.z_m[i_bus])
                self.Lambda[s]['d'][i_bus] = self.Lambda[s]['d'][i_bus] + self.rho * (self.opt_x[s].d[i_bus]() - self.z_d[i_bus])
            for i_bus in self.i_gen:
                for ki in self.M_lower[s]:
                    self.Lambda[s]['w'][ki[0][0], ki[0][1], ki[1], i_bus] = self.Lambda[s]['w'][ki[0][0], ki[0][1], ki[1], i_bus] + self.rho *  ( self.opt_x[s].w[ki[0][0], ki[0][1], ki[1], 0,i_bus]() - self.z_w[ki[0][0], ki[0][1], ki[1], i_bus] )
                for ki in self.M_upper[s]:
                    self.Lambda[s]['w'][ki[0][0], ki[0][1], ki[1], i_bus] = self.Lambda[s]['w'][ki[0][0], ki[0][1], ki[1], i_bus] + self.rho * (self.opt_x[s].w[ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], i_bus]() - self.z_w[ki[0][0], ki[0][1], ki[1] + 1, i_bus])
            for i_bus in self.i_gl:
                for ki in self.M_lower[s]:
                    self.Lambda[s]['theta'][ki[0][0], ki[0][1], ki[1], i_bus] = self.Lambda[s]['theta'][ki[0][0], ki[0][1], ki[1], i_bus] + self.rho * (self.opt_x[s].theta[ki[0][0], ki[0][1], ki[1], 0,i_bus]() - self.z_theta[ki[0][0], ki[0][1], ki[1], i_bus])
                for ki in self.M_upper[s]:
                    self.Lambda[s]['theta'][ki[0][0], ki[0][1], ki[1], i_bus] = self.Lambda[s]['theta'][ki[0][0], ki[0][1], ki[1], i_bus] + self.rho * (self.opt_x[s].theta[ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], i_bus]() - self.z_theta[ki[0][0], ki[0][1], ki[1] + 1, i_bus])

    def termination(self):

        if self.kappa == 0:
            self.r_norm_2 = dict()
            self.s_norm_2 = dict()
            
        self.r_norm_2[self.kappa] = (1.0/self.rho) * (sum([np.square(np.array(list((Counter(self.Lambda[s]['m']) - Counter(self.Lambda_p[s]['m'])).values()))).sum() + np.square(np.array(list((Counter(self.Lambda[s]['d']) - Counter(self.Lambda_p[s]['d'])).values()))).sum() +  np.square(np.array(list((Counter(self.Lambda[s]['w']) - Counter(self.Lambda_p[s]['w'])).values()))).sum() +  np.square(np.array(list((Counter(self.Lambda[s]['theta']) - Counter(self.Lambda_p[s]['theta'])).values()))).sum() for s in self.P]))**0.5

        self.s_norm_2[self.kappa] =  self.rho * (self.N_s * np.square(np.array(list((Counter(self.z_m) -  Counter(self.z_p_m)).values()))).sum() + self.N_s * np.square(np.array(list((Counter(self.z_d) -  Counter(self.z_p_d)).values()))).sum() +  2 * np.square(np.array(list((Counter(self.z_w) -  Counter(self.z_p_w)).values()))).sum() + 2 * np.square(np.array(list((Counter(self.z_theta) -  Counter(self.z_p_w)).values()))).sum() )**0.5

        print(self.kappa, ':    ', self.r_norm_2[self.kappa], self.s_norm_2[self.kappa])


    def opt_sin_admm(self, rho = 2, sep_mode = 'k', N_s = 2, N_s_k = 2, N_s_i = 2):

        # parameters of admm
        self.rho = rho
        self.sep_mode = sep_mode
        self.N_s = N_s
        self.N_s_k = N_s_k
        self.N_s_i = N_s_i
        # separate the problem
        self.seperation_compute() 
        # define z anf lambda
        self.define_z_lambda()
        # initialize all the optimization model for x, get self.opt_x = dict() with the opt model for each s \in P
        self.opt_x_init()
        # iteration, including modify model, solve opt_x, update z and lambda
        t1 = time.time()
        for self.kappa in range(200):
            self.opt_x_modify()
            self.opt_x_solve()
            #self.opt_x_solve_pyomomp()
            #self.opt_x_solve_mp()  # use multiprocessing
            self.update_z()
            self.update_lambda()
            self.termination()
        t2 = time.time()

        print(t2-t1)

        self.opt_result_sin_admm  = deepcopy(self.opt_x)

    def opt_sin(self):

        '''
        solve the dvid model in centralized method with IPOPT with sin power flow model
        '''
        opt_sm = ConcreteModel()
        #===================== variables=========================
        # problem variables x_ls
        # index of opt variables and initial values of opt variables
        index_w = list()
        index_theta = list()
        w_0 = dict()
        theta_0 = dict()
        for (k, i) in self.set_k_i:
            for j in range(self.param_dd[k[0]][i][2]+1):
                # differential variables and algebraic variables
                for i_bus in self.i_gen:
                    index_w.append( (k[0], k[1], i, j, i_bus) )
                    w_0[(k[0], k[1], i, j, i_bus)] = self.w_0[i_bus-1] 
                for i_bus in self.i_all:
                    index_theta.append( (k[0], k[1], i, j, i_bus) )
                    theta_0[(k[0], k[1], i, j, i_bus)] = self.theta_0[i_bus-1]

        opt_sm.m = Var(self.i_gen, initialize = self.m_0)
        opt_sm.d = Var(self.i_gen, initialize = self.d_0)
        opt_sm.w = Var(index_w, initialize = w_0)
        opt_sm.theta = Var(index_theta, initialize = theta_0)


        opt_sm.con = ConstraintList()
    
        J_ki = dict()
        for (k, i) in self.set_k_i:
            ########### J_ki ###############
            print( k,i)
            [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, 0)
            nc = self.param_dd[k[0]][i][2]  # order of disturbance k[0] (the same type of disturbances is with the same 
            s_tau= self.param_dd[k[0]][i][3]  # collocation points, tau, of disturbance k[0]
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  # length of time element i for disturbance k[0]

            J_theta = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum( ( opt_sm.theta[(k[0], k[1], i, j1, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, j1, i_bus_t)] ) * ( opt_sm.theta[(k[0], k[1], i, j2, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, j2, i_bus_t)] ) * ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )
                
            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum( ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum( ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( opt_sm.d[i_bus]**2 * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( opt_sm.m[i_bus]**2 * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( opt_sm.m[i_bus]* opt_sm.d[i_bus] * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * ( opt_sm.theta[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.theta[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            J_ki[(k,i)] =  h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]


            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    opt_sm.con.add( sum( opt_sm.theta[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * opt_sm.w[(k[0], k[1], i, r, i_bus)] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    opt_sm.con.add( sum( opt_sm.theta[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] * sin(opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    opt_sm.con.add( sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * (1/ opt_sm.m[i_bus]) * ( - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)]  +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] * sin(opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) for j_bus in self.i_all)) )
            
            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    opt_sm.con.add( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] * sin(opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                
                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] >= self.casedata['freq_band'][k[0]][key_fb][0] - 50 )
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] <= self.casedata['freq_band'][k[0]][key_fb][1] - 50 )
                            break
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        opt_sm.con.add(  ( opt_sm.theta[(k[0], k[1], i, r, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, r, i_bus_t)] ) <= 135/180*np.pi  )
                        opt_sm.con.add(  ( opt_sm.theta[(k[0], k[1], i, r, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, r, i_bus_t)] ) >= -135/180*np.pi  )
                
                    
                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    opt_sm.con.add( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *opt_sm.m[i_bus] * sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    opt_sm.con.add( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *opt_sm.m[i_bus] * sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)] <= self.casedata['gencontrol'][i_gc, 7][0] )


        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            opt_sm.con.add( opt_sm.m[i_bus] >=  self.casedata['gencontrol'][i_gc, 2][0])
            opt_sm.con.add( opt_sm.m[i_bus] <=  self.casedata['gencontrol'][i_gc, 3][0])
            opt_sm.con.add( opt_sm.d[i_bus] >=  self.casedata['gencontrol'][i_gc, 4][0])
            opt_sm.con.add( opt_sm.d[i_bus] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    opt_sm.con.add( opt_sm.w[(k[0], k[1], i, 0, i_bus)] ==  self.w_0[i_bus-1] )
                for i_bus in self.i_gen + self.i_load + self.i_non:
                    opt_sm.con.add( opt_sm.theta[(k[0], k[1], i, 0, i_bus)] ==  self.theta_0[i_bus-1] )
            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    opt_sm.con.add( opt_sm.w[(k[0], k[1], i, 0, i_bus)] ==  opt_sm.w[(k[0], k[1], i-1 , nc, i_bus)] )
                for i_bus in self.i_gen + self.i_load:
                    opt_sm.con.add( opt_sm.theta[(k[0], k[1], i, 0, i_bus)] ==  opt_sm.theta[(k[0], k[1], i-1, nc, i_bus)] )

        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in self.set_k_i)
        opt_sm.J = J
        opt_sm.obj = Objective( expr = opt_sm.J , sense=minimize)

        ############# solver ################
        solver = SolverFactory('ipopt')
        solver.set_options('constr_viol_tol=1e-10')
        solver.solve(opt_sm,tee=True)

        self.opt_result_sin = deepcopy(opt_sm)

    def opt_ln(self):
        '''
        solve the dvid model in centralized method with IPOPT with linearized power flow model
        '''
        opt_sm = ConcreteModel()
        #===================== variables=========================
        # problem variables x_ls
        # index of opt variables and initial values of opt variables
        index_w = list()
        index_theta = list()
        w_0 = dict()
        theta_0 = dict()
        for (k, i) in self.set_k_i:
            for j in range(self.param_dd[k[0]][i][2]+1):
                # differential variables and algebraic variables
                for i_bus in self.i_gen:
                    index_w.append( (k[0], k[1], i, j, i_bus) )
                    w_0[(k[0], k[1], i, j, i_bus)] = self.w_0[i_bus-1] 
                for i_bus in self.i_all:
                    index_theta.append( (k[0], k[1], i, j, i_bus) )
                    theta_0[(k[0], k[1], i, j, i_bus)] = self.theta_0[i_bus-1]

        opt_sm.m = Var(self.i_gen, initialize = self.m_0)
        opt_sm.d = Var(self.i_gen, initialize = self.d_0)
        opt_sm.w = Var(index_w, initialize = w_0)
        opt_sm.theta = Var(index_theta, initialize = theta_0)


        opt_sm.con = ConstraintList()
    
        J_ki = dict()
        for (k, i) in self.set_k_i:
            ########### J_ki ###############
            print( k,i)
            [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, 0)
            nc = self.param_dd[k[0]][i][2]  # order of disturbance k[0] (the same type of disturbances is with the same 
            s_tau= self.param_dd[k[0]][i][3]  # collocation points, tau, of disturbance k[0]
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  # length of time element i for disturbance k[0]

            J_theta = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum( ( opt_sm.theta[(k[0], k[1], i, j1, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, j1, i_bus_t)] ) * ( opt_sm.theta[(k[0], k[1], i, j2, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, j2, i_bus_t)] ) * ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )
                
            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum( ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum( ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( opt_sm.d[i_bus]**2 * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( opt_sm.m[i_bus]**2 * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( opt_sm.m[i_bus]* opt_sm.d[i_bus] * ( opt_sm.w[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.w[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * ( opt_sm.theta[(k[0], k[1], i, j1, i_bus)] ) * ( opt_sm.theta[(k[0], k[1], i, j2, i_bus)] )  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            J_ki[(k,i)] =  h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]


            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    opt_sm.con.add( sum( opt_sm.theta[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * opt_sm.w[(k[0], k[1], i, r, i_bus)] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    opt_sm.con.add( sum( opt_sm.theta[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * (np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) )
                     for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    opt_sm.con.add( sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * (1/ opt_sm.m[i_bus]) * ( - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)]  +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    *( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) )
                    for j_bus in self.i_all)) )
            
            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    opt_sm.con.add( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                     * (np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (opt_sm.theta[(k[0], k[1], i, r, i_bus)] - opt_sm.theta[(k[0], k[1], i, r, j_bus)]) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ))
                     for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                
                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] >= self.casedata['freq_band'][k[0]][key_fb][0] - 50 )
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] <= self.casedata['freq_band'][k[0]][key_fb][1] - 50 )
                            break
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        opt_sm.con.add(  ( opt_sm.theta[(k[0], k[1], i, r, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, r, i_bus_t)] ) <= 135/180*np.pi  )
                        opt_sm.con.add(  ( opt_sm.theta[(k[0], k[1], i, r, i_bus_f)] -  opt_sm.theta[(k[0], k[1], i, r, i_bus_t)] ) >= -135/180*np.pi  )
                
                    
                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    opt_sm.con.add( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *opt_sm.m[i_bus] * sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    opt_sm.con.add( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *opt_sm.m[i_bus] * sum( opt_sm.w[(k[0], k[1], i, j, i_bus)] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - opt_sm.d[i_bus] * opt_sm.w[(k[0], k[1], i, r, i_bus)] <= self.casedata['gencontrol'][i_gc, 7][0] )


        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            opt_sm.con.add( opt_sm.m[i_bus] >=  self.casedata['gencontrol'][i_gc, 2][0])
            opt_sm.con.add( opt_sm.m[i_bus] <=  self.casedata['gencontrol'][i_gc, 3][0])
            opt_sm.con.add( opt_sm.d[i_bus] >=  self.casedata['gencontrol'][i_gc, 4][0])
            opt_sm.con.add( opt_sm.d[i_bus] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    opt_sm.con.add( opt_sm.w[(k[0], k[1], i, 0, i_bus)] ==  self.w_0[i_bus-1] )
                for i_bus in self.i_gen + self.i_load + self.i_non:
                    opt_sm.con.add( opt_sm.theta[(k[0], k[1], i, 0, i_bus)] ==  self.theta_0[i_bus-1] )
            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    opt_sm.con.add( opt_sm.w[(k[0], k[1], i, 0, i_bus)] ==  opt_sm.w[(k[0], k[1], i-1 , nc, i_bus)] )
                for i_bus in self.i_gen + self.i_load:
                    opt_sm.con.add( opt_sm.theta[(k[0], k[1], i, 0, i_bus)] ==  opt_sm.theta[(k[0], k[1], i-1, nc, i_bus)] )

        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in self.set_k_i)
        opt_sm.J = J
        opt_sm.obj = Objective( expr = opt_sm.J , sense=minimize)

        ############# solver ################
        solver = SolverFactory('ipopt')
        solver.set_options('constr_viol_tol=1e-10')
        solver.solve(opt_sm,tee=True)

        self.opt_result_ln = deepcopy(opt_sm)

    def opt_ln_sdp(self):
        '''
        lineralized power flow mode and SDP
        '''
        X = dict()
        n_X = dict()

        for (k, i) in self.set_k_i:
            n_X = 2 * len(self.i_gen) + (self.param_dd[k[0]][i][2]+1) * (self.n_bus + len(self.i_gen)) + 2 * (self.param_dd[k[0]][i][2]+1) * len(self.i_gen) 
            X[(k[0], k[1], i)] = cp.Variable((1 + n_X, 1 + n_X), symmetric =True)  
            
        # index map between X and [m, d, theta, w, lm, ld]
        idm = dict() # index map
        for (k, i) in self.set_k_i:
            idm[(k[0], k[1], i)] = dict()
            i_X = 1
            for i_bus in self.i_gen:
                idm[(k[0], k[1], i)][('m', i_bus)] = i_X 
                i_X += 1 
            for i_bus in self.i_gen:
                idm[(k[0], k[1], i)][('d', i_bus)] = i_X 
                i_X += 1

            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_all:
                    idm[(k[0], k[1], i)][('theta', j, i_bus)] = i_X 
                    i_X += 1
            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_gen:
                    idm[(k[0], k[1], i)][('w', j, i_bus)] = i_X 
                    i_X += 1
            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_gen:
                    idm[(k[0], k[1], i)][('lm', j, i_bus)] = i_X 
                    i_X += 1
            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_gen:
                    idm[(k[0], k[1], i)][('ld', j, i_bus)] = i_X 
                    i_X += 1
        self.idm = idm
        
        constraints = list()
        J_ki = dict()

        for (k, i) in self.set_k_i:
            ########### J_ki ###############
            print( k,i)
            [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, 0)
            nc = self.param_dd[k[0]][i][2]  # order of disturbance k[0] (the same type of disturbances is with the same 
            s_tau= self.param_dd[k[0]][i][3]  # collocation points, tau, of disturbance k[0]
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  # length of time element i for disturbance k[0]

            J_theta = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_f)], idm[(k[0], k[1], i)][('theta', j2, i_bus_f)]] 
                - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_f)], idm[(k[0], k[1], i)][('theta', j2, i_bus_t)]] 
                - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_t)], idm[(k[0], k[1], i)][('theta', j2, i_bus_f)]] 
                + X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_t)], idm[(k[0], k[1], i)][('theta', j2, i_bus_t)]] 
                * ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )
                
            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', j1, i_bus)], idm[(k[0], k[1], i)][('w', j2, i_bus)]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum(
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', j1, i_bus)], idm[(k[0], k[1], i)][('w', j2, i_bus)]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('ld', j1, i_bus)], idm[(k[0], k[1], i)][('ld', j2, i_bus)]]  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j1, i_bus)], idm[(k[0], k[1], i)][('lm', j2, i_bus)]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j1, i_bus)], idm[(k[0], k[1], i)][('ld', j2, i_bus)]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus)], idm[(k[0], k[1], i)][('theta', j2, i_bus)]]  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            J_ki[(k,i)] =  h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]


            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    constraints.append( sum( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    constraints.append( sum( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * (np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus)]] -  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    constraints.append( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * ( - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]]  +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus)]] -  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all)) )
            
            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    constraints.append( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                     * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus)]] -  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                
                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] >= self.casedata['freq_band'][k[0]][key_fb][0] - 50 )
                            constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] <= self.casedata['freq_band'][k[0]][key_fb][1] - 50 )
                            break 
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        constraints.append(  (  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_f)]] -   X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_t)]] ) <= 135/180*np.pi  )
                        constraints.append(  (  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_f)]] -   X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_t)]] ) >= -135/180*np.pi  )
                
                    
                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *sum(  X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] <= self.casedata['gencontrol'][i_gc, 7][0] )


        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('m', i_bus)]] >=  self.casedata['gencontrol'][i_gc, 2][0])
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('m', i_bus)]] <=  self.casedata['gencontrol'][i_gc, 3][0])
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('d', i_bus)]] >=  self.casedata['gencontrol'][i_gc, 4][0])
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('d', i_bus)]] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', 0, i_bus)]] ==  self.w_0[i_bus-1] )
                for i_bus in self.i_gen + self.i_load + self.i_non:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', 0, i_bus)]] ==  self.theta_0[i_bus-1] )
            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', 0, i_bus)]] ==  X[(k[0], k[1], i-1)][0, idm[(k[0], k[1], i-1)][('w', nc, i_bus)]] )
                for i_bus in self.i_gen + self.i_load:
                    constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', 0, i_bus)]] ==  X[(k[0], k[1], i-1)][0, idm[(k[0], k[1], i-1)][('theta', nc, i_bus)]]  )

        # constraints for the lifting variables l_m and l_d
        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]
            for r in range(0, nc+1):
                for i_bus in self.i_gen:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('lm', r, i_bus)]] == X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] )
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('ld', r, i_bus)]] == X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] )

        # constraints for the first element of matrices X
        for (k, i) in self.set_k_i:
            constraints.append( X[(k[0], k[1], i)][0, 0] == 1)

        # sharing variables in X_[m,d] for different (k,i)
        for (k, i) in self.set_k_i[1:]:
            constraints.append(X[(k[0], k[1], i)][0:2*len(self.i_gen)+1+1, 0:2*len(self.i_gen)+1+1] 
                == X[self.set_k_i[0][0][0],self.set_k_i[0][0][1], self.set_k_i[0][1]][0:2*len(self.i_gen)+1+1, 0:2*len(self.i_gen)+1+1])
            
        #semidefine constraints
        for (k, i) in self.set_k_i:
            constraints.append( X[(k[0], k[1], i)] >> 0 )


        # constraints for tightening the relaxation
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                for (j1, j2) in list(combinations(range(self.param_dd[k[0]][i][2]+1),2)):
                    constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j1, i_bus)], idm[(k[0], k[1], i)][('ld', j2, i_bus)]] == X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j2, i_bus)], idm[(k[0], k[1], i)][('ld', j1, i_bus)]] )


        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]  
            s_tau= self.param_dd[k[0]][i][3]  
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  
            for r in range(1, nc+1): # for each colloction point
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # frequency constraints 
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', r, i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] <= (self.casedata['freq_band'][k[0]][key_fb][1] - 50)**2 )
                            break

                # branch rotor angle difference constraints 
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        constraints.append( 
                        X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_f)], idm[(k[0], k[1], i)][('theta', r, i_bus_f)]] 
                        - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_f)], idm[(k[0], k[1], i)][('theta', r, i_bus_t)]]
                        - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_t)], idm[(k[0], k[1], i)][('theta', r, i_bus_f)]]
                        + X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_t)], idm[(k[0], k[1], i)][('theta', r, i_bus_t)]]  
                        <= (135/180*np.pi)**2  )
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('m', i_bus)]]  >=  (self.casedata['gencontrol'][i_gc, 2][0])**2  )
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('m', i_bus)]]  <=  (self.casedata['gencontrol'][i_gc, 3][0])**2 )
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('d', i_bus)]]  >=  (self.casedata['gencontrol'][i_gc, 4][0])**2 )
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('d', i_bus)]]  <=  (self.casedata['gencontrol'][i_gc, 5][0])**2 )

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for j in range(0, self.param_dd[k[0]][i][2] + 1):
                    for i_bus in self.i_gen:
                        constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', 0, i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]] ==  (self.w_0[i_bus-1]) * X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', j, i_bus)]] )
                    for i_bus in self.i_gen + self.i_load + self.i_non:
                        constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', 0, i_bus)], idm[(k[0], k[1], i)][('theta', j, i_bus)]] ==  (self.theta_0[i_bus-1]) * X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', j, i_bus)]] )

            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    constraints.append(  X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', 0, i_bus)], idm[(k[0], k[1], i)][('w', 0, i_bus)]] == X[(k[0], k[1], i-1)][idm[(k[0], k[1], i-1)][('w', nc, i_bus)], idm[(k[0], k[1], i-1)][('w', nc, i_bus)]] )
                for i_bus in self.i_gen + self.i_load:
                    constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', 0, i_bus)], idm[(k[0], k[1], i)][('theta', 0, i_bus)]] == X[(k[0], k[1], i-1)][idm[(k[0], k[1], i-1)][('theta', nc, i_bus)], idm[(k[0], k[1], i-1)][('theta', nc, i_bus)]] )



        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in self.set_k_i)
        objective = cp.Minimize(J)
        opt_sm = cp.Problem(objective, constraints)
        opt_sm.solve(verbose = True)

        self.opt_result_ln_sdp = deepcopy(opt_sm)
        
        self.opt_result_ln_sdp_X = X
        
    def opt_ln_sdp_spa(self):
        '''
        lineralized power flow mode and SDP
        '''
        X = dict()
        n_X = dict()

        for (k, i) in self.set_k_i:
            n_X = 2 * len(self.i_gen) + (self.param_dd[k[0]][i][2]+1) * (self.n_bus + len(self.i_gen)) + 2 * (self.param_dd[k[0]][i][2]+1) * len(self.i_gen) 
            X[(k[0], k[1], i)] = cp.Variable((1 + n_X, 1 + n_X), symmetric =True)  
            
        # index map between X and [m, d, theta, w, lm, ld]
        idm = dict() # index map
        for (k, i) in self.set_k_i:
            idm[(k[0], k[1], i)] = dict()
            i_X = 1
            for i_bus in self.i_gen:
                idm[(k[0], k[1], i)][('m', i_bus)] = i_X 
                i_X += 1 
            for i_bus in self.i_gen:
                idm[(k[0], k[1], i)][('d', i_bus)] = i_X 
                i_X += 1

            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_all:
                    idm[(k[0], k[1], i)][('theta', j, i_bus)] = i_X 
                    i_X += 1
            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_gen:
                    idm[(k[0], k[1], i)][('w', j, i_bus)] = i_X 
                    i_X += 1
            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_gen:
                    idm[(k[0], k[1], i)][('lm', j, i_bus)] = i_X 
                    i_X += 1
            for j in range(self.param_dd[k[0]][i][2]+1):
                for i_bus in self.i_gen:
                    idm[(k[0], k[1], i)][('ld', j, i_bus)] = i_X 
                    i_X += 1
        self.idm = idm
        
        constraints = list()
        J_ki = dict()

        for (k, i) in self.set_k_i:
            ########### J_ki ###############
            print( k,i)
            [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, 0)
            nc = self.param_dd[k[0]][i][2]  # order of disturbance k[0] (the same type of disturbances is with the same 
            s_tau= self.param_dd[k[0]][i][3]  # collocation points, tau, of disturbance k[0]
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  # length of time element i for disturbance k[0]

            J_theta = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_f)], idm[(k[0], k[1], i)][('theta', j2, i_bus_f)]] 
                - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_f)], idm[(k[0], k[1], i)][('theta', j2, i_bus_t)]] 
                - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_t)], idm[(k[0], k[1], i)][('theta', j2, i_bus_f)]] 
                + X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_t)], idm[(k[0], k[1], i)][('theta', j2, i_bus_t)]] 
                * ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )
                
            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', j1, i_bus)], idm[(k[0], k[1], i)][('w', j2, i_bus)]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum(
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', j1, i_bus)], idm[(k[0], k[1], i)][('w', j2, i_bus)]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('ld', j1, i_bus)], idm[(k[0], k[1], i)][('ld', j2, i_bus)]]  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j1, i_bus)], idm[(k[0], k[1], i)][('lm', j2, i_bus)]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j1, i_bus)], idm[(k[0], k[1], i)][('ld', j2, i_bus)]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus)], idm[(k[0], k[1], i)][('theta', j2, i_bus)]]  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            J_ki[(k,i)] =  h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]


            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    constraints.append( sum( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    constraints.append( sum( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * (np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus)]] -  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    constraints.append( sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * ( - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]]  +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus)]] -  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all)) )
            
            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    constraints.append( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                     * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus)]] -  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                
                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] >= self.casedata['freq_band'][k[0]][key_fb][0] - 50 )
                            constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] <= self.casedata['freq_band'][k[0]][key_fb][1] - 50 )
                            break 
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        constraints.append(  (  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_f)]] -   X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_t)]] ) <= 135/180*np.pi  )
                        constraints.append(  (  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_f)]] -   X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', r, i_bus_t)]] ) >= -135/180*np.pi  )
                
                    
                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) * sum( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) * sum(  X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] <= self.casedata['gencontrol'][i_gc, 7][0] )


        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('m', i_bus)]] >=  self.casedata['gencontrol'][i_gc, 2][0])
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('m', i_bus)]] <=  self.casedata['gencontrol'][i_gc, 3][0])
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('d', i_bus)]] >=  self.casedata['gencontrol'][i_gc, 4][0])
            constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('d', i_bus)]] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', 0, i_bus)]] ==  self.w_0[i_bus-1] )
                for i_bus in self.i_gen + self.i_load + self.i_non:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', 0, i_bus)]] ==  self.theta_0[i_bus-1] )
            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', 0, i_bus)]] ==  X[(k[0], k[1], i-1)][0, idm[(k[0], k[1], i-1)][('w', nc, i_bus)]] )
                for i_bus in self.i_gen + self.i_load:
                    constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', 0, i_bus)]] ==  X[(k[0], k[1], i-1)][0, idm[(k[0], k[1], i-1)][('theta', nc, i_bus)]]  )

        # constraints for the lifting variables l_m and l_d
        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]
            for r in range(0, nc+1):
                for i_bus in self.i_gen:
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('lm', r, i_bus)]] == X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] )
                    constraints.append( X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('ld', r, i_bus)]] == X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] )

        # constraints for the first element of matrices X
        for (k, i) in self.set_k_i:
            constraints.append( X[(k[0], k[1], i)][0, 0] == 1)

        # sharing variables in X_[m,d] for different (k,i)
        for (k, i) in self.set_k_i[1:]:
            constraints.append(X[(k[0], k[1], i)][0:2*len(self.i_gen)+1+1, 0:2*len(self.i_gen)+1+1] 
                == X[self.set_k_i[0][0][0],self.set_k_i[0][0][1], self.set_k_i[0][1]][0:2*len(self.i_gen)+1+1, 0:2*len(self.i_gen)+1+1])
            
        #semidefine constraints
        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                index_block = list()
                index_block.append(0)
                index_block.append( idm[(k[0], k[1], i)][('m', i_bus)] ) 
                [ index_block.append( idm[(k[0], k[1], i)][('w', r, i_bus)] ) for r in range(0, nc + 1)]

                constraints.append( X[(k[0], k[1], i)][:, index_block][index_block, :]     >> 0 )

            for i_bus in self.i_gen:
                index_block = list()
                index_block.append(0)
                index_block.append( idm[(k[0], k[1], i)][('d', i_bus)] )
                [ index_block.append( idm[(k[0], k[1], i)][('w', r, i_bus)] ) for r in range(0, nc + 1)]

                constraints.append( X[(k[0], k[1], i)][:, index_block][index_block, :]     >> 0 )

            for i_bus in self.i_gen:
                index_block = list()
                index_block.append(0)
                [ index_block.append( idm[(k[0], k[1], i)][('lm', r, i_bus)] ) for r in range(0, nc + 1)]
                [ index_block.append( idm[(k[0], k[1], i)][('ld', r, i_bus)] ) for r in range(0, nc + 1)]

                constraints.append( X[(k[0], k[1], i)][:, index_block][index_block, :]     >> 0 )
            
            index_block = list()
            index_block.append(0)
            for i_bus in self.i_all:
                [ index_block.append( idm[(k[0], k[1], i)][('theta', r, i_bus)] ) for r in range(0, nc + 1)]
            constraints.append( X[(k[0], k[1], i)][:, index_block][index_block, :]     >> 0 )



        # constraints for tightening the relaxation
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                for (j1, j2) in list(combinations(range(self.param_dd[k[0]][i][2]+1),2)):
                    constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j1, i_bus)], idm[(k[0], k[1], i)][('ld', j2, i_bus)]] == X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('lm', j2, i_bus)], idm[(k[0], k[1], i)][('ld', j1, i_bus)]] )


        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]  
            s_tau= self.param_dd[k[0]][i][3]  
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  
            for r in range(1, nc+1): # for each colloction point
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # frequency constraints 
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', r, i_bus)], idm[(k[0], k[1], i)][('w', r, i_bus)]] <= (self.casedata['freq_band'][k[0]][key_fb][1] - 50)**2 )
                            break

                # branch rotor angle difference constraints 
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        constraints.append( 
                        X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_f)], idm[(k[0], k[1], i)][('theta', r, i_bus_f)]] 
                        - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_f)], idm[(k[0], k[1], i)][('theta', r, i_bus_t)]]
                        - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_t)], idm[(k[0], k[1], i)][('theta', r, i_bus_f)]]
                        + X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', r, i_bus_t)], idm[(k[0], k[1], i)][('theta', r, i_bus_t)]]  
                        <= (135/180*np.pi)**2  )
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('m', i_bus)]]  >=  (self.casedata['gencontrol'][i_gc, 2][0])**2  )
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('m', i_bus)], idm[(k[0], k[1], i)][('m', i_bus)]]  <=  (self.casedata['gencontrol'][i_gc, 3][0])**2 )
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('d', i_bus)]]  >=  (self.casedata['gencontrol'][i_gc, 4][0])**2 )
            constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('d', i_bus)], idm[(k[0], k[1], i)][('d', i_bus)]]  <=  (self.casedata['gencontrol'][i_gc, 5][0])**2 )

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for j in range(0, self.param_dd[k[0]][i][2] + 1):
                    for i_bus in self.i_gen:
                        constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', 0, i_bus)], idm[(k[0], k[1], i)][('w', j, i_bus)]] ==  (self.w_0[i_bus-1]) * X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', j, i_bus)]] )
                    for i_bus in self.i_gen + self.i_load + self.i_non:
                        constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', 0, i_bus)], idm[(k[0], k[1], i)][('theta', j, i_bus)]] ==  (self.theta_0[i_bus-1]) * X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('theta', j, i_bus)]] )

            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    constraints.append(  X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('w', 0, i_bus)], idm[(k[0], k[1], i)][('w', 0, i_bus)]] == X[(k[0], k[1], i-1)][idm[(k[0], k[1], i-1)][('w', nc, i_bus)], idm[(k[0], k[1], i-1)][('w', nc, i_bus)]] )
                for i_bus in self.i_gen + self.i_load:
                    constraints.append( X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', 0, i_bus)], idm[(k[0], k[1], i)][('theta', 0, i_bus)]] == X[(k[0], k[1], i-1)][idm[(k[0], k[1], i-1)][('theta', nc, i_bus)], idm[(k[0], k[1], i-1)][('theta', nc, i_bus)]] )

        
        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in self.set_k_i)
        objective = cp.Minimize(J)
        opt_sm = cp.Problem(objective, constraints)
        opt_sm.solve(verbose = True)

        self.opt_result_ln_sdp = deepcopy(opt_sm)
        
        self.opt_result_ln_sdp_X = X


    def save(self, path_result):

        self.path_result = path_result
        with open(self.path_result, 'wb') as fpr:
            pickle.dump(self, fpr)
        print('|==== Successfully save results ====|')

