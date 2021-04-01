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
import cvxpy as cp
from itertools import combinations
from operator import itemgetter
from numpy import linalg as LA
import mosek

class OptLin(PreParam):

    def __init__(self, path_casedata, set_disturb, mode):
        super(OptLin,self).__init__(path_casedata, set_disturb, mode)

        self.optsolver = SolverFactory('ipopt')
        self.optsolver.set_options('constr_viol_tol=1e-10')
        
        if   mode == 20: self.opt_lin()
        elif mode == 21: self.opt_lin_sdp_large()
        elif mode == 22: self.opt_lin_sdp_small()
        elif mode == 23: self.opt_lin_sdp_small_admm()
        elif mode == 24: self.opt_lin_sdp_small_admm_fs()
            

    def opt_lin(self):
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
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] >= (self.casedata['freq_band'][k[0]][key_fb][0] - 50) * 2 * np.pi )
                            opt_sm.con.add( opt_sm.w[(k[0], k[1], i, r, i_bus)] <= (self.casedata['freq_band'][k[0]][key_fb][1] - 50) * 2 * np.pi )
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

        self.opt_result = deepcopy(opt_sm)

    def opt_lin_sdp_large(self):
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
                (
                X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_f)], idm[(k[0], k[1], i)][('theta', j2, i_bus_f)]] 
                - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_f)], idm[(k[0], k[1], i)][('theta', j2, i_bus_t)]] 
                - X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_t)], idm[(k[0], k[1], i)][('theta', j2, i_bus_f)]] 
                + X[(k[0], k[1], i)][idm[(k[0], k[1], i)][('theta', j1, i_bus_t)], idm[(k[0], k[1], i)][('theta', j2, i_bus_t)]] 
                ) * ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )
                
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
                            constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] >= (self.casedata['freq_band'][k[0]][key_fb][0] - 50) * 2 * np.pi )
                            constraints.append(  X[(k[0], k[1], i)][0, idm[(k[0], k[1], i)][('w', r, i_bus)]] <= (self.casedata['freq_band'][k[0]][key_fb][1] - 50) * 2 * np.pi )
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

        '''
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
        '''


        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in self.set_k_i)
        objective = cp.Minimize(J)
        opt_sm = cp.Problem(objective, constraints)
        opt_sm.solve(solver = 'MOSEK', verbose = False)

        self.opt_result_ln_sdp = deepcopy(opt_sm)
        
        self.opt_result_ln_sdp_X = X
        
    def opt_lin_sdp_small(self):
        '''
        lineralized power flow mode and SDP
        '''
        X_lmd = dict()
        X_mdw = dict()
        X_theta = dict()

        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                # X_lmd :  [1 lm(nc + 1)  ld(nc + 1)]^T [1 lm(nc + 1)  ld(nc + 1)]
                n_lmd = 1 + 2*(nc + 1)
                X_lmd[(k[0], k[1], i, i_bus)] = cp.Variable(( n_lmd, n_lmd), symmetric =True)  
                
            for i_bus in self.i_gen:
                # X_mdw :  [1 m(1) d(1)  w(nc + 1)]^T [1 m(1) d(1)  w(nc + 1)]  
                n_mdw = 1 + 2+ (nc + 1) 
                X_mdw[(k[0], k[1], i, i_bus)] = cp.Variable(( n_mdw, n_mdw), symmetric =True)  

            for i_clique  in self.clique_tree_theta['node'].keys():
                # X_theta : [1 ...]
                n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                X_theta[(k[0], k[1], i, i_clique)] = cp.Variable(( n_theta, n_theta), symmetric =True)  

        ind_lmd = dict()
        ind_mdw = dict()
        ind_theta = dict()

        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]

            i_lmd = 1
            for j in range(0, nc+1):
                ind_lmd[(k[0], k[1], i, j, 'lm')] = i_lmd
                i_lmd += 1
            for j in range(0, nc+1):
                ind_lmd[(k[0], k[1], i, j, 'ld')] = i_lmd
                i_lmd += 1

            i_mdw = 1
            ind_mdw[(k[0], k[1], i, 'm')] = i_mdw
            i_mdw += 1
            ind_mdw[(k[0], k[1], i, 'd')] = i_mdw
            i_mdw += 1
            for j in range(0, nc+1):
                ind_mdw[(k[0], k[1], i, j, 'w')] = i_mdw
                i_mdw += 1

            for i_clique  in self.clique_tree_theta['node'].keys():
                i_theta = 1
                for j in range(0, nc+1):
                    for i_bus in self.clique_tree_theta['node'][i_clique]:
                        ind_theta[(k[0], k[1], i, i_clique, j, i_bus)] = i_theta
                        i_theta += 1


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
                (
                  X_theta[(k[0], k[1], i, self.bus_clique[i_bus_f])][ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], j1, i_bus_f)], ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], j2, i_bus_f)]]
                - X_theta[(k[0], k[1], i, self.branch_clique[(i_bus_f, i_bus_t)])][ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_f, i_bus_t)], j1, i_bus_f)], ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_f, i_bus_t)], j2, i_bus_t)]]
                - X_theta[(k[0], k[1], i, self.branch_clique[(i_bus_t, i_bus_f)])][ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_t, i_bus_f)], j1, i_bus_t)], ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_t, i_bus_f)], j2, i_bus_f)]]
                + X_theta[(k[0], k[1], i, self.bus_clique[i_bus_t])][ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], j1, i_bus_t)], ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], j2, i_bus_t)]]
                )* ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, j1, 'w')], ind_mdw[(k[0], k[1], i, j2, 'w')]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum(
                X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, j1, 'w')], ind_mdw[(k[0], k[1], i, j2, 'w')]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( X_lmd[(k[0], k[1], i, i_bus)][ind_lmd[(k[0], k[1], i, j1, 'ld')], ind_lmd[(k[0], k[1], i, j2, 'ld')]]  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( X_lmd[(k[0], k[1], i, i_bus)][ind_lmd[(k[0], k[1], i, j1, 'lm')], ind_lmd[(k[0], k[1], i, j2, 'lm')]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( X_lmd[(k[0], k[1], i, i_bus)][ind_lmd[(k[0], k[1], i, j1, 'lm')], ind_lmd[(k[0], k[1], i, j2, 'ld')]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )


            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j1, i_bus)], ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j2, i_bus)]]  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            J_ki[(k,i)] =   h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]


            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    constraints.append( sum( X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, r, 'w')]] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    constraints.append( sum( X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * (np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], r, i_bus)]] -  X_theta[(k[0], k[1], i, self.bus_clique[j_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[j_bus], r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    constraints.append( sum( X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'm')], ind_mdw[(k[0], k[1], i, j, 'w')]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * ( - X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'd')], ind_mdw[(k[0], k[1], i, r, 'w')]] +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (  X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], r, i_bus)]] -  X_theta[(k[0], k[1], i, self.bus_clique[j_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[j_bus], r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all)) )


            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    constraints.append( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                     * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], r, i_bus)]] -  X_theta[(k[0], k[1], i, self.bus_clique[j_bus])][0, ind_theta[(k[0], k[1], i, self.bus_clique[j_bus], r, j_bus)]] ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                
                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            constraints.append(  X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, r, 'w')]] >= (self.casedata['freq_band'][k[0]][key_fb][0] - 50) * 2 * np.pi )
                            constraints.append(  X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, r, 'w')]] <= (self.casedata['freq_band'][k[0]][key_fb][1] - 50) * 2 * np.pi )
                            break 
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        constraints.append(  (  X_theta[(k[0], k[1], i, self.bus_clique[i_bus_f])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], r, i_bus_f)]] - X_theta[(k[0], k[1], i, self.bus_clique[i_bus_t])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], r, i_bus_t)]] ) <= 135/180*np.pi  )
                        constraints.append(  (  X_theta[(k[0], k[1], i, self.bus_clique[i_bus_f])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], r, i_bus_f)]] - X_theta[(k[0], k[1], i, self.bus_clique[i_bus_t])][0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], r, i_bus_t)]] ) >= -135/180*np.pi  )


                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) * sum( X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'm')], ind_mdw[(k[0], k[1], i, j, 'w')]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'd')], ind_mdw[(k[0], k[1], i, r, 'w')]] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *sum(  X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'm')], ind_mdw[(k[0], k[1], i, j, 'w')]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'd')], ind_mdw[(k[0], k[1], i, r, 'w')]] <= self.casedata['gencontrol'][i_gc, 7][0] )


        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            constraints.append( X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, 'm')]] >=  self.casedata['gencontrol'][i_gc, 2][0])
            constraints.append( X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, 'm')]] <=  self.casedata['gencontrol'][i_gc, 3][0])
            constraints.append( X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, 'd')]] >=  self.casedata['gencontrol'][i_gc, 4][0])
            constraints.append( X_mdw[(k[0], k[1], i, i_bus)][0, ind_mdw[(k[0], k[1], i, 'd')]] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in self.set_k_i:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    constraints.append( X_mdw[(k[0], k[1], i, i_bus)][[0, ind_mdw[(k[0], k[1], i, 0, 'w')]],:][:,[0, ind_mdw[(k[0], k[1], i, 0, 'w')]]] ==   np.array([[1, self.w_0[i_bus-1]]]).T.dot(np.array([[1, self.w_0[i_bus-1]]])))
                for i_bus in self.i_gen + self.i_load + self.i_non:
                    constraints.append( X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][[0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]],:][:,[0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]]] ==  np.array([[1, self.theta_0[i_bus-1] ]]).T.dot(np.array([[1, self.theta_0[i_bus-1] ]]) ) )
            elif (k, i - 1) in self.set_k_i: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    constraints.append( X_mdw[(k[0], k[1], i, i_bus)][[0, ind_mdw[(k[0], k[1], i, 0, 'w')]],:][:,[0, ind_mdw[(k[0], k[1], i, 0, 'w')]]] ==  X_mdw[(k[0], k[1], i-1, i_bus)][[0, ind_mdw[(k[0], k[1], i-1, nc, 'w')]],:][:,[0, ind_mdw[(k[0], k[1], i-1, nc, 'w')]]] )
                for i_bus in self.i_gen + self.i_load:
                    constraints.append(  X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][[0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]],:][:,[0, ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]]] ==  X_theta[(k[0], k[1], i-1, self.bus_clique[i_bus])][[0, ind_theta[(k[0], k[1], i-1, self.bus_clique[i_bus], nc, i_bus)]],:][:,[0, ind_theta[(k[0], k[1], i-1, self.bus_clique[i_bus], nc, i_bus)]]] )

        # constraints for the lifting variables l_m and l_d
        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]
            for r in range(0, nc+1):
                for i_bus in self.i_gen:
                    constraints.append( X_lmd[(k[0], k[1], i, i_bus)][0, ind_lmd[(k[0], k[1], i, r, 'lm')]] == X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'm')], ind_mdw[(k[0], k[1], i, r, 'w')]] )
                    constraints.append(  X_lmd[(k[0], k[1], i, i_bus)][0, ind_lmd[(k[0], k[1], i, r, 'ld')]] == X_mdw[(k[0], k[1], i, i_bus)][ind_mdw[(k[0], k[1], i, 'd')], ind_mdw[(k[0], k[1], i, r, 'w')]] )

        # constraints for the first element of matrices X
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                constraints.append(X_lmd[(k[0], k[1], i, i_bus)][0, 0] == 1)
                constraints.append(X_mdw[(k[0], k[1], i, i_bus)][0, 0] == 1)

            for i_clique  in self.clique_tree_theta['node'].keys():
                constraints.append( X_theta[(k[0], k[1], i, i_clique)][0, 0] == 1 )

        # sharing variables in X_mdw for different (k,i)
        for (k, i) in self.set_k_i[1:]:
            root_k0, root_k1, root_i = self.set_k_i[0][0][0], self.set_k_i[0][0][1], self.set_k_i[0][1]
            for i_bus in self.i_gen:
                constraints.append( X_mdw[(k[0], k[1], i, i_bus)][[0, ind_mdw[(k[0], k[1], i, 'm')], ind_mdw[(k[0], k[1], i, 'd')]], :][:, [0, ind_mdw[(k[0], k[1], i, 'm')], ind_mdw[(k[0], k[1], i, 'd')]]] ==
                        X_mdw[(root_k0, root_k1, root_i, i_bus)][[0, ind_mdw[(root_k0, root_k1, root_i, 'm')], ind_mdw[(root_k0, root_k1, root_i, 'd')]], :][:, [0, ind_mdw[(root_k0, root_k1, root_i, 'm')], ind_mdw[(root_k0, root_k1, root_i, 'd')]]]  )
        # sharing variables in X_theta for different cliques.
        for (k, i) in self.set_k_i:
            nc = self.param_dd[k[0]][i][2]
            for edge_clique in self.clique_tree_theta['edge']:
                share_bus = list(set(self.clique_tree_theta['node'][edge_clique[0]] ).intersection(set(self.clique_tree_theta['node'][edge_clique[1]])))
                share_index_0, share_index_1 = [0], [0]
                for j in range(0, nc+1):
                    for i_bus in share_bus:
                        share_index_0.append(ind_theta[(k[0], k[1], i, edge_clique[0], j, i_bus)])
                        share_index_1.append(ind_theta[(k[0], k[1], i, edge_clique[1], j, i_bus)])

                constraints.append( X_theta[(k[0], k[1], i, edge_clique[0])][share_index_0, :][:,share_index_0] == X_theta[(k[0], k[1], i, edge_clique[1])][share_index_1, :][:,share_index_1] )


        #semidefine constraints
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                constraints.append( X_lmd[(k[0], k[1], i, i_bus)] >> 0 )
                constraints.append( X_mdw[(k[0], k[1], i, i_bus)] >> 0 )

            for i_clique  in self.clique_tree_theta['node'].keys():
                constraints.append( X_theta[(k[0], k[1], i, i_clique)] >> 0 )

        ###### objective function ###############
        J  = sum(J_ki[(k,i)]  for (k, i) in self.set_k_i)
        objective = cp.Minimize(J)
        opt_sm = cp.Problem(objective, constraints)
        opt_sm.solve(solver = 'MOSEK', verbose = True, mosek_params={mosek.iparam.num_threads: 1})

        self.opt_result = deepcopy(opt_sm)
        
        self.opt_result_X_lmd = dict()
        self.opt_result_X_mdw = dict()
        self.opt_result_X_theta = dict()
        for (k, i) in self.set_k_i:
            for i_bus in self.i_gen:
                self.opt_result_X_lmd[(k[0], k[1], i, i_bus)] = X_lmd[(k[0], k[1], i, i_bus)]
                self.opt_result_X_mdw[(k[0], k[1], i, i_bus)] = X_mdw[(k[0], k[1], i, i_bus)]
            for i_clique  in self.clique_tree_theta['node'].keys():
                self.opt_result_X_theta[(k[0], k[1], i, i_clique)] = X_theta[(k[0], k[1], i, i_clique)]


#=============================The following is the block for function "opt_ln_sdp_small_admm" ant its subfunction================================#

    def opt_lin_sdp_small_admm(self, rho = 1, sep_mode = 'i', N_s = 3, N_s_k = 1, N_s_i = 3):

        # parameters of admm
        self.rho = rho
        self.sep_mode = sep_mode
        self.N_s = N_s
        self.N_s_k = N_s_k
        self.N_s_i = N_s_i

        self.rho_tau  = 2#1.2
        self.rho_mu  = 10

        self.epsilon_abs = 1e-5
        self.epsilon_rel = 1e-3

        # separate the problem
        self.seperation_compute()
        self.make_select_array()
        # define z and lambda
        self.define_z_lambda()

        # initialize all the optimization model for x, get self.opt_x = dict() with the opt model for each s \in P
        self.opt_x_init()
        # iteration, including modify model, solve opt_x, update z and lambda

        for self.kappa in range(1):
            self.opt_x_modify()
            self.opt_x_solve()
            self.update_z()
            self.update_lambda()
            self.termination()

            # self.update_rho()

    def define_z_lambda(self):
        '''
        define z and lambda and initialize
        '''
        # define global variables z
        self.z_md = dict()   # R^{3 x 3}
        for i_bus in self.i_gen: self.z_md[i_bus] = np.array([[1, self.m_0[i_bus], self.d_0[i_bus]]]).T.dot( np.array([[1, self.m_0[i_bus], self.d_0[i_bus]]])  )

        # self.z_w = dict()  R^{2 x 2}
        self.z_w = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower_union for i_bus in self.i_gen], [np.array([[1, self.w_0[i_bus - 1] ]]).T.dot( np.array([[1, self.w_0[i_bus - 1] ]]) )  for ki in self.M_lower_union for i_bus in self.i_gen] ))

        # self.z_theta = dict()   R^{ 1 + number of gen_load buses in clique i_clique}
        self.z_theta = dict(zip([(ki[0][0], ki[0][1], ki[1], i_clique) for ki in self.M_lower_union for i_clique  in self.clique_tree_theta['node_gl'].keys()], 
                [np.array( [[1] + self.theta_0[(np.array(self.clique_tree_theta['node_gl'][i_clique]) - 1).tolist()].tolist() ] ).T.dot( np.array( [[1] + self.theta_0[(np.array(self.clique_tree_theta['node_gl'][i_clique]) - 1).tolist()].tolist() ] )  )   for ki in self.M_lower_union for i_clique in self.clique_tree_theta['node_gl'].keys() ] ))

        # define Lambda for each s \in P
        self.Lambda =  dict()
        for s in self.P:
            self.Lambda[s] =  dict()

            self.Lambda[s]['md'] = dict()
            for i_bus in self.i_gen: self.Lambda[s]['md'][i_bus] = np.zeros((3,3))

            self.Lambda[s]['w'] = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower[s] + self.M_upper[s] for i_bus in self.i_gen], [ np.zeros((2,2)) for ki in self.M_lower[s] + self.M_upper[s] for i_bus in self.i_gen] ))

            self.Lambda[s]['theta'] = dict(zip([(ki[0][0], ki[0][1], ki[1], i_clique) for ki in self.M_lower[s] + self.M_upper[s] for i_clique  in self.clique_tree_theta['node_gl'].keys()], [np.zeros(( 1 + len(self.clique_tree_theta['node_gl'][i_clique]) , 1 + len(self.clique_tree_theta['node_gl'][i_clique]) )) for ki in self.M_lower[s] + self.M_upper[s] for i_clique in self.clique_tree_theta['node_gl'].keys() ] ))


    def _opt_xs_init(self, s):
        '''
        initialize the optimization model for optimization x_s, for a given s
        '''
        set_k_i_s = self.Xi[s]  # set_k_i for the subproblem s

        # optimization variables
        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                # X_lmd :  [1 lm(nc + 1)  ld(nc + 1)]^T [1 lm(nc + 1)  ld(nc + 1)]
                n_lmd = 1 + 2*(nc + 1)
                self.X_lmd[(k[0], k[1], i, i_bus)] = cp.Variable(( n_lmd, n_lmd), symmetric =True) 
                self.epsilon_n = self.epsilon_n + n_lmd**2
                
            for i_bus in self.i_gen:
                # X_mdw :  [1 m(1) d(1)  w(nc + 1)]^T [1 m(1) d(1)  w(nc + 1)]  
                n_mdw = 1 + 2+ (nc + 1) 
                self.X_mdw[(k[0], k[1], i, i_bus)] = cp.Variable(( n_mdw, n_mdw), symmetric =True)  
                self.epsilon_n = self.epsilon_n +  n_mdw**2

            for i_clique  in self.clique_tree_theta['node'].keys():
                # X_theta : [1 ...]
                n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                self.X_theta[(k[0], k[1], i, i_clique)] = cp.Variable(( n_theta, n_theta), symmetric =True)  
                self.epsilon_n = self.epsilon_n + n_theta**2
                
        self.phi[s] =dict()
        self.phi[s]['md'] = dict()
        self.phi[s]['w'] = dict()
        self.phi[s]['theta'] = dict()
        for i_bus in self.i_gen: self.phi[s]['md'][i_bus] = cp.Variable()
        for ki in self.M_lower[s]:
            for i_bus in self.i_gen:
                self.phi[s]['w'][(ki[0][0], ki[0][1], ki[1], i_bus)] = cp.Variable()

        for ki in self.M_upper[s]:
            for i_bus in self.i_gen:
                self.phi[s]['w'][(ki[0][0], ki[0][1], ki[1], i_bus)] = cp.Variable()

        for ki in self.M_lower[s]:
            for i_clique in self.clique_tree_theta['node_gl'].keys():
                self.phi[s]['theta'][(ki[0][0], ki[0][1], ki[1], i_clique)] = cp.Variable()

        for ki in self.M_upper[s]:
            for i_clique in self.clique_tree_theta['node_gl'].keys():
                self.phi[s]['theta'][(ki[0][0], ki[0][1], ki[1], i_clique)] = cp.Variable()

        # index map
        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]

            i_lmd = 1
            for j in range(0, nc+1):
                self.ind_lmd[(k[0], k[1], i, j, 'lm')] = i_lmd
                i_lmd += 1
            for j in range(0, nc+1):
                self.ind_lmd[(k[0], k[1], i, j, 'ld')] = i_lmd
                i_lmd += 1

            i_mdw = 1
            self.ind_mdw[(k[0], k[1], i, 'm')] = i_mdw
            i_mdw += 1
            self.ind_mdw[(k[0], k[1], i, 'd')] = i_mdw
            i_mdw += 1
            for j in range(0, nc+1):
                self.ind_mdw[(k[0], k[1], i, j, 'w')] = i_mdw
                i_mdw += 1

            for i_clique  in self.clique_tree_theta['node'].keys():
                i_theta = 1
                for j in range(0, nc+1):
                    for i_bus in self.clique_tree_theta['node'][i_clique]:
                        self.ind_theta[(k[0], k[1], i, i_clique, j, i_bus)] = i_theta
                        i_theta += 1

        # parameters
        self.P_Lambda[s] =  dict()

        self.P_Lambda[s]['md'] = dict()
        for i_bus in self.i_gen: self.P_Lambda[s]['md'][i_bus] =  cp.Parameter((3,3),  symmetric=True)

        self.P_Lambda[s]['w'] = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower[s] + self.M_upper[s] for i_bus in self.i_gen], [ cp.Parameter((2,2),  symmetric=True) for ki in self.M_lower[s] + self.M_upper[s] for i_bus in self.i_gen] ))

        self.P_Lambda[s]['theta'] = dict(zip([(ki[0][0], ki[0][1], ki[1], i_clique) for ki in self.M_lower[s] + self.M_upper[s] for i_clique  in self.clique_tree_theta['node_gl'].keys()], [cp.Parameter(( 1 + len(self.clique_tree_theta['node_gl'][i_clique]) , 1 + len(self.clique_tree_theta['node_gl'][i_clique]) ),  symmetric=True) for ki in self.M_lower[s] + self.M_upper[s] for i_clique in self.clique_tree_theta['node_gl'].keys() ] ))


        # optimization model
        constraints = list()
        
        for (k, i) in set_k_i_s:
            ########### J_ki ###############
            print( k,i)
            [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, 0)
            nc = self.param_dd[k[0]][i][2]  # order of disturbance k[0] (the same type of disturbances is with the same 
            s_tau= self.param_dd[k[0]][i][3]  # collocation points, tau, of disturbance k[0]
            h = self.param_dd[k[0]][i][1] - self.param_dd[k[0]][i][0]  # length of time element i for disturbance k[0]

            J_theta = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                (
                  self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus_f])][self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], j1, i_bus_f)], self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], j2, i_bus_f)]]
                - self.X_theta[(k[0], k[1], i, self.branch_clique[(i_bus_f, i_bus_t)])][self.ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_f, i_bus_t)], j1, i_bus_f)], self.ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_f, i_bus_t)], j2, i_bus_t)]]
                - self.X_theta[(k[0], k[1], i, self.branch_clique[(i_bus_t, i_bus_f)])][self.ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_t, i_bus_f)], j1, i_bus_t)], self.ind_theta[(k[0], k[1], i, self.branch_clique[(i_bus_t, i_bus_f)], j2, i_bus_f)]]
                + self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus_t])][self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], j1, i_bus_t)], self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], j2, i_bus_t)]]
                )* ratio_B[i_bus_f - 1, i_bus_t - 1]  for (i_bus_f, i_bus_t) in self.ind_branch )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_w = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * sum(
                self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, j1, 'w')], self.ind_mdw[(k[0], k[1], i, j2, 'w')]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_der_w = (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * sum(
                self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, j1, 'w')], self.ind_mdw[(k[0], k[1], i, j2, 'w')]]  for i_bus in self.i_gen )  for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )

            J_ce_gen = sum( sum( self.int_l[1, nc, s_tau, j1, j2] * ( sum( self.X_lmd[(k[0], k[1], i, i_bus)][self.ind_lmd[(k[0], k[1], i, j1, 'ld')], self.ind_lmd[(k[0], k[1], i, j2, 'ld')]]  for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                +  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( self.X_lmd[(k[0], k[1], i, i_bus)][self.ind_lmd[(k[0], k[1], i, j1, 'lm')], self.ind_lmd[(k[0], k[1], i, j2, 'lm')]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) \
                    + 2 *(1/h) * sum( sum( self.int_l[3, nc, s_tau, j1, j2] * ( sum( self.X_lmd[(k[0], k[1], i, i_bus)][self.ind_lmd[(k[0], k[1], i, j1, 'lm')], self.ind_lmd[(k[0], k[1], i, j2, 'ld')]] for i_bus in self.i_gen ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) )


            J_ce_load =  (1.0/h)**2 * sum( sum( self.int_l[2, nc, s_tau, j1, j2] * ( sum( (self.D[i_bus-1])**2 * self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j1, i_bus)], self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j2, i_bus)]]  for i_bus in self.i_load ) ) for j2 in range(0, nc+1) )  for j1 in range(0, nc+1) ) 

            self.J_ki[(k,i)] =  h * (J_theta + J_w + J_der_w + J_ce_gen + J_ce_load) * self.casedata['disturbance'][k[0]][k[1]-1][-1]

            # ( 127.44357370467415   /21.00778069974123 )  1
            # ( 191.14407395083     /13.900200695381915)   2
            # ( 74.95065908260192   /26.829236366725194)   3
            # ( 46.104854448839994  /8.607371093162719 )   4
            # ( 83.1014925561       /11.757517973459112)   5
            # ( 57.599196728337276  /5.9194825682193475)   6
            ###########constrains of collocation equations###############
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                # der(theta) = w for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    constraints.append( sum( self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, r, 'w')]] )
                # der (theta) for load buses.
                for i_bus in self.i_load:
                    constraints.append( sum( self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], j, i_bus)]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) ==
                    h * (1/self.D[i_bus-1]) * (self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * (np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], r, i_bus)]] -  self.X_theta[(k[0], k[1], i, self.bus_clique[j_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[j_bus], r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) ) )
                # der w = for all generators
                for i_bus in self.i_gen:
                    constraints.append( sum( self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'm')], self.ind_mdw[(k[0], k[1], i, j, 'w')]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) == h * ( - self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'd')], self.ind_mdw[(k[0], k[1], i, r, 'w')]] +  self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                    * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( (  self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], r, i_bus)]] -  self.X_theta[(k[0], k[1], i, self.bus_clique[j_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[j_bus], r, j_bus)]]  ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all)) )


            # 0 = g for non load & generator buses
            if i == 1: ii = 1
            else: ii=0
            for r in range(ii, nc+1): # for each colloction point including r=0
                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                for i_bus in self.i_non:
                    constraints.append( 0 == self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - self.v_0[i_bus-1]**2 * delta_G[i_bus-1] -  self.v_0[i_bus-1] * sum(self.B_0[i_bus-1,j_bus-1] *  ratio_B[i_bus-1,j_bus-1] * self.v_0[j_bus-1] 
                     * ( np.sin( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) + ( ( self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], r, i_bus)]] -  self.X_theta[(k[0], k[1], i, self.bus_clique[j_bus])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[j_bus], r, j_bus)]] ) - (self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1]) ) * np.cos( self.theta_0[i_bus - 1] - self.theta_0[j_bus - 1] ) ) for j_bus in self.i_all) )
            ########### frequency constraints, resources constraints, and also constraints for m and d ################
            '''
            for r in range(1, nc+1): # for each colloction point

                [ratio_pg, ratio_pl, delta_G, ratio_B] = self.array_disturbance(k, i, r)
                
                # frequency constraints w_l(t) <= w(t) <= w_u(t) for all generator/inverter busese, i.e., i_gen
                for i_bus in self.i_gen:
                    for key_fb in self.casedata['freq_band'][k[0]]:
                        if self.param_dd[k[0]][i][0] + h * s_tau[r] > key_fb[0] and self.param_dd[k[0]][i][0] + h * s_tau[r] <= key_fb[1]:
                            constraints.append(  self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, r, 'w')]] >= (self.casedata['freq_band'][k[0]][key_fb][0] - 50) * 2 * np.pi )
                            constraints.append(  self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, r, 'w')]] <= (self.casedata['freq_band'][k[0]][key_fb][1] - 50) * 2 * np.pi )
                            break 
                # branch rotor angle difference constraints
                for (i_bus_f, i_bus_t) in self.ind_branch:
                    if ratio_B[i_bus_f-1, i_bus_t-1] != 0:                        
                        constraints.append(  (  self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus_f])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], r, i_bus_f)]] - self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus_t])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], r, i_bus_t)]] ) <= 135/180*np.pi  )
                        constraints.append(  (  self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus_f])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_f], r, i_bus_f)]] - self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus_t])][0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus_t], r, i_bus_t)]] ) >= -135/180*np.pi  )


                # resources constraints p_l <= p - m*der(w) - d*w <= p_u for all generators, and also constraints for m and d
                for i_bus in self.i_gen:
                    i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *sum( self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'm')], self.ind_mdw[(k[0], k[1], i, j, 'w')]]  *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'd')], self.ind_mdw[(k[0], k[1], i, r, 'w')]] >= self.casedata['gencontrol'][i_gc, 6][0] )
                    constraints.append( self.pg_0[i_bus-1] * ratio_pg[i_bus-1] - self.pl_0[i_bus-1] * ratio_pl[i_bus-1] - (1/h) *sum(  self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'm')], self.ind_mdw[(k[0], k[1], i, j, 'w')]] *  self.der_l[nc, s_tau, j, r] for j in range(0, nc+1) ) - self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'd')], self.ind_mdw[(k[0], k[1], i, r, 'w')]] <= self.casedata['gencontrol'][i_gc, 7][0] )

        '''
        for i_bus in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i_bus)[0]
            constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, 'm')]] >=  self.casedata['gencontrol'][i_gc, 2][0])
            constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, 'm')]] <=  self.casedata['gencontrol'][i_gc, 3][0])
            constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, 'd')]] >=  self.casedata['gencontrol'][i_gc, 4][0])
            constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][0, self.ind_mdw[(k[0], k[1], i, 'd')]] <=  self.casedata['gencontrol'][i_gc, 5][0])

        # continuity constraints of differential variable profiles across time element boundaries within the subproblem ls, i.e., x[i, 0] = x[i-1, n_c], and also initial value constraints for the first time element.
        for (k, i) in set_k_i_s:
            if i == 1: # whether time element i is the first time element, if yes, add initial value constraints
                for i_bus in self.i_gen:
                    constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][[0, self.ind_mdw[(k[0], k[1], i, 0, 'w')]],:][:,[0, self.ind_mdw[(k[0], k[1], i, 0, 'w')]]] ==   np.array([[1, self.w_0[i_bus-1]]]).T.dot(np.array([[1, self.w_0[i_bus-1]]])))
                for i_bus in self.i_gen + self.i_load + self.i_non:
                    constraints.append( self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][[0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]],:][:,[0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]]] ==  np.array([[1, self.theta_0[i_bus-1] ]]).T.dot(np.array([[1, self.theta_0[i_bus-1]]]) ) )
            elif (k, i - 1) in set_k_i_s: # whether two adjacent time elements are in the subproblem.
                nc = self.param_dd[k[0]][i-1][2]
                for i_bus in self.i_gen:
                    constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][[0, self.ind_mdw[(k[0], k[1], i, 0, 'w')]],:][:,[0, self.ind_mdw[(k[0], k[1], i, 0, 'w')]]] ==  self.X_mdw[(k[0], k[1], i-1, i_bus)][[0, self.ind_mdw[(k[0], k[1], i-1, nc, 'w')]],:][:,[0, self.ind_mdw[(k[0], k[1], i-1, nc, 'w')]]] )
                for i_bus in self.i_gen + self.i_load:
                    constraints.append(  self.X_theta[(k[0], k[1], i, self.bus_clique[i_bus])][[0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]],:][:,[0, self.ind_theta[(k[0], k[1], i, self.bus_clique[i_bus], 0, i_bus)]]] ==  self.X_theta[(k[0], k[1], i-1, self.bus_clique[i_bus])][[0, self.ind_theta[(k[0], k[1], i-1, self.bus_clique[i_bus], nc, i_bus)]],:][:,[0, self.ind_theta[(k[0], k[1], i-1, self.bus_clique[i_bus], nc, i_bus)]]] )


        # constraints for the lifting variables l_m and l_d
        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for r in range(0, nc+1):
                for i_bus in self.i_gen:
                    constraints.append( self.X_lmd[(k[0], k[1], i, i_bus)][0, self.ind_lmd[(k[0], k[1], i, r, 'lm')]] == self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'm')], self.ind_mdw[(k[0], k[1], i, r, 'w')]] )
                    constraints.append(  self.X_lmd[(k[0], k[1], i, i_bus)][0, self.ind_lmd[(k[0], k[1], i, r, 'ld')]] == self.X_mdw[(k[0], k[1], i, i_bus)][self.ind_mdw[(k[0], k[1], i, 'd')], self.ind_mdw[(k[0], k[1], i, r, 'w')]] )

        # constraints for the first element of matrices X
        for (k, i) in set_k_i_s:
            for i_bus in self.i_gen:
                constraints.append( self.X_lmd[(k[0], k[1], i, i_bus)][0, 0] == 1)
                constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][0, 0] == 1)

            for i_clique  in self.clique_tree_theta['node'].keys():
                constraints.append( self.X_theta[(k[0], k[1], i, i_clique)][0, 0] == 1 )

        # sharing variables in X_mdw for different (k,i)
        for (k, i) in set_k_i_s[1:]:
            root_k0, root_k1, root_i = set_k_i_s[0][0][0], set_k_i_s[0][0][1], set_k_i_s[0][1]
            for i_bus in self.i_gen:
                constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)][[0, self.ind_mdw[(k[0], k[1], i, 'm')], self.ind_mdw[(k[0], k[1], i, 'd')]], :][:, [0, self.ind_mdw[(k[0], k[1], i, 'm')], self.ind_mdw[(k[0], k[1], i, 'd')]]] ==
                        self.X_mdw[(root_k0, root_k1, root_i, i_bus)][[0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]], :][:, [0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]]]  )
        # sharing variables in X_theta for different cliques.
        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for edge_clique in self.clique_tree_theta['edge']:
                share_bus = list(set(self.clique_tree_theta['node'][edge_clique[0]] ).intersection(set(self.clique_tree_theta['node'][edge_clique[1]])))
                share_index_0, share_index_1 = [0], [0]
                for j in range(0, nc+1):
                    for i_bus in share_bus:
                        share_index_0.append(self.ind_theta[(k[0], k[1], i, edge_clique[0], j, i_bus)])
                        share_index_1.append(self.ind_theta[(k[0], k[1], i, edge_clique[1], j, i_bus)])

                constraints.append( self.X_theta[(k[0], k[1], i, edge_clique[0])][share_index_0, :][:,share_index_0] == self.X_theta[(k[0], k[1], i, edge_clique[1])][share_index_1, :][:,share_index_1] )

        #semidefine constraints
        for (k, i) in set_k_i_s:
            for i_bus in self.i_gen:
                constraints.append( self.X_lmd[(k[0], k[1], i, i_bus)] >> 0 )
                constraints.append( self.X_mdw[(k[0], k[1], i, i_bus)] >> 0 )

            for i_clique  in self.clique_tree_theta['node'].keys():
                constraints.append( self.X_theta[(k[0], k[1], i, i_clique)] >> 0 )


        # additional constraints for the admm

        # md coupling
        root_k0, root_k1, root_i = set_k_i_s[0][0][0], set_k_i_s[0][0][1], set_k_i_s[0][1]  # for the md coupling between subproblems
        for i_bus in self.i_gen:
            cool_v = self.X_mdw[(root_k0, root_k1, root_i, i_bus)][[0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]], :][:, [0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]]]  - ( self.P_z_md[i_bus]  -  self.P_Lambda[s]['md'][i_bus] * 1/self.rho  )

            cool_v =  cp.hstack([  cp.reshape(cool_v[0,:],(1,3)) , cp.reshape(cool_v[1:,0],(1,2))])

            cool_u = cp.hstack( [cp.reshape(self.phi[s]['md'][i_bus], (1,1)), cp.reshape(cool_v, (1,5))] )
            cool_l = cp.hstack( [ cp.reshape(cool_v, (5,1)), np.eye(5)] )
            cool = cp.vstack([cool_u, cool_l])

            constraints.append(  cool >> 0  )
        # w
        for ki in self.M_lower[s]:
            for i_bus in self.i_gen:
                cool_v = self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]]] - ( self.P_z_w[(ki[0][0], ki[0][1], ki[1], i_bus)] - self.P_Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ] * 1/self.rho     )
    
                cool_u = cp.hstack( [cp.reshape(self.phi[s]['w'][(ki[0][0], ki[0][1], ki[1], i_bus)], (1,1)), cp.reshape(cool_v, (1,4))] )
                cool_l = cp.hstack( [ cp.reshape(cool_v, (4,1)), np.eye(4)] )
                cool = cp.vstack([cool_u, cool_l])
                constraints.append(  cool >> 0  )

        for ki in self.M_upper[s]:
            for i_bus in self.i_gen:
                cool_v = self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1],  self.param_dd[ki[0][0]][ki[1]][2], 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], 'w')]]] - ( self.P_z_w[(ki[0][0], ki[0][1], ki[1] + 1, i_bus)] - self.P_Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ] * 1/self.rho     )

                cool_u = cp.hstack( [cp.reshape(self.phi[s]['w'][(ki[0][0], ki[0][1], ki[1], i_bus)], (1,1)), cp.reshape(cool_v, (1,4))] )
                cool_l = cp.hstack( [ cp.reshape(cool_v, (4,1)), np.eye(4)] )
                cool = cp.vstack([cool_u, cool_l])
                constraints.append(  cool >> 0  )
        # theta
        for ki in self.M_lower[s]:
            for i_clique in self.clique_tree_theta['node_gl'].keys():
                ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, 0, i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]
                cool_v = self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique]  - ( self.P_z_theta[(ki[0][0], ki[0][1], ki[1], i_clique)] -   self.P_Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ] * 1/self.rho)

                n_clique_gl  = len(self.clique_tree_theta['node_gl'][i_clique])
                n_select = (n_clique_gl + 1) + 2*n_clique_gl

                cool_v = cp.hstack([cp.reshape( cp.diag(cool_v), (1,n_clique_gl+1) ), cp.reshape( cool_v[0,1:], (1,n_clique_gl) ), cp.reshape( cool_v[1:,0], (1,n_clique_gl) )])
                cool_u = cp.hstack( [cp.reshape(self.phi[s]['theta'][(ki[0][0], ki[0][1], ki[1], i_clique)], (1,1)), cp.reshape(cool_v, (1, n_select ))] )
                cool_l = cp.hstack( [ cp.reshape(cool_v, (n_select,1)), np.eye(n_select)] )

                cool = cp.vstack([cool_u, cool_l])
                constraints.append(  cool >> 0  )

        for ki in self.M_upper[s]:
            for i_clique in self.clique_tree_theta['node_gl'].keys():
                ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, self.param_dd[ki[0][0]][ki[1]][2], i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]
                cool_v = self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique]  - ( self.P_z_theta[(ki[0][0], ki[0][1], ki[1] + 1, i_clique)] -   self.P_Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ] * 1/self.rho)

                n_clique_gl  = len(self.clique_tree_theta['node_gl'][i_clique])
                n_select = (n_clique_gl + 1) + 2*n_clique_gl
 
                cool_v = cp.hstack([cp.reshape( cp.diag(cool_v), (1,n_clique_gl+1) ), cp.reshape( cool_v[0,1:], (1,n_clique_gl) ), cp.reshape( cool_v[1:,0], (1,n_clique_gl) )])
                cool_u = cp.hstack( [cp.reshape(self.phi[s]['theta'][(ki[0][0], ki[0][1], ki[1], i_clique)], (1,1)), cp.reshape(cool_v, (1, n_select ))] )
                cool_l = cp.hstack( [ cp.reshape(cool_v, (n_select,1)), np.eye(n_select)] )

                cool = cp.vstack([cool_u, cool_l])
                constraints.append(  cool >> 0  )


        ###### objective function ###############
        self.J[s]  = sum(self.J_ki[(k,i)]  for (k, i) in set_k_i_s)

        self.phi_sum[s] = sum( self.phi[s]['md'].values() ) + sum(self.phi[s]['w'].values()) + sum( self.phi[s]['theta'].values())

        self.J_phi[s] = self.J[s] + (self.rho/2.0) * self.phi_sum[s] 

        objective = cp.Minimize( self.J_phi[s] )
        self.opt_x[s] = cp.Problem(objective, constraints)

    def opt_x_init(self):
        '''
        initialize model opt_x_s for all s \in P
        '''
        # optimization variable
        self.X_lmd = dict()
        self.X_mdw = dict()
        self.X_theta = dict()

        self.phi = dict()  # slack variable

        # index map
        self.ind_lmd = dict()
        self.ind_mdw = dict()
        self.ind_theta = dict()

        # objective and problem
        self.J_ki = dict()
        self.J = dict()
        self.phi_sum = dict()
        self.J_phi = dict()  # J + rho/2 * phi
        self.opt_x = dict()

        # Parameters in optimization model (Lambda and Z_a)
        self.P_Lambda =  dict()  # generate in each s in P

        self.P_z_md = dict()   #generate for all
        for i_bus in self.i_gen: self.P_z_md[i_bus] = cp.Parameter((3,3),  symmetric=True)

        self.P_z_w = dict(zip([(ki[0][0], ki[0][1], ki[1], i_bus) for ki in self.M_lower_union for i_bus in self.i_gen], [cp.Parameter((2,2),  symmetric=True) for ki in self.M_lower_union for i_bus in self.i_gen] )) #generate for all

        self.P_z_theta = dict(zip([(ki[0][0], ki[0][1], ki[1], i_clique) for ki in self.M_lower_union for i_clique  in self.clique_tree_theta['node_gl'].keys()], 
                [cp.Parameter(( 1 + len(self.clique_tree_theta['node_gl'][i_clique]) , 1 + len(self.clique_tree_theta['node_gl'][i_clique]) ),  symmetric=True)  for ki in self.M_lower_union for i_clique in self.clique_tree_theta['node_gl'].keys() ] )) #generate for all

        self.epsilon_n = 0

        # generate for each s in P
        for s in self.P:
            self._opt_xs_init(s)


    def _opt_xs_modify(self, s):
        '''
        massign the values of P_Lambda
        '''

        for i_bus in self.i_gen: self.P_Lambda[s]['md'][i_bus].value =  self.Lambda[s]['md'][i_bus]

        for ki in self.M_lower[s] + self.M_upper[s]:
            for i_bus in self.i_gen:
                self.P_Lambda[s]['w'][(ki[0][0], ki[0][1], ki[1], i_bus)].value = self.Lambda[s]['w'][(ki[0][0], ki[0][1], ki[1], i_bus)]
        
        for ki in self.M_lower[s] + self.M_upper[s]:
            for i_clique  in self.clique_tree_theta['node_gl'].keys():
                self.P_Lambda[s]['theta'][(ki[0][0], ki[0][1], ki[1], i_clique)].value = self.Lambda[s]['theta'][(ki[0][0], ki[0][1], ki[1], i_clique)]
        

    def opt_x_modify(self):
        '''
        modify opt_xs, i.e, assign the values of parameters P_Lambda and P_z
        '''
        for s in self.P: self._opt_xs_modify(s)

        for i_bus in self.i_gen: self.P_z_md[i_bus].value = self.z_md[i_bus]
        for ki in self.M_lower_union:
            for i_bus in self.i_gen:
                self.P_z_w[(ki[0][0], ki[0][1], ki[1], i_bus)].value = self.z_w[(ki[0][0], ki[0][1], ki[1], i_bus)]
        for ki in self.M_lower_union:
            for i_clique  in self.clique_tree_theta['node_gl'].keys():
                self.P_z_theta[(ki[0][0], ki[0][1], ki[1], i_clique)].value = self.z_theta[(ki[0][0], ki[0][1], ki[1], i_clique)]

    def opt_x_solve(self):
        '''
        solve opt_x for all s \in P, without multiprocess
        '''
        for s in self.P:
            self.opt_x[s].solve(solver='MOSEK', verbose = True, mosek_params={mosek.iparam.num_threads: 1})


    def solve_single_llin(self, s):
        '''
        solve one opt_x[s]
        '''

        self.opt_x[s].solve(solver='MOSEK', verbose = False)


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
        self.z_md_pre = deepcopy(self.z_md)
        self.z_w_pre = deepcopy( self.z_w)
        self.z_theta_pre = deepcopy(self.z_theta)

        #
        for i_bus in self.i_gen:
            self.z_md[i_bus] = (1.0/self.N_s) * sum( self.X_mdw[(self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1], i_bus)][[0, self.ind_mdw[(self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1], 'm')], self.ind_mdw[(self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1], 'd')]], :][:, [0, self.ind_mdw[(self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1], 'm')], self.ind_mdw[(self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1], 'd')]]].value  for s in self.P )
        
        
        for ki in self.M_lower_union:
            nc_1 = self.param_dd[ki[0][0]][ki[1] - 1][2]  # n_c  for (k, i - 1)
            
            for i_bus in self.i_gen:
                self.z_w[(ki[0][0], ki[0][1], ki[1], i_bus)] =  0.5 * ( self.X_mdw[(ki[0][0], ki[0][1], ki[1] - 1, i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1] - 1, nc_1, 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1]-1, nc_1, 'w')]]].value + self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]]].value )
            
            for i_clique in self.clique_tree_theta['node_gl'].keys():
                ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, 0, i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]

                ind_gl_clique_1 = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1] - 1, i_clique, nc_1, i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]

                self.z_theta[(ki[0][0], ki[0][1], ki[1], i_clique)] = 0.5 * (self.X_theta[(ki[0][0], ki[0][1], ki[1] - 1, i_clique)][ind_gl_clique_1,:][:,ind_gl_clique_1].value + self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique].value)
            

    def update_lambda(self):
        '''
        update self.Lambda_ for each iteration
        '''
        # store the previous Lambda before update for compute dual residual r, unlike s, it not neessary

        self.Lambda_pre = deepcopy(self.Lambda)

        for s in self.P:
            # md
            root_k0, root_k1, root_i = self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1]  # for the md coupling between subproblems
            for i_bus in self.i_gen:
                self.Lambda[s]['md'][i_bus] = self.Lambda[s]['md'][i_bus] + ( self.X_mdw[(root_k0, root_k1, root_i, i_bus)][[0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]], :][:, [0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]]].value  - self.z_md[i_bus] ) * self.rho
            # w
            for ki in self.M_lower[s]:
                for i_bus in self.i_gen:
                    self.Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ] = self.Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ]  + (self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]]].value - self.z_w[(ki[0][0], ki[0][1], ki[1], i_bus)] ) * self.rho

            for ki in self.M_upper[s]:
                for i_bus in self.i_gen:
                    self.Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ] =  self.Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ] +  ( self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1],  self.param_dd[ki[0][0]][ki[1]][2], 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], 'w')]]].value - self.z_w[(ki[0][0], ki[0][1], ki[1] + 1, i_bus)] ) * self.rho 

            # theta
            for ki in self.M_lower[s]:
                for i_clique in self.clique_tree_theta['node_gl'].keys():
                    ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, 0, i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]
                    self.Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ] = self.Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ] + (                
                    self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique].value  - self.z_theta[(ki[0][0], ki[0][1], ki[1], i_clique)] ) * self.rho

            for ki in self.M_upper[s]:
                for i_clique in self.clique_tree_theta['node_gl'].keys():
                    ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, self.param_dd[ki[0][0]][ki[1]][2], i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]
                    self.Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ]  = self.Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ] + (                
                    self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique].value  -  self.z_theta[(ki[0][0], ki[0][1], ki[1] + 1, i_clique)] ) * self.rho

    def _dict_sq(self, dict_sq_1, dict_sq_2):

        array_all = np.square( np.array(itemgetter(*dict_sq_1.keys())(dict_sq_1)) - np.array(itemgetter(*dict_sq_1.keys())(dict_sq_2)))
        sum_all = sum(array_one.sum()  for array_one in array_all)
        return sum_all

    def _dict_sq_md(self, dict_sq_1, dict_sq_2):

        array_all = np.square( np.array(itemgetter(*dict_sq_1.keys())(dict_sq_1)) - np.array(itemgetter(*dict_sq_1.keys())(dict_sq_2)))
        sum_all = sum(array_one[0,:].sum()  for array_one in array_all) + sum(array_one[1:,0].sum()  for array_one in array_all)
        return sum_all

    def _dict_sq_theta(self, dict_sq_1, dict_sq_2):

        array_all = np.square( (np.array(itemgetter(*dict_sq_1.keys())(dict_sq_1)) - np.array(itemgetter(*dict_sq_1.keys())(dict_sq_2))) * 
        np.array(itemgetter(*dict_sq_1.keys())( self.array_select_sdp)) )
        sum_all = sum(array_one.sum()  for array_one in array_all)
        return sum_all



    def _epsilon_great(self):

        x_norm2, z_norm2, l_norm2 = 0,0,0

        for s in self.P:
            # md
            root_k0, root_k1, root_i = self.Xi[s][0][0][0], self.Xi[s][0][0][1], self.Xi[s][0][1]  # for the md coupling between subproblems
            for i_bus in self.i_gen:
                l_norm2 = l_norm2 + np.linalg.norm(self.Lambda[s]['md'][i_bus] * self.array_select_md)
                x_norm2 = x_norm2 + np.linalg.norm(self.X_mdw[(root_k0, root_k1, root_i, i_bus)][[0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]], :][:, [0, self.ind_mdw[(root_k0, root_k1, root_i, 'm')], self.ind_mdw[(root_k0, root_k1, root_i, 'd')]]].value * self.array_select_md )
                z_norm2 = z_norm2 + np.linalg.norm(self.z_md[i_bus] * self.array_select_md)

            if self.sep_mode != 'k':
                # w
                for ki in self.M_lower[s]:
                    for i_bus in self.i_gen:
                        l_norm2 = l_norm2 + np.linalg.norm(self.Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ])**2
                        x_norm2 = x_norm2 + np.linalg.norm(self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], 0, 'w')]]].value)**2
                        z_norm2 = z_norm2 + np.linalg.norm(self.z_w[(ki[0][0], ki[0][1], ki[1], i_bus)])**2

                for ki in self.M_upper[s]:
                    for i_bus in self.i_gen:
                        l_norm2 = l_norm2 + np.linalg.norm(self.Lambda[s]['w'][ (ki[0][0], ki[0][1], ki[1], i_bus) ])**2
                        x_norm2 = x_norm2 + np.linalg.norm(self.X_mdw[(ki[0][0], ki[0][1], ki[1], i_bus)][[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1],  self.param_dd[ki[0][0]][ki[1]][2], 'w')]],:][:,[0, self.ind_mdw[(ki[0][0], ki[0][1], ki[1], self.param_dd[ki[0][0]][ki[1]][2], 'w')]]].value)**2
                        z_norm2 = z_norm2 + np.linalg.norm(self.z_w[(ki[0][0], ki[0][1], ki[1] + 1, i_bus)])**2


                # theta
                for ki in self.M_lower[s]:
                    for i_clique in self.clique_tree_theta['node_gl'].keys():
                        ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, 0, i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]
                        l_norm2 = l_norm2 + np.linalg.norm(self.Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ]  * self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ])**2       
                        x_norm2 = x_norm2 + np.linalg.norm(self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique].value  * self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ] )**2
                        z_norm2 = z_norm2 + np.linalg.norm(self.z_theta[(ki[0][0], ki[0][1], ki[1], i_clique)] * self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ])**2

                for ki in self.M_upper[s]:
                    for i_clique in self.clique_tree_theta['node_gl'].keys():
                        ind_gl_clique = [0] + [self.ind_theta[(ki[0][0], ki[0][1], ki[1], i_clique, self.param_dd[ki[0][0]][ki[1]][2], i_bus)]  for i_bus in self.clique_tree_theta['node_gl'][i_clique] ]
                        l_norm2 = l_norm2 + np.linalg.norm(self.Lambda[s]['theta'][ (ki[0][0], ki[0][1], ki[1], i_clique) ] * self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ])**2         
                        x_norm2 = x_norm2 + np.linalg.norm(self.X_theta[(ki[0][0], ki[0][1], ki[1], i_clique)][ind_gl_clique,:][:,ind_gl_clique].value * self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ] )**2
                        z_norm2 = z_norm2 + np.linalg.norm(self.z_theta[(ki[0][0], ki[0][1], ki[1] + 1, i_clique)] * self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ])**2

        return x_norm2**0.5, z_norm2**0.5, l_norm2**0.5

    def termination(self):

        if self.kappa == 0:
            self.r_norm_2 = dict()
            self.s_norm_2 = dict()

            self.J_iter = dict()
            self.J_phi_iter=dict()
            self.solve_time = dict()

            self.epsilon_pri = dict()
            self.epsilon_dual = dict()
            self.term_epsilon_pri = dict()
            self.term_epsilon_dual = dict()

            self.epsilon_p = (2 + 3) * self.N_s * len(self.i_gen)  + 2**2 * 2 * len(self.M_lower_union) * len(self.i_gen) + sum( (len(tong) + 1) + 2*len(tong) for tong in self.clique_tree_theta['node_gl'].values()) * len(self.M_lower_union)  * 2
            #self.epsilon_n count in _opt_xs_init

        if self.sep_mode != 'k':
            self.r_norm_2[self.kappa] = (1.0/self.rho) * (sum([self._dict_sq_md(self.Lambda[s]['md'], self.Lambda_pre[s]['md'])  +  self._dict_sq(self.Lambda[s]['w'], self.Lambda_pre[s]['w']) +  self._dict_sq_theta(self.Lambda[s]['theta'], self.Lambda_pre[s]['theta']) for s in self.P]))**0.5
            self.s_norm_2[self.kappa] =  self.rho * (self.N_s * self._dict_sq_md(self.z_md, self.z_md_pre) +  2 * self._dict_sq(self.z_w, self.z_w_pre) + 2 *self._dict_sq_theta(self.z_theta, self.z_theta_pre) )**0.5
           
        else:
            self.r_norm_2[self.kappa] = (1.0/self.rho) * (sum([self._dict_sq_md(self.Lambda[s]['md'], self.Lambda_pre[s]['md']) for s in self.P]))**0.5
            self.s_norm_2[self.kappa] =  self.rho * (self.N_s * self._dict_sq_md(self.z_md, self.z_md_pre) )**0.5            

        ######################## 
        x_norm2, z_norm2, l_norm2  = self._epsilon_great()
        self.epsilon_pri[self.kappa] =  ( self.epsilon_p )**0.5 * self.epsilon_abs + self.epsilon_rel * max(x_norm2, z_norm2)
        self.epsilon_dual[self.kappa] =  ( self.epsilon_n )**0.5 * self.epsilon_abs + self.epsilon_rel * l_norm2

        self.term_epsilon_pri[self.kappa] = max(x_norm2, z_norm2)
        self.term_epsilon_dual[self.kappa] = l_norm2

        self.J_iter[self.kappa] = sum(self.J[s].value for s in self.P)
        self.J_phi_iter[self.kappa] = sum(self.J_phi[s].value for s in self.P)
        self.solve_time[self.kappa] = [self.opt_x[s].solution.attr['solve_time'] for s in self.P]

        print(self.kappa, ':    ', self.r_norm_2[self.kappa], self.s_norm_2[self.kappa],  self.J_iter[self.kappa] , self.solve_time[self.kappa])
        print(self.kappa, ':    ', self.epsilon_pri[self.kappa],  self.epsilon_dual[self.kappa] , self.rho, self.J_phi_iter[self.kappa] )

    def update_rho(self):

        if self.r_norm_2[self.kappa] >  self.rho_mu * self.s_norm_2[self.kappa]:
            self.rho =  self.rho * self.rho_tau
        elif  self.s_norm_2[self.kappa] >  self.rho_mu * self.r_norm_2[self.kappa]:
            self.rho =  self.rho * (1/self.rho_tau)
        else:
            pass
#=============================The following is the block for function "opt_ln_sdp_small_admm_fs" ant its subfunction================================#

    def opt_lin_sdp_small_admm_fs(self, rho = 1, rho_fs = 1, sep_mode = 'i', N_s = 4, N_s_k = 4, N_s_i = 4):

        # parameters of admm
        self.rho = rho
        self.rho_fs = rho_fs ##
        self.sep_mode = sep_mode
        self.N_s = N_s
        self.N_s_k = N_s_k
        self.N_s_i = N_s_i

        self.rho_tau  = 1.2
        self.rho_mu  = 10

        self.rho_tau_fs  = 1.2 ##
        self.rho_mu_fs  = 10 ##

        self.epsilon_abs = 1e-6
        self.epsilon_rel = 1e-6

        self.epsilon_abs_fs = 1e-6  ##
        self.epsilon_rel_fs = 1e-6  ##

        # separate the problem
        self.seperation_compute() 
        self.make_select_array()
        # define z and lambda
        self.define_z_lambda_fs()  ##

        # initialize all the optimization model for x, get self.opt_x = dict() with the opt model for each s \in P
        self.opt_x_init_fs()
        # iteration, including modify model, solve opt_x, update z and lambda

        for self.kappa in range(2000):
            self.opt_x_modify_fs()
            self.opt_x_solve_fs()
            self.update_z_fs()
            self.update_lambda_fs()
            self.termination_fs()

            #self.update_rho_fs()

    def define_z_lambda_fs(self):
        '''
        define Y and lambda_fs and initialize
        '''
        self.define_z_lambda() 

        self.Y_lmd = dict()
        self.Y_mdw = dict()
        self.Y_theta = dict()

        # define the auxiliary variable Y. Y is with the same shape as X
        for s in self.P:
            set_k_i_s = self.Xi[s]  # set_k_i for the subproblem s

            # optimization variables
            for (k, i) in set_k_i_s:
                nc = self.param_dd[k[0]][i][2]
                for i_bus in self.i_gen:
                    n_lmd = 1 + 2*(nc + 1)
                    self.Y_lmd[(k[0], k[1], i, i_bus)] =  np.zeros((n_lmd, n_lmd))
                    
                for i_bus in self.i_gen:
                    n_mdw = 1 + 2+ (nc + 1) 
                    self.Y_mdw[(k[0], k[1], i, i_bus)] = np.zeros(( n_mdw, n_mdw))

                for i_clique  in self.clique_tree_theta['node'].keys():
                    # X_theta : [1 ...]
                    n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                    self.Y_theta[(k[0], k[1], i, i_clique)] = np.zeros(( n_theta, n_theta))  

        # define the lambda variable
        self.Lambda_fs =  dict()
        self.Lambda_fs['lmd'] =  dict()
        self.Lambda_fs['mdw'] =  dict()
        self.Lambda_fs['theta'] =  dict()
        for s in self.P:
            set_k_i_s = self.Xi[s]  # set_k_i for the subproblem s

            # optimization variables
            for (k, i) in set_k_i_s:
                nc = self.param_dd[k[0]][i][2]
                for i_bus in self.i_gen:
                    n_lmd = 1 + 2*(nc + 1)
                    self.Lambda_fs['lmd'][(k[0], k[1], i, i_bus)] =  np.zeros((n_lmd, n_lmd))
                    
                for i_bus in self.i_gen:
                    n_mdw = 1 + 2+ (nc + 1) 
                    self.Lambda_fs['mdw'][(k[0], k[1], i, i_bus)] = np.zeros(( n_mdw, n_mdw))

                for i_clique  in self.clique_tree_theta['node'].keys():
                    # X_theta : [1 ...]
                    n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                    self.Lambda_fs['theta'][(k[0], k[1], i, i_clique)] = np.zeros(( n_theta, n_theta))  
        

    def _opt_xs_init_fs(self, s):
        '''
        initialize the optimization model for optimization x_s, for a given s
        '''

        set_k_i_s = self.Xi[s]  # set_k_i for the subproblem s

        self.phi_fs[s] = dict()    #cp.Variable()
        self.phi_fs[s]['lmd'] = dict()
        self.phi_fs[s]['mdw'] = dict()
        self.phi_fs[s]['theta'] = dict()

        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                # X_lmd :  [1 lm(nc + 1)  ld(nc + 1)]^T [1 lm(nc + 1)  ld(nc + 1)]
                n_lmd = 1 + 2*(nc + 1)
                for i_row in range(0, n_lmd):
                    self.phi_fs[s]['lmd'][k, i, i_bus, i_row] = cp.Variable()
            for i_bus in self.i_gen:
                n_mdw = 1 + 2+ (nc + 1)     
                for i_row in range(0, n_mdw):
                    self.phi_fs[s]['mdw'][k, i, i_bus, i_row] = cp.Variable()

            for i_clique  in self.clique_tree_theta['node'].keys():
                n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                for i_row in range(0, n_theta):
                    self.phi_fs[s]['theta'][k, i, i_clique, i_row] = cp.Variable()

        # parameters
        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                n_lmd = 1 + 2*(nc + 1)
                self.P_Lambda_fs['lmd'][(k[0], k[1], i, i_bus)] =  cp.Parameter((n_lmd, n_lmd),  symmetric=True)
                
            for i_bus in self.i_gen:
                n_mdw = 1 + 2+ (nc + 1) 
                self.P_Lambda_fs['mdw'][(k[0], k[1], i, i_bus)] = cp.Parameter(( n_mdw, n_mdw),  symmetric=True)

            for i_clique  in self.clique_tree_theta['node'].keys():
                # X_theta : [1 ...]
                n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                self.P_Lambda_fs['theta'][(k[0], k[1], i, i_clique)] = cp.Parameter(( n_theta, n_theta ),  symmetric=True)

        # parameter P_Y_
        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                n_lmd = 1 + 2*(nc + 1)
                self.P_Y_lmd[(k[0], k[1], i, i_bus)] =  cp.Parameter((n_lmd, n_lmd),  symmetric=True)
                
            for i_bus in self.i_gen:
                n_mdw = 1 + 2+ (nc + 1) 
                self.P_Y_mdw[(k[0], k[1], i, i_bus)] =  cp.Parameter(( n_mdw, n_mdw),  symmetric=True)

            for i_clique  in self.clique_tree_theta['node'].keys():
                # X_theta : [1 ...]
                n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                self.P_Y_theta[(k[0], k[1], i, i_clique)] = cp.Parameter(( n_theta, n_theta ),  symmetric=True)

        # additional constraints for the admm with feasibility considered
        constraints_fs = self.opt_x[s].constraints



        for (k, i) in set_k_i_s:
            nc = self.param_dd[k[0]][i][2]
            for i_bus in self.i_gen:
                # X_lmd :  [1 lm(nc + 1)  ld(nc + 1)]^T [1 lm(nc + 1)  ld(nc + 1)]
                n_lmd = 1 + 2*(nc + 1)
                for i_row in range(0, n_lmd):
                    cool_v = self.X_lmd[(k[0], k[1], i, i_bus)][i_row, :] - ( self.P_Y_lmd[(k[0], k[1], i, i_bus)][i_row, :] -  self.P_Lambda_fs['lmd'][(k[0], k[1], i, i_bus)][i_row, :]  * 1/self.rho_fs )

                    cool_u = cp.hstack( [cp.reshape(self.phi_fs[s]['lmd'][k, i, i_bus, i_row], (1,1)), cp.reshape(cool_v, (1,n_lmd))] )
                    cool_l = cp.hstack( [ cp.reshape(cool_v, (n_lmd,1)), np.eye(n_lmd)] )
                    cool = cp.vstack([cool_u, cool_l])
                    constraints_fs.append(  cool >> 0  )
            for i_bus in self.i_gen:
                n_mdw = 1 + 2+ (nc + 1)     
                for i_row in range(0, n_mdw):
                    cool_v = self.X_mdw[(k[0], k[1], i, i_bus)][i_row, :] - ( self.P_Y_mdw[(k[0], k[1], i, i_bus)][i_row, :] -  self.P_Lambda_fs['mdw'][(k[0], k[1], i, i_bus)][i_row, :]  * 1/self.rho_fs )

                    cool_u = cp.hstack( [cp.reshape(self.phi_fs[s]['mdw'][k, i, i_bus, i_row], (1,1)), cp.reshape(cool_v, (1,n_mdw))] )
                    cool_l = cp.hstack( [ cp.reshape(cool_v, (n_mdw,1)), np.eye(n_mdw)] )
                    cool = cp.vstack([cool_u, cool_l])
                    constraints_fs.append(  cool >> 0  )

            for i_clique  in self.clique_tree_theta['node'].keys():
                n_theta = 1 + len(self.clique_tree_theta['node'][i_clique]) * (1 + nc)
                for i_row in range(0, n_theta):
                    cool_v = self.X_theta[(k[0], k[1], i, i_clique)][i_row, :] - ( self.P_Y_theta[(k[0], k[1], i, i_clique)][i_row, :] -  self.P_Lambda_fs['theta'][(k[0], k[1], i, i_clique)][i_row, :]  * 1/self.rho_fs )

                    cool_u = cp.hstack( [cp.reshape(self.phi_fs[s]['theta'][k, i, i_clique, i_row], (1,1)), cp.reshape(cool_v, (1,n_theta))] )
                    cool_l = cp.hstack( [ cp.reshape(cool_v, (n_theta,1)), np.eye(n_theta)] )
                    cool = cp.vstack([cool_u, cool_l])
                    constraints_fs.append(  cool >> 0  )
  
        ###### objective function ###############

        self.phi_fs_sum[s] = sum(  self.phi_fs[s]['lmd'].values() ) + sum(  self.phi_fs[s]['mdw'].values() )+ sum(  self.phi_fs[s]['theta'].values() )

        self.J_phi_fs[s] = self.J_phi[s] + (self.rho_fs/2.0) * self.phi_fs_sum[s]
        objective = cp.Minimize( self.J_phi_fs[s] )
        self.opt_x[s] = cp.Problem(objective, constraints_fs)

    def opt_x_init_fs(self):
        '''
        initialize model opt_x_s for all s \in P
        '''

        self.opt_x_init()

        self.phi_fs = dict()  ##  # slack variable

        self.phi_fs_sum = dict()

        self.J_phi_fs = dict()  ## J_phi + rho_fs/2 * phi_fs

        self.P_Lambda_fs =  dict()  ##
        self.P_Lambda_fs['lmd'] =  dict()  ##
        self.P_Lambda_fs['mdw'] =  dict()  ##
        self.P_Lambda_fs['theta'] =  dict()  ##

        self.P_Y_lmd = dict()  ##
        self.P_Y_mdw = dict()  ##
        self.P_Y_theta = dict()  ##

        # generate for each s in P
        for s in self.P:
            self._opt_xs_init_fs(s)


    def _opt_xs_modify_fs(self, s):
        '''
        massign the values of P_Lambda
        '''

        set_k_i_s = self.Xi[s]

        for (k, i) in set_k_i_s:
            for i_bus in self.i_gen:
                self.P_Lambda_fs['lmd'][(k[0], k[1], i, i_bus)].value =  self.Lambda_fs['lmd'][(k[0], k[1], i, i_bus)]
                
            for i_bus in self.i_gen:
                self.P_Lambda_fs['mdw'][(k[0], k[1], i, i_bus)].value = self.Lambda_fs['mdw'][(k[0], k[1], i, i_bus)]

            for i_clique  in self.clique_tree_theta['node'].keys():
                self.P_Lambda_fs['theta'][(k[0], k[1], i, i_clique)].value = self.Lambda_fs['theta'][(k[0], k[1], i, i_clique)]

        for (k, i) in set_k_i_s:
            for i_bus in self.i_gen:
                self.P_Y_lmd[(k[0], k[1], i, i_bus)].value =  self.Y_lmd[(k[0], k[1], i, i_bus)]
                
            for i_bus in self.i_gen:
                self.P_Y_mdw[(k[0], k[1], i, i_bus)].value =  self.Y_mdw[(k[0], k[1], i, i_bus)]

            for i_clique  in self.clique_tree_theta['node'].keys():
                self.P_Y_theta[(k[0], k[1], i, i_clique)].value =  self.Y_theta[(k[0], k[1], i, i_clique)]


    def opt_x_modify_fs(self):
        '''
        modify opt_xs, i.e, assign the values of parameters P_Lambda and P_z
        '''
        self.opt_x_modify()

        for s in self.P: self._opt_xs_modify_fs(s)

    def opt_x_solve_fs(self):
        '''
        solve opt_x for all s \in P, without multiprocess
        '''
        for s in self.P:
            self.opt_x[s].solve(solver='MOSEK', verbose = False)


    def _find_s(self, ki):
        '''
        return s which ki in self.Xi[s]
        '''
        for s in self.P:
            if ki in self.Xi[s]:
                return s
                break

    def update_z_fs(self):
        '''
        update self.z_ for each iteration
        '''
        self.update_z()


        self.Y_lmd_pre  =   deepcopy(self.Y_lmd  )
        self.Y_mdw_pre  =   deepcopy(self.Y_mdw  )
        self.Y_theta_pre  = deepcopy(self.Y_theta)

        for s in self.P:
            set_k_i_s = self.Xi[s]

            for (k, i) in set_k_i_s:
                for i_bus in self.i_gen:
                    eig_val, eig_vec = LA.eigh(self.X_lmd[(k[0], k[1], i, i_bus)].value + self.Lambda_fs['lmd'][(k[0], k[1], i, i_bus)] * 1/self.rho_fs )
                    eig_val_1 = eig_val[-1]
                    eig_vec_1 = np.array([eig_vec[:, -1]])
                    self.Y_lmd[(k[0], k[1], i, i_bus)] = ( eig_val_1 * eig_vec_1.T * eig_vec_1 ).T

                for i_bus in self.i_gen:
                    eig_val, eig_vec = LA.eigh(self.X_mdw[(k[0], k[1], i, i_bus)].value + self.Lambda_fs['mdw'][(k[0], k[1], i, i_bus)] * 1/self.rho_fs )
                    eig_val_1 = eig_val[-1]
                    eig_vec_1 = np.array([eig_vec[:, -1]])
                    self.Y_mdw[(k[0], k[1], i, i_bus)] = ( eig_val_1 * eig_vec_1.T * eig_vec_1 ).T

                for i_clique  in self.clique_tree_theta['node'].keys():
                    eig_val, eig_vec = LA.eigh(self.X_theta[(k[0], k[1], i, i_clique)].value + self.Lambda_fs['theta'][(k[0], k[1], i, i_clique)] * 1/self.rho_fs )
                    eig_val_1 = eig_val[-1]
                    eig_vec_1 = np.array([eig_vec[:, -1]])
                    self.Y_theta[(k[0], k[1], i, i_clique)] = ( eig_val_1 * eig_vec_1.T * eig_vec_1 ).T

###############

    def update_lambda_fs(self):
        '''
        update self.Lambda_ for each iteration
        '''
        # store the previous Lambda before update for compute dual residual r, unlike s, it not neessary

        self.update_lambda()

        self.Lambda_fs_pre = deepcopy(self.Lambda_fs)

        for s in self.P:
            set_k_i_s = self.Xi[s]

            for (k, i) in set_k_i_s:
                for i_bus in self.i_gen:
                    self.Lambda_fs['lmd'][(k[0], k[1], i, i_bus)] = self.Lambda_fs['lmd'][(k[0], k[1], i, i_bus)] + self.rho_fs * ( self.X_lmd[(k[0], k[1], i, i_bus)].value -  self.Y_lmd[(k[0], k[1], i, i_bus)] )

                for i_bus in self.i_gen:
                    self.Lambda_fs['mdw'][(k[0], k[1], i, i_bus)] = self.Lambda_fs['mdw'][(k[0], k[1], i, i_bus)] + self.rho_fs * ( self.X_mdw[(k[0], k[1], i, i_bus)].value -  self.Y_mdw[(k[0], k[1], i, i_bus)] )

                for i_clique  in self.clique_tree_theta['node'].keys():
                    self.Lambda_fs['theta'][(k[0], k[1], i, i_clique)] = self.Lambda_fs['theta'][(k[0], k[1], i, i_clique)] + self.rho_fs * ( self.X_theta[(k[0], k[1], i, i_clique)].value -  self.Y_theta[(k[0], k[1], i, i_clique)] )


    def _epsilon_great_fs(self):

        x_norm2_fs, z_norm2_fs, l_norm2_fs= 0,0,0

        for s in self.P:
            set_k_i_s = self.Xi[s]

            for (k, i) in set_k_i_s:
                for i_bus in self.i_gen:
                    x_norm2_fs = x_norm2_fs + np.linalg.norm( self.X_lmd[(k[0], k[1], i, i_bus)].value )
                    z_norm2_fs = z_norm2_fs + np.linalg.norm( self.Y_lmd[(k[0], k[1], i, i_bus)] )
                    l_norm2_fs = l_norm2_fs + np.linalg.norm( self.Lambda_fs['lmd'][(k[0], k[1], i, i_bus)] )

                for i_bus in self.i_gen:
                    x_norm2_fs = x_norm2_fs + np.linalg.norm( self.X_mdw[(k[0], k[1], i, i_bus)].value )
                    z_norm2_fs = z_norm2_fs + np.linalg.norm( self.Y_mdw[(k[0], k[1], i, i_bus)] )
                    l_norm2_fs = l_norm2_fs + np.linalg.norm( self.Lambda_fs['mdw'][(k[0], k[1], i, i_bus)] )

                for i_clique  in self.clique_tree_theta['node'].keys():
                    x_norm2_fs = x_norm2_fs + np.linalg.norm( self.X_theta[(k[0], k[1], i, i_clique)].value )
                    z_norm2_fs = z_norm2_fs + np.linalg.norm( self.Y_theta[(k[0], k[1], i, i_clique)] )
                    l_norm2_fs = l_norm2_fs + np.linalg.norm( self.Lambda_fs['theta'][(k[0], k[1], i, i_clique)]  )

        return x_norm2_fs, z_norm2_fs, l_norm2_fs

    def termination_fs(self):

        # old termination code is repeated partially

        if self.kappa == 0:
            self.r_norm_2 = dict()
            self.s_norm_2 = dict()

            self.J_iter = dict()
            self.solve_time = dict()

            self.epsilon_pri = dict()
            self.epsilon_dual = dict()

            self.epsilon_p = (3 + 2) * self.N_s * len(self.i_gen)  + 2**2 * 2 * len(self.M_lower_union) * len(self.i_gen) + sum( (len(tong) + 1) + 2*len(tong) for tong in self.clique_tree_theta['node_gl'].values()) * len(self.M_lower_union)  * 2
            #self.epsilon_n count in _opt_xs_init

            self.r_norm_2_fs = dict()
            self.s_norm_2_fs = dict()

            self.epsilon_pri_fs = dict()
            self.epsilon_dual_fs = dict()

            self.epsilon_p_fs = self.epsilon_n
            self.epsilon_n_fs = self.epsilon_n

        if self.sep_mode != 'k':
            self.r_norm_2[self.kappa] = (1.0/self.rho) * (sum([self._dict_sq_md(self.Lambda[s]['md'], self.Lambda_pre[s]['md'])  +  self._dict_sq(self.Lambda[s]['w'], self.Lambda_pre[s]['w']) +  self._dict_sq_theta(self.Lambda[s]['theta'], self.Lambda_pre[s]['theta']) for s in self.P]))**0.5
            self.s_norm_2[self.kappa] =  self.rho * (self.N_s * self._dict_sq_md(self.z_md, self.z_md_pre) +  2 * self._dict_sq(self.z_w, self.z_w_pre) + 2 *self._dict_sq_theta(self.z_theta, self.z_theta_pre) )**0.5
           
        else:
            self.r_norm_2[self.kappa] = (1.0/self.rho) * (sum([self._dict_sq_md(self.Lambda[s]['md'], self.Lambda_pre[s]['md']) for s in self.P]))**0.5
            self.s_norm_2[self.kappa] =  self.rho * (self.N_s * self._dict_sq_md(self.z_md, self.z_md_pre) )**0.5           

        x_norm2, z_norm2, l_norm2  = self._epsilon_great()
        self.epsilon_pri[self.kappa] =  ( self.epsilon_p )**0.5 * self.epsilon_abs + self.epsilon_rel * max(x_norm2, z_norm2)
        self.epsilon_dual[self.kappa] =  ( self.epsilon_n )**0.5 * self.epsilon_abs + self.epsilon_rel * l_norm2

        # for _fs
        self.r_norm_2_fs[self.kappa] = (1.0/self.rho_fs) * (self._dict_sq(self.Lambda_fs['lmd'], self.Lambda_fs_pre['lmd'])  +  self._dict_sq(self.Lambda_fs['mdw'], self.Lambda_fs_pre['mdw']) +  self._dict_sq(self.Lambda_fs['theta'], self.Lambda_fs_pre['theta']) )**0.5

        self.s_norm_2_fs[self.kappa] =  self.rho_fs * ( self._dict_sq(self.Y_lmd, self.Y_lmd_pre) +  self._dict_sq(self.Y_mdw, self.Y_mdw_pre) + self._dict_sq(self.Y_theta, self.Y_theta_pre) )**0.5

        x_norm2_fs, z_norm2_fs, l_norm2_fs  = self._epsilon_great_fs()

        self.epsilon_pri_fs[self.kappa] =  ( self.epsilon_p_fs )**0.5 * self.epsilon_abs_fs + self.epsilon_rel_fs * max(x_norm2_fs, z_norm2_fs)
        self.epsilon_dual_fs[self.kappa] =  ( self.epsilon_n_fs )**0.5 * self.epsilon_abs_fs + self.epsilon_rel_fs * l_norm2_fs

        self.J_iter[self.kappa] = sum(self.J[s].value for s in self.P)
        self.solve_time[self.kappa] = sum(self.opt_x[s].solution.attr['solve_time'] for s in self.P)


        print(self.kappa, ':    ', self.r_norm_2[self.kappa], self.s_norm_2[self.kappa], self.r_norm_2_fs[self.kappa], self.s_norm_2_fs[self.kappa],  self.J_iter[self.kappa] , self.solve_time[self.kappa])
        print(self.kappa, ':    ', self.epsilon_pri[self.kappa],  self.epsilon_dual[self.kappa] , self.epsilon_pri_fs[self.kappa],  self.epsilon_dual_fs[self.kappa]  , self.rho)

    def update_rho_fs(self):

        #self.update_rho()

        if self.kappa <5:
            self.rho_fs = 0.000001
        '''

        if self.r_norm_2_fs[self.kappa] >  self.rho_mu_fs * self.s_norm_2_fs[self.kappa]:
            self.rho_fs =  self.rho_fs * self.rho_tau_fs
        elif  self.s_norm_2_fs[self.kappa] >  self.rho_mu_fs * self.r_norm_2_fs[self.kappa]:
            self.rho_fs =  self.rho_fs * (1/self.rho_tau_fs)
        else:
            pass
        '''

