# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:44:03 2018

pre-process class

@author: Tong
"""
from pypower.api import makeYbus, runpf, loadcase, ext2int, ppoption
import numpy as np
from pyomo.environ import *
from sympy import Symbol, integrate
from copy import deepcopy
from random import shuffle
import networkx as nx
import scipy as sp
from sksparse.cholmod import cholesky
from networkx.algorithms import approximation
from itertools import combinations

class PreParam:
    def __init__(self, path_casedata, set_disturb, mode):

        self.path_casedata = path_casedata
        self.mode = mode
        self.set_disturb = set_disturb

        self.casedata = loadcase(path_casedata)
        self._get_param()
        self.dict_int_l()
        self.dict_der_l()
        self.param_collocation()
        self.make_clique_tree_theta()

    def _get_param(self):

        # delect the disturbances not defined in the set_disturb, and renumber the new disturbance set.
        for d_type in {1,2,3,4}:
            i_d_all = list()
            for i_d in range( len( self.casedata['disturbance'][d_type] ) ):
                if (d_type, self.casedata['disturbance'][d_type][i_d, 0]) in self.set_disturb:
                    i_d_all.append(i_d)
            self.casedata['disturbance'][d_type] = self.casedata['disturbance'][d_type][i_d_all, :]
            for i_d in range( len( self.casedata['disturbance'][d_type] ) ):
                self.casedata['disturbance'][d_type][i_d, 0]  = i_d  + 1
            if len( self.casedata['disturbance'][d_type] ) == 0:
                del self.casedata['disturbance'][d_type]

        # parameters
        if len(self.casedata['bus']) >= 200 or len(self.casedata['bus']) == 59:
            opt = ppoption(PF_TOL=1e-12, PF_MAX_IT= 20)
        else:
            opt = ppoption(PF_TOL=1e-13, PF_MAX_IT= 20)
        s_pf = runpf(self.casedata,opt)
        
        pcc_y = ext2int(self.casedata)     
        Ybus = makeYbus(pcc_y["baseMVA"], pcc_y["bus"], pcc_y["branch"])[0].toarray()
        n_bus = len(Ybus)
        B_0 = np.imag(Ybus)
        for i in range(len(B_0)): B_0[i,i] = 0
        pg_0  = np.zeros(n_bus)
        pg_0[(s_pf[0]['gen'][:,0] - 1).astype(int)] = s_pf[0]['gen'][:,1]
        i_gen = self.casedata['gen'][:,0].astype(int).tolist()
        i_non_0 = (np.where(self.casedata['bus'][:,2]==0)[0]+1).tolist()
        i_non = list(set(i_non_0).difference(set(i_gen)))
        i_load_0 = list(set(self.casedata['bus'][:,0].astype(int).tolist()).difference(set(i_gen)))
        i_load = list(set(i_load_0).difference(set(i_non)))

        self.Ybus = Ybus
        self.n_bus = n_bus
        self.B_0 = B_0
        self.v_0     = s_pf[0]['bus'][:,7]
        self.theta_0 = np.radians(s_pf[0]['bus'][:,8])
        self.pl_0    = s_pf[0]['bus'][:,2]/100.0
        self.pg_0 = pg_0/100.0
        self.w_0 = np.zeros(self.n_bus)

        self.i_gen = i_gen
        self.i_non = i_non
        self.i_load = i_load
        self.i_gl = i_gen + i_load
        self.i_all = self.casedata['bus'][:, 0].astype(int).tolist()

        self.theta_gl_0 = self.theta_0[(np.array(self.i_gl) - 1).tolist()]

        # initial values and bounds of optimization variables
        self.m_0, self.m_l, self.m_u = dict(), dict(), dict()
        self.d_0, self.d_l, self.d_u = dict(), dict(), dict()
        self.M = self.casedata['bus'][:,13].tolist()
        self.D = self.casedata['bus'][:,14].tolist()

        
        for i in self.i_gen:
            i_gc = np.where(self.casedata['gencontrol'][:,0]==i)[0]
            i_bus = np.where(self.casedata['bus'][:,0]==i)[0][0]
            self.m_0[i] = self.M[i_bus]
            self.m_l[i] = self.casedata['gencontrol'][i_gc, 2][0]
            self.m_u[i] = self.casedata['gencontrol'][i_gc, 3][0]
            self.d_0[i] = self.D[i_bus]
            self.d_l[i] = self.casedata['gencontrol'][i_gc, 4][0]
            self.d_u[i] = self.casedata['gencontrol'][i_gc, 5][0]
  
        # set of index for each kind of branch, used for computing the objective function
        self.ind_branch = list() # index set [(ind_bus_from, ind_bus_to)] of all branch
        for i_b in self.casedata['branch'][:,[0,1]]:
            self.ind_branch.append( (int(i_b[0]), int(i_b[1])) )

        set_k_i = list()  # list of (k,i) of all disturbances and time elements
        for k0 in self.casedata['disturbance'].keys():
            if k0 != 9:
                for k1 in  self.casedata['disturbance'][k0][:,0]:
                    for i in range(1, self.casedata['param_disc']['time_ele'][k0] + 1):   
                        set_k_i.append(((int(k0),int(k1)),i))
        self.set_k_i = set_k_i


    def dict_der_l(self):
        # get dict of  lagrange polynomials derivatives, der_l[j,r] for all j in 0~nc, and r in 1~nc
        der_l = dict()
        for nc_l in set(self.casedata['param_disc']['order'].values()):
            s_tau_l = self.casedata['param_disc']['colloc_point_radau'][nc_l]
            for j_l in range(0, nc_l + 1):
                for r_l in range(1, nc_l + 1):
                    NC = list(range(0, nc_l + 1))
                    der_l[nc_l, s_tau_l, j_l, r_l] = sum( (1/(s_tau_l[j_l] - s_tau_l[rr])) * prod( (s_tau_l[r_l] - s_tau_l[rrr])/(s_tau_l[j_l] - s_tau_l[rrr]) for rrr in set(NC) - set([j_l]) - set([rr]))  for rr in set(NC)-set([j_l]) ) 
        self.der_l = der_l


    def f_int_l(self, flag, nc_l, s_tau_l, j1, j2):
        '''
        compute S_{flag}^{j1 j2}, flag \in {1,2,3,4}
        nc_l - order
        s_tau_l - set of collocation points
        '''
        NC = list(range(0, nc_l + 1))
        tau = Symbol('tau')
        if flag == 1: # f_j1 * f_j2 = l_{j1} * l_{j2}
            f_j1 =  prod( (tau - s_tau_l[rr])/(s_tau_l[j1] - s_tau_l[rr]) for rr in set(NC) - set([j1]))
            f_j2 =  prod( (tau - s_tau_l[rr])/(s_tau_l[j2] - s_tau_l[rr]) for rr in set(NC) - set([j2]))
        elif flag == 2: # f_j1 * f_j2 = \dot{l}_{j1} * \dot{l}_{j2}
            f_j1 = sum( (1/(s_tau_l[j1] - s_tau_l[rr])) * prod( (tau - s_tau_l[rrr])/(s_tau_l[j1] - s_tau_l[rrr]) for rrr in set(NC) - set([j1]) - set([rr]))  for rr in set(NC)-set([j1]) )
            f_j2 = sum( (1/(s_tau_l[j2] - s_tau_l[rr])) * prod( (tau - s_tau_l[rrr])/(s_tau_l[j2] - s_tau_l[rrr]) for rrr in set(NC) - set([j2]) - set([rr]))  for rr in set(NC)-set([j2]) )
        elif flag == 3: # f_j1 * f_j2 = \dot{l}_{j1} * {l}_{j2}
            f_j1 = sum( (1/(s_tau_l[j1] - s_tau_l[rr])) * prod( (tau - s_tau_l[rrr])/(s_tau_l[j1] - s_tau_l[rrr]) for rrr in set(NC) - set([j1]) - set([rr]))  for rr in set(NC)-set([j1]) )
            f_j2 = prod( (tau - s_tau_l[rr])/(s_tau_l[j2] - s_tau_l[rr]) for rr in set(NC) - set([j2]))
        else:
            print("ERROR: Wrong flag")

        return integrate( f_j1 * f_j2, (tau, 0, 1))

    def dict_int_l(self):
        int_l = dict()
        for flag in [1,2,3]:
            for nc in set(self.casedata['param_disc']['order'].values()):
                s_tau = self.casedata['param_disc']['colloc_point_radau'][nc]
                for jj1 in range(0, nc+1):
                    for jj2 in range(0, nc+1):
                        int_l[flag, nc, s_tau, jj1, jj2] = self.f_int_l(flag, nc, s_tau, jj1, jj2)
        self.int_l = int_l


    def param_collocation(self):

        self.param_dd = dict()
        for k0 in self.casedata['disturbance'].keys():
            param_d = dict()
            if k0 in [1, 3]:
                if k0 == 1:
                    tp = np.logspace(0.2, 1, self.casedata['param_disc']['time_ele'][k0], endpoint=True)
                else:
                    tp = np.array([1] * self.casedata['param_disc']['time_ele'][k0])
                th = self.casedata['param_disc']['t_f'][k0]*tp/sum(tp)
                param_d[1] = (0, th[0], self.casedata['param_disc']['order'][k0], self.casedata['param_disc']['colloc_point_radau'][self.casedata['param_disc']['order'][k0]])
                for i in range(2,self.casedata['param_disc']['time_ele'][k0]+1):
                    param_d[i] = (param_d[i-1][1], param_d[i-1][1] + th[i-1], self.casedata['param_disc']['order'][k0], self.casedata['param_disc']['colloc_point_radau'][self.casedata['param_disc']['order'][k0]])
            elif k0 in [2, 4]:
                if k0 == 2: i_t = 3
                else: i_t = 6
                th_d = np.linspace(0,  self.casedata['disturbance'][k0][0,i_t], self.casedata['param_disc']['time_ele_d'][k0] + 1)
                tp = np.logspace(0.2, 1, self.casedata['param_disc']['time_ele'][k0] - self.casedata['param_disc']['time_ele_d'][k0], endpoint=True)
                th = (self.casedata['param_disc']['t_f'][k0] - self.casedata['disturbance'][k0][0,i_t]) * tp/sum(tp)
                for i in range(1, self.casedata['param_disc']['time_ele_d'][k0]+1):
                    param_d[i] = (th_d[i-1], th_d[i], self.casedata['param_disc']['order'][k0], self.casedata['param_disc']['colloc_point_radau'][self.casedata['param_disc']['order'][k0]])
                for i in range(self.casedata['param_disc']['time_ele_d'][2] + 1, self.casedata['param_disc']['time_ele'][k0] + 1  ):
                    param_d[i] = (param_d[i-1][1], param_d[i-1][1] + th[i - self.casedata['param_disc']['time_ele_d'][k0] - 1],  self.casedata['param_disc']['order'][k0], self.casedata['param_disc']['colloc_point_radau'][self.casedata['param_disc']['order'][k0]])
            self.param_dd[k0] = param_d


    def array_disturbance(self, kk, ii, rr):
        '''
        return the array-form disturbance parameter for given disturbance and time moment.
        kk - index of disturbance, resign of k
        ii - index of time element, i
        rr - index of collocation point, r
        '''
        ratio_pg = np.ones(self.n_bus)
        ratio_pl = np.ones(self.n_bus)
        delta_G = np.zeros(self.n_bus)
        ratio_B = np.ones([self.n_bus, self.n_bus])
        
        param_dd_ki = self.param_dd[kk[0]][ii]
        t_ir = param_dd_ki[0] + (param_dd_ki[1] - param_dd_ki[0]) * param_dd_ki[3][rr]  # time moment

        if kk[0] == 1:
            if self.casedata['disturbance'][1][kk[1]-1, 1] in self.i_gen:
                ratio_pg[int(self.casedata['disturbance'][1][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][1][kk[1]-1, 3]
            elif self.casedata['disturbance'][1][kk[1]-1, 1] in self.i_load:
                ratio_pl[int(self.casedata['disturbance'][1][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][1][kk[1]-1, 3]
            else:
                print("ERROR: Power disturbance with non load or generator buses")
        
        if kk[0] == 2:
            if t_ir < self.casedata['disturbance'][2][kk[1]-1, 3]:
                if self.casedata['disturbance'][2][kk[1]-1, 1] in self.i_gen:
                    ratio_pg[int(self.casedata['disturbance'][2][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][2][kk[1]-1, 4] * (t_ir/self.casedata['disturbance'][2][kk[1]-1, 3])
                elif self.casedata['disturbance'][2][kk[1]-1, 1] in self.i_load:
                    ratio_pl[int(self.casedata['disturbance'][2][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][2][kk[1]-1, 4] * (t_ir/self.casedata['disturbance'][2][kk[1]-1, 3])
                else:
                    print("ERROR: Power disturbance with non load or generator buses")
            else:
                if self.casedata['disturbance'][2][kk[1]-1, 1] in self.i_gen:
                    ratio_pg[int(self.casedata['disturbance'][2][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][2][kk[1]-1, 4]
                elif self.casedata['disturbance'][2][kk[1]-1, 1] in self.i_load:
                    ratio_pl[int(self.casedata['disturbance'][2][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][2][kk[1]-1, 4]
                else:
                    print("ERROR: Power disturbance with non load or generator buses")

        if kk[0] == 3:
            if self.casedata['disturbance'][3][kk[1]-1, 1] in self.i_gen:
                ratio_pg[int(self.casedata['disturbance'][3][kk[1]-1, 1] )- 1] = 1 + self.casedata['disturbance'][self.casedata['disturbance'][3][kk[1]-1, 4]][int(t_ir/self.casedata['disturbance'][3][kk[1]-1, 3])]    
            elif self.casedata['disturbance'][3][kk[1]-1, 1] in self.i_load:
                ratio_pl[int(self.casedata['disturbance'][3][kk[1]-1, 1]) - 1] = 1 + self.casedata['disturbance'][self.casedata['disturbance'][3][kk[1]-1, 4]][int(t_ir/self.casedata['disturbance'][3][kk[1]-1, 3])] 
            else:
                print("ERROR: Power disturbance with non load or generator buses")

        if kk[0] == 4:
            if t_ir <= self.casedata['disturbance'][4][kk[1]-1, 6]:
                delta_G[int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1] = 0.5 * self.B_0[int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1, int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1]
                delta_G[int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1] = 0.5 * self.B_0[int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1, int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1]
                ratio_B[int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1, int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1] = 0
                ratio_B[int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1, int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1] = 0

            else:
                ratio_B[int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1, int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1] = 0
                ratio_B[int(self.casedata['disturbance'][4][kk[1]-1, 2]) - 1, int(self.casedata['disturbance'][4][kk[1]-1, 1]) - 1] = 0
            
        return  [ratio_pg, ratio_pl, delta_G, ratio_B]

    def _partition_list(self, lst, n):
        division = len(lst) / n
        return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

    def _max_multiple(self, lstt):
        lst = deepcopy(lstt)
        s_max = list()
        s_max.append(max(lst))
        lst.remove(max(lst))
        s_min = [9999]
        while lst != []:
            if max(lst) == min(s_max) - 1 or max(lst) == min(s_min) - 1:
                s_min.append(max(lst))
                lst.remove(max(lst))
            else:
                s_max.append(max(lst))
                lst.remove(max(lst))
        return s_max

    def _min_multiple(self, lstt):
        lst = deepcopy(lstt)
        s_min = list()
        s_min.append(min(lst))
        lst.remove(min(lst))
        s_max = [-9999]
        while lst != []:
            if min(lst) == max(s_min) + 1 or min(lst) == max(s_max) + 1:
                s_max.append(min(lst))
                lst.remove(min(lst))
            else:
                s_min.append(min(lst))
                lst.remove(min(lst))
        return s_min

    def seperation_compute(self):
        '''
        from the computational perspective
        divide  set_k_i into N_s disjoint sets Xi_s, also generate M_upper and M_lower
        sep_mode :  'r' - random seperation; 'k'- some disturbance with its all time elements
                    'i' - some time elements with all disturbances; 'ki'
        N_s_k and N_s_i only for sep_mode = 'ki', when N_s_k = 1 or N_s_i = 1, 'ki' mode while be the same as 'i' mode of 'k' mode 
        '''
        Xi = dict()
        P = range(1, self.N_s + 1)   #s \in P = {1,...,N_s}
        for s in P: Xi[s] = list()
        set_k = list(set(np.array(self.set_k_i)[:,0]))  # list of all disturbances
        set_k = sorted(set(list(np.array(self.set_k_i)[:,0])), key = list(np.array(self.set_k_i)[:,0]).index )

        if self.sep_mode == 'k':
            set_k_parti = self._partition_list(set_k, self.N_s)
            for ki in self.set_k_i:
                for s in P:
                    if ki[0] in set_k_parti[s - 1]:
                        Xi[s].append(ki)
                        break

        if self.sep_mode == 'i':
            for k in set_k:
                set_i_parti = self._partition_list( list(range(1, self.casedata['param_disc']['time_ele'][k[0]] + 1)), self.N_s)
                for s in P:
                    for i in set_i_parti[s - 1]:
                        Xi[s].append( (k, i) )

        if self.sep_mode == 'r':
            set_k_i = deepcopy(self.set_k_i)
            shuffle(set_k_i)
            set_k_i_parti = self._partition_list( set_k_i, self.N_s)
            for s in P:
                for ki in set_k_i_parti[s-1]:
                    Xi[s].append( ki )

        if self.sep_mode == 'ki':
            if self.N_s_k ==0 or self.N_s_i == 0:
                print('Values for N_s_k and N_s_i cannot be 0')
            else:
                set_k_parti = self._partition_list(set_k, self.N_s_k)
                for (kk, index_kk) in zip(set_k_parti, range(1, self.N_s_k + 1)):
                    for k in kk:
                        set_i_parti = self._partition_list( list(range(1, self.casedata['param_disc']['time_ele'][k[0]] + 1)), self.N_s_i)
                        for (ii, index_ii) in zip(set_i_parti, range(1, self.N_s_i + 1)):
                            for i in ii:
                                Xi[(index_ii - 1) * self.N_s_k + index_kk].append( (k, i) )

        M_upper = dict()
        M_lower = dict()

        for s in P:
            M_upper[s] = list()
            M_lower[s] = list()
            Xi_s_np = np.array(Xi[s])
            set_k_s = list(set(Xi_s_np[:,0]))
            for k in set_k_s:
                ii_k = Xi_s_np[:,1][list(set(np.where(np.array(Xi_s_np[:,0].tolist())[:,0] == k[0])[0]).intersection(set(np.where(np.array(Xi_s_np[:,0].tolist())[:,1] == k[1])[0])))].tolist()
                ii_max = self._max_multiple(ii_k)
                ii_min = self._min_multiple(ii_k)
                if self.casedata['param_disc']['time_ele'][k[0]] in ii_max:
                    ii_max.remove(self.casedata['param_disc']['time_ele'][k[0]])
                if 1 in ii_min:
                    ii_min.remove(1)
                [M_upper[s].append( (k,i) ) for i in ii_max]
                [M_lower[s].append( (k,i) ) for i in ii_min]

        M_lower_union = list()
        [M_lower_union.extend(M_lower[s]) for s in P]

        self.Xi = Xi
        self.P = P
        self.set_k = set_k
        self.M_upper = M_upper
        self.M_lower = M_lower
        self.M_lower_union = M_lower_union

    def make_clique_tree_grid(self):
        '''
        construct the clique tree of graph of the grid
        '''

        # construct the graph of the power grid, and node_array gives the node order in A
        G = nx.Graph()
        G.add_edges_from(self.ind_branch)
        node_array = np.array(G.nodes)
        A = nx.linalg.graphmatrix.adjacency_matrix(G)
        A = sp.sparse.csc_matrix(A)

        factor = cholesky(A, beta= 100)
        L =factor.L()
        P = factor.P()

        node_array = node_array[P[:, np.newaxis]]  # reorder node_array according to P

        for (v1,v2) in zip( node_array[L.nonzero()[0], 0], node_array[L.nonzero()[1], 0] ):
            if v1 != v2 and (v1, v2) not in G.edges():
                G.add_edge(v1, v2)

        # max_clique is the set of all maximal cliques, which is different from G_clique.nodes. 
        # max_clique is a subset of G_clique.nodes
        if nx.is_chordal(G):
            max_clique = [cliq for cliq in nx.find_cliques(G)]
        else:
            print('The  graph is not chordal')
        max_clique = [set(i) for i in max_clique]


        # approach 1
        G_clique = G = nx.Graph()

        G_clique_nodes = dict(zip(range(1, len(max_clique)+1), max_clique))

        G_clique.add_nodes_from(G_clique_nodes.keys())
        for ed in  list(combinations(G_clique.nodes, 2)):
            it = G_clique_nodes[ed[0]].intersection(G_clique_nodes[ed[1]])
            if it !=  set():
                G_clique.add_edge(ed[0], ed[1], weight = len(it))

        G_clique = nx.algorithms.tree.mst.maximum_spanning_tree(G_clique, weight='weight', algorithm='kruskal')

        G_clique_edges = list( G_clique.edges() )

        G_clique_nodes_gl = deepcopy(G_clique_nodes)
        for key in G_clique_nodes_gl.keys():
            G_clique_nodes_gl[key] = G_clique_nodes_gl[key].intersection(self.i_gl)

        for key in G_clique_nodes.keys():
            G_clique_nodes[key] = list(G_clique_nodes[key])
            G_clique_nodes_gl[key] = list(G_clique_nodes_gl[key])

        for key in G_clique_nodes.keys():
            if  G_clique_nodes_gl[key] == []:
                del G_clique_nodes_gl[key]
        clique_tree_grid = {'node': G_clique_nodes, 'edge': G_clique_edges, 'node_gl': G_clique_nodes_gl}


        # approach 2
        # tree decomposition. It is not a strict clique tree.
        # G_clique = approximation.treewidth.treewidth_min_fill_in(G)[1]

        # G_clique_nodes = dict(zip(range(1, len(G_clique.nodes)+1), list(G_clique.nodes)))
        # G_clique_edges = list()
        # for edge_clique in G_clique.edges:
        #     G_clique_edges.append( [list(G_clique_nodes.keys())[list(G_clique_nodes.values()).index(edge_clique[0])], 
        #         list(G_clique_nodes.keys())[list(G_clique_nodes.values()).index(edge_clique[1])]] )

        # G_clique_nodes_gl = deepcopy(G_clique_nodes)
        # for key in G_clique_nodes_gl.keys():
        #     G_clique_nodes_gl[key] = G_clique_nodes_gl[key].intersection(self.i_gl)

        # for key in G_clique_nodes.keys():
        #     G_clique_nodes[key] = list(G_clique_nodes[key])
        #     G_clique_nodes_gl[key] = list(G_clique_nodes_gl[key])

        # for key in G_clique_nodes.keys():
        #     if  G_clique_nodes_gl[key] == []:
        #         del G_clique_nodes_gl[key]
            
        # clique_tree_grid = {'node': G_clique_nodes, 'edge': G_clique_edges, 'node_gl': G_clique_nodes_gl}

        return clique_tree_grid

    def make_clique_tree_theta(self):
        '''
        generate the clique tree for the decomposition of theta, which is actually the clique tree of the power system graph.
        also generate a dict for  ij_branch -> node of clique containing ij_branch
        '''

        self.clique_tree_theta =  self.make_clique_tree_grid()

        branch_clique = dict()
        for ij_branch in self.ind_branch:
            for node_clique in self.clique_tree_theta['node'].keys():
                if ( ij_branch[0] in self.clique_tree_theta['node'][node_clique] ) and (ij_branch[1] in self.clique_tree_theta['node'][node_clique]):
                    branch_clique[ij_branch] = node_clique
                    branch_clique[(ij_branch[1], ij_branch[0])] = node_clique
                    break
                else:
                    pass
        self.branch_clique = branch_clique

        bus_clique = dict()
        for i_bus in self.i_all:
            for node_clique in self.clique_tree_theta['node'].keys():
                if i_bus in self.clique_tree_theta['node'][node_clique]:
                    bus_clique[i_bus] = node_clique
                    break
                else:
                    pass
        self.bus_clique = bus_clique

    def make_select_array(self):

        # matrix for epsilon_pri and epsilon_dual
        self.array_select_md  = np.array( [[1,1,1],[1,0,0],[1,0,0]] )

        # array to select the variables for coupling
        self.array_select_sdp = dict()

        for ki in self.M_lower_union:
            for i_clique in self.clique_tree_theta['node_gl'].keys():
                n_gl = len(self.clique_tree_theta['node_gl'][i_clique])
                self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ] = np.ones([n_gl + 1, n_gl+1])
                self.array_select_sdp[(ki[0][0], ki[0][1], ki[1]-1, i_clique) ] = np.ones([n_gl + 1, n_gl+1])
                if n_gl == 1:
                    pass
                else:
                    for i_gl in range(n_gl):
                        for j_gl in range(i_gl + 1, n_gl):
                            if True: #[self.clique_tree_theta['node_gl'][i_clique][i_gl], self.clique_tree_theta['node_gl'][i_clique][j_gl]]  not in self.ind_branch and [self.clique_tree_theta['node_gl'][i_clique][j_gl], self.clique_tree_theta['node_gl'][i_clique][i_gl]]  not in self.ind_branch:
                                self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ][i_gl+1, j_gl +1] = 0
                                self.array_select_sdp[(ki[0][0], ki[0][1], ki[1], i_clique) ][j_gl+1, i_gl +1] = 0
                                self.array_select_sdp[(ki[0][0], ki[0][1], ki[1]-1, i_clique) ][i_gl+1, j_gl +1] = 0
                                self.array_select_sdp[(ki[0][0], ki[0][1], ki[1]-1, i_clique) ][j_gl+1, i_gl +1] = 0


                





        









