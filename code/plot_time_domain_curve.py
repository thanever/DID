# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:41:27 2018

@author: eee
"""

from os.path import dirname, join
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pyomo.environ import *
from matplotlib import gridspec


plt.rc('text', usetex=True)
plt.rc('font', size=8,family='Times New Roman')
plt.rcParams.update({'figure.max_open_warning': 0})

plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['xtick.major.width'] = 0.4




def get_result(casename, mode, opt_num):
    '''
    load the opt results from saved file
    '''
    path_result = join(dirname(__file__), 'result//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')

    with open(path_result, 'rb') as fp:
        opt_result = pickle.load(fp)

    return opt_result

def get_curve_data(opt_result):


    x_w = dict()
    x_theta = dict()

    x_w.update(opt_result.opt_result.w.get_values())
    x_theta.update(opt_result.opt_result.theta.get_values())

    n_sample = 50 # number of samples in each time element
    d_type = opt_result.casedata['disturbance'].keys() - [9]
    curve_w = dict()
    for i_bus in opt_result.i_gen:
        for k0 in d_type:
            for k1 in  opt_result.casedata['disturbance'][k0][:,0]:
                k = (k0, k1)
                t_i = np.array([])
                x_w_t = np.array([])
                for i in range(1, len(opt_result.param_dd[k[0]].keys()) + 1):
                    t_i = np.r_[t_i, np.linspace(opt_result.param_dd[k[0]][i][0], opt_result.param_dd[k[0]][i][1], n_sample)]
                    tau_i = (np.linspace(opt_result.param_dd[k[0]][i][0], opt_result.param_dd[k[0]][i][1], n_sample) - opt_result.param_dd[k[0]][i][0])/(opt_result.param_dd[k[0]][i][1] - opt_result.param_dd[k[0]][i][0])
                    nc_l = opt_result.param_dd[k[0]][i][2]
                    s_tau_l = opt_result.param_dd[k[0]][i][3]
                    NC = list(range(0, nc_l + 1))
                    x_w_t = np.r_[x_w_t, sum( x_w[k[0], k[1], i, j, i_bus] * prod( (tau_i -  s_tau_l[r])/(s_tau_l[j] - s_tau_l[r]) for r in set(NC)-set([j]) )  for j in NC)]  
                curve_w[k, i_bus] = [t_i, x_w_t]

    curve_theta = dict()
    for i_bus in opt_result.i_gen + opt_result.i_load + opt_result.i_non:
        for k0 in d_type:
            for k1 in  opt_result.casedata['disturbance'][k0][:,0]:
                k = (k0, k1)
                t_i = np.array([])
                x_theta_t = np.array([])
                for i in range(1, len(opt_result.param_dd[k[0]].keys()) + 1):
                    t_i = np.r_[t_i, np.linspace(opt_result.param_dd[k[0]][i][0], opt_result.param_dd[k[0]][i][1], n_sample)]
                    tau_i = (np.linspace(opt_result.param_dd[k[0]][i][0], opt_result.param_dd[k[0]][i][1], n_sample) - opt_result.param_dd[k[0]][i][0])/(opt_result.param_dd[k[0]][i][1] - opt_result.param_dd[k[0]][i][0])
                    nc_l = opt_result.param_dd[k[0]][i][2]
                    s_tau_l = opt_result.param_dd[k[0]][i][3]
                    NC = list(range(0, nc_l + 1))
                    x_theta_t = np.r_[x_theta_t, sum( x_theta[k[0], k[1], i, j, i_bus] * prod( (tau_i -  s_tau_l[r])/(s_tau_l[j] - s_tau_l[r]) for r in set(NC)-set([j]) )  for j in NC)]
                curve_theta[k, i_bus] = [t_i, x_theta_t]

    curve_delta = dict()
    for i_branch in opt_result.ind_branch:
        for k0 in d_type:
            for k1 in  opt_result.casedata['disturbance'][k0][:,0]:
                k = (k0, k1)
                curve_delta[k, i_branch] = [curve_theta[k, i_bus][0], curve_theta[k, i_branch[0]][1] - curve_theta[k, i_branch[1]][1]]


    curve_diff_w = dict()
    for i_bus in opt_result.i_gen:
        for k0 in d_type:
            for k1 in  opt_result.casedata['disturbance'][k0][:,0]:
                k = (k0, k1)
                t_i = np.array([])
                x_diff_w_t = np.array([])
                for i in range(1, len(opt_result.param_dd[k[0]].keys()) + 1):
                    t_i = np.r_[t_i, np.linspace(opt_result.param_dd[k[0]][i][0], opt_result.param_dd[k[0]][i][1], n_sample)]
                    tau_i = (np.linspace(opt_result.param_dd[k[0]][i][0], opt_result.param_dd[k[0]][i][1], n_sample) - opt_result.param_dd[k[0]][i][0])/(opt_result.param_dd[k[0]][i][1] - opt_result.param_dd[k[0]][i][0])
                    nc_l = opt_result.param_dd[k[0]][i][2]
                    s_tau_l = opt_result.param_dd[k[0]][i][3]
                    NC = list(range(0, nc_l + 1))
                    h = opt_result.param_dd[k[0]][i][1] - opt_result.param_dd[k[0]][i][0]
                    x_diff_w_t = np.r_[x_diff_w_t,  (1/h) * sum( x_w[k[0], k[1], i, j, i_bus] *  (sum( (1/(s_tau_l[j] - s_tau_l[rr])) * prod( (tau_i - s_tau_l[rrr])/(s_tau_l[j] - s_tau_l[rrr]) for rrr in set(NC) - set([j]) - set([rr]))  for rr in set(NC)-set([j]) ) ) for j in NC)  ]

                curve_diff_w[k, i_bus] = [t_i, x_diff_w_t]


    curve_p_effort =  dict()
    for i_bus in opt_result.i_gen:
        mm = opt_result.opt_result.m[i_bus]()
        dd = opt_result.opt_result.d[i_bus]()
        for k0 in d_type:
            for k1 in  opt_result.casedata['disturbance'][k0][:,0]:
                k = (k0, k1)
                t_i = curve_w[k, i_bus][0]
                x_p_effort_t =  opt_result.pg_0[i_bus - 1]  - opt_result.pl_0[i_bus - 1] - mm * curve_diff_w[k, i_bus][1] - dd * curve_w[k, i_bus][1]
                curve_p_effort[k, i_bus]  = [t_i, x_p_effort_t]

    return {'delta':curve_delta, 'w':curve_w, 'diff_w':curve_diff_w, 'p_effort':curve_p_effort, 'theta':curve_theta}
    

casename = 'case200'
mode = 10
opt_num = 'd_1_2_3_4'

opt_result = get_result( casename, mode, opt_num )

curve_data = get_curve_data(opt_result)
d_type = opt_result.casedata['disturbance'].keys() - [9]
#================================================================

for k0 in d_type:
    for k1 in  opt_result.casedata['disturbance'][k0][:,0]:
        k =  (k0, k1)

        # plot delta
        path_fig = join(dirname(__file__), 'result//' + 'plot-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '-' + str(k0) + '_'+ str(int(k1)) + '-delta.pdf' )

        fig, axs = plt.subplots(figsize=(3.6,1.8))
        for i_branch in opt_result.ind_branch:
            axs.plot(curve_data['delta'][k, i_branch][0], (180/np.pi) * curve_data['delta'][k, i_branch][1]  , linewidth=1.2)
        axs.set_xlabel('$t$ (s)')
        axs.set_ylabel('$\delta$ (degree)')

        fig.savefig(path_fig,dpi = 300, transparent=True, bbox_inches='tight')

        # plot w
        path_fig = join(dirname(__file__), 'result//' + 'plot-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '-' +  str(k0) + '_'+ str(int(k1)) + '-w.pdf' )

        fig, axs = plt.subplots(figsize=(3.6,1.8))
        for i_bus in opt_result.i_gen:
            axs.plot(curve_data['w'][k, i_bus][0], 100* curve_data['w'][k, i_bus][1], linewidth=1.2)
        axs.set_xlabel('$t$ (s)')
        axs.set_ylabel('$\omega$ (cHz)')

        fig.savefig(path_fig,dpi = 300, transparent=True, bbox_inches='tight')

        # plot diff_w
        path_fig = join(dirname(__file__), 'result//' + 'plot-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '-' +  str(k0) + '_'+ str(int(k1)) + '-diff_w.pdf' )

        fig, axs = plt.subplots(figsize=(3.6,1.8))
        for i_bus in opt_result.i_gen:
            axs.plot(curve_data['diff_w'][k, i_bus][0], 100* curve_data['diff_w'][k, i_bus][1], linewidth=1.2 )
        axs.set_xlabel('$t$ (s)')
        axs.set_ylabel('$\dot{\omega}$ (cHz/s)')

        fig.savefig(path_fig,dpi = 300, transparent=True, bbox_inches='tight')

        # plot p_effort
        path_fig = join(dirname(__file__), 'result//' + 'plot-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '-' +  str(k0) + '_'+ str(int(k1)) + '-p_effort.pdf' )

        fig, axs = plt.subplots(figsize=(3.6,1.8))
        for i_bus in opt_result.i_gen:
            axs.plot(curve_data['p_effort'][k, i_bus][0], 100* curve_data['p_effort'][k, i_bus][1] , linewidth=1.2 , label = str(i_bus))
        axs.set_xlabel('$t$ (s)')
        axs.set_ylabel('$p_{ctrl}$ (MW)')

        axs.legend()

        fig.savefig(path_fig,dpi = 300, transparent=True, bbox_inches='tight')


