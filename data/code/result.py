
from os.path import dirname, join
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

def read_result():
       casename = 'case9_mdf_1'
       mode = 23

       J_iter = dict()
       r_norm_2 = dict()
       s_norm_2 = dict()
       epsilon_pri  = dict()
       epsilon_dual = dict()

       for opt_num in ['d11']:#, 'd2', 'd3', 'd4', 'd5']:
              path_result = join(dirname(__file__), 'result//case-9//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')

              with open(path_result, 'rb') as fpr:
                     opt_result =  pickle.load(fpr)

              np.array(list(opt_result.r_norm_2.values()))

              J_iter[opt_num]       = np.array(list(opt_result.J_iter.values()))
              r_norm_2[opt_num]     = np.array(list(opt_result.r_norm_2.values()))
              s_norm_2[opt_num]     = np.array(list(opt_result.s_norm_2.values()))
              epsilon_pri[opt_num]  = np.array(list(opt_result.epsilon_pri.values()))
              epsilon_dual[opt_num] = np.array(list(opt_result.epsilon_dual.values()))

       res_all = dict()
       res_all['J_iter']       =    J_iter
       res_all['r_norm_2']     =    r_norm_2
       res_all['s_norm_2']     =    s_norm_2
       res_all['epsilon_pri']  =    epsilon_pri 
       res_all['epsilon_dual'] =    epsilon_dual

       path_res_all = join(dirname(__file__), 'result//case-9//' + 'result-' + casename + '-mode_' + str(mode) + '.p')
       with open(path_res_all, 'wb') as fpr:
            pickle.dump(res_all, fpr)
       
#read_result()




casename = 'case9_mdf_1'
mode = 23
path_res_all = join(dirname(__file__), 'result//case-9//' + 'result-' + casename + '-mode_' + str(mode) + '.p')
with open(path_res_all, 'rb') as fpr:
       res_all =  pickle.load(fpr)

plt.rc('text', usetex=True)
plt.rc('font', size=10,family='Times New Roman')



x_ind = range(1, 101)
x0_ind = range(0, 101)

path_fig = join(dirname(__file__), 'result//case-9//' + 'fig-J-' + '.pdf')

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,2.5))


axs.plot(x_ind, res_all['J_iter']['d11'], '-*', linewidth=1.2, label = 'ADMM')

axs.legend(loc='upper right')


axs.set_xlabel('$\\kappa$')
axs.set_xlim(5,101)
axs.set_xticks([5,20, 40, 60, 80, 100])
axs.set_xticklabels([5,20, 40, 60, 80, 100])

axs.set_ylim(0.8235, 0.8241)

axs.set_ylabel('Objective function $\\hat{J}$')

axs.grid(color='gray', linestyle='--', linewidth=0.4)

plt.show()

fig.savefig(path_fig, dpi = 300, transparent=True, bbox_inches='tight')






'''
J_nlp = {'d1': 0.9138, 'd2': 0.7097, 'd3': 1.0023, 'd4': 0.5527, 'd5': 3.2391 }

J_sdp = {'d1': 0.7912, 'd2': 0.6406, 'd3': 0.9035, 'd4': 0.4501, 'd5': 2.7905 }


casename = 'case9_mdf_1'
mode = 23
path_res_all = join(dirname(__file__), 'result//case-9//' + 'result-' + casename + '-mode_' + str(mode) + '.p')
with open(path_res_all, 'rb') as fpr:
       res_all =  pickle.load(fpr)

plt.rc('text', usetex=True)
plt.rc('font', size=10,family='Times New Roman')

opt_num = 'd5'

x_ind = range(1, 26)
x0_ind = range(0, 26)

path_fig = join(dirname(__file__), 'result//case-9//' + 'fig-J-' + opt_num + '.pdf')

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,2.5))

axs.plot(x_ind, [J_nlp[opt_num]]*25, '--', linewidth=1.2, label = 'NLP')
axs.plot(x_ind, [J_sdp[opt_num]]*25, '--', linewidth=1.2, label = 'SDP')
axs.plot(x_ind, res_all['J_iter'][opt_num], '-*', linewidth=1.2, label = 'ADMM')

axs.legend(loc='upper right')


axs.set_xlabel('$\\kappa$')
axs.set_xlim(0,26)
axs.set_xticks([0,5,10,15,20,25])
axs.set_xticklabels([0,5,10,15,20,25])

axs.set_ylabel('Objective function $\\hat{J}$')

axs.grid(color='gray', linestyle='--', linewidth=0.4)

plt.show()

fig.savefig(path_fig, dpi = 300, transparent=True, bbox_inches='tight')

'''



casename = 'case9_mdf_1'
mode = 23
path_res_all = join(dirname(__file__), 'result//case-9//' + 'result-' + casename + '-mode_' + str(mode) + '.p')
with open(path_res_all, 'rb') as fpr:
       res_all =  pickle.load(fpr)

plt.rc('text', usetex=True)
plt.rc('font', size=10,family='Times New Roman')

opt_num = 'd11'

x_ind = range(1, 101)
x0_ind = range(0, 101)

path_fig = join(dirname(__file__), 'result//case-9//' + 'fig-' + opt_num + '.pdf')

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5,5))
axs[0].semilogy(x_ind, res_all['r_norm_2'][opt_num] * 5, '-', linewidth=1.2, label = '$||r||_2$' , lw=10)
axs[0].semilogy(x_ind, res_all['epsilon_pri'][opt_num] * 5, '-', linewidth=1.2, label = '$\\epsilon^{pri}$' , lw=10)
axs[1].semilogy(x_ind, res_all['s_norm_2'][opt_num] * 2, '-', linewidth=1.2, label = '$||s||_2$' , lw=10)
axs[1].semilogy(x_ind, res_all['epsilon_dual'][opt_num] * 2, '-', linewidth=1.2, label = '$\\epsilon^{dual}$' , lw=10)

axs[0].legend()
axs[1].legend()


axs[1].set_xlabel('$\\kappa$')
axs[0].set_xlim(0,101)
axs[0].set_xticks([0,20, 40, 60, 80, 100])
axs[0].set_xticklabels([0,20, 40, 60, 80, 100])
axs[1].set_xlim(0,26)
axs[1].set_xticks([0,20, 40, 60, 80, 100])
axs[1].set_xticklabels([0,20, 40, 60, 80, 100])

axs[0].set_ylabel('$||r||_2$ or $\\epsilon^{pri}$')
axs[1].set_ylabel('$||s||_2$ or $\\epsilon^{dual}$')

axs[0].grid(color='gray', linestyle='--', linewidth=0.4)
axs[1].grid(color='gray', linestyle='--', linewidth=0.4)

plt.show()

fig.savefig(path_fig, dpi = 300, transparent=True, bbox_inches='tight')
