
# %%
import matplotlib.pyplot as plt
import pickle
from os.path import dirname, join
import numpy as np
import pickle
from make_parameter import PreParam
import os

plt.style.use('ggplot')


def get_np(casename_real, N_s, N_s_k, N_s_i):
    path_casedata = join(os.path.abspath('') , 'data//casedata//'+ casename_real +'.py')

    case_real = PreParam(path_casedata, [(1,1),(2,2),(3,1),(4,2)], 23)

    epsilon_n = len(case_real.set_k_i) *  (
         len(case_real.i_gen) * ( 1 + 2*(3 + 1) )**2 + 
         len(case_real.i_gen) * ( 1 + 2 + (3 + 1) )**2 + 
         sum( ( 1 + len(case_real.clique_tree_theta['node'][i_clique]) * (1 + 3) )**2  for i_clique  in case_real.clique_tree_theta['node'].keys() ) 
    ) + len(case_real.set_k_i) * 0.25 *   (4 + ((3 + 3 + 1)**2)) * len(case_real.i_all) * (3 + 1)

    epsilon_p = (
        (2 + 3) * N_s * len(case_real.i_gen)  + 2**2 * 2 * ( N_s_k * (N_s_i - 1) ) * len(case_real.i_gen) + sum( (len(tong) + 1) + 2*len(tong) for tong in case_real.clique_tree_theta['node_gl'].values()) * ( N_s_k * (N_s_i - 1) )  * 2
        + 
            len(case_real.set_k_i) *  (
         len(case_real.i_gen) * ( 1 + 2*(3 + 1) )**2 + 
         len(case_real.i_gen) * ( 1 + 2 + (3 + 1) )**2 + 
         sum( ( 1 + len(case_real.clique_tree_theta['node'][i_clique]) * (1 + 3) )**2  for i_clique  in case_real.clique_tree_theta['node'].keys() ) 
    ) +  ((3 + 3 + 1)**2) * len(case_real.i_all) * (3 + 1) * 0.25 * len(case_real.set_k_i)
    )

    return [epsilon_n, epsilon_p]



def get_result(casename, mode, opt_num):
    '''
    load the opt results from saved file
    '''
    path_result = join(os.path.abspath('') , 'result//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')
    with open(path_result, 'rb') as fp:
        opt_result = pickle.load(fp)
    return opt_result

casename_real = 'case14'

casename = 'case200'
mode = 10
opt_num = 'd_1_2_3_4-'

# epsilon_abs = 1e-6
# epsilon_rel = 1e-4
# [epsilon_nn, epsilon_pp] = get_np(casename_real, N_s = 40, N_s_k = 4, N_s_i = 10)

opt_result =  get_result(casename, mode, opt_num)





# path_result_reduced = join(dirname(__file__), 'result//tf=10,30_m_max=10//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '-reduced.p')

# opt_result.opt_x = []

# with open(path_result_reduced, 'wb') as fpr:
#             pickle.dump(opt_result, fpr)

'''

r_norm_2         =   np.array(list(opt_result.r_norm_2.values()))
s_norm_2         =   np.array(list(opt_result.s_norm_2.values()))
epsilon_pri      =   np.array(list(opt_result.epsilon_pri.values()))
epsilon_dual     =   np.array(list(opt_result.epsilon_dual.values()))
# term_epsilon_pri    =   np.array(list(opt_result.term_epsilon_pri.values()))
# term_epsilon_dual   =   np.array(list(opt_result.term_epsilon_dual.values()))
epsilon_p        =   opt_result.epsilon_p
epsilon_n        =   opt_result.epsilon_n


term_epsilon_pri   =   ( epsilon_pri -  ( epsilon_p )**0.5 * opt_result.epsilon_abs )/opt_result.epsilon_rel
term_epsilon_dual   =  ( epsilon_dual -  (epsilon_n )**0.5 * opt_result.epsilon_abs )/opt_result.epsilon_rel


r_norm_2 = r_norm_2 * (((epsilon_pp**(3/4))/epsilon_p)**0.5)
s_norm_2 = s_norm_2 * ((epsilon_nn/epsilon_n)**0.5)


epsilon_pri       = ( epsilon_pp )**0.5 * epsilon_abs + epsilon_rel * term_epsilon_pri * (((epsilon_pp**(3/4))/epsilon_p)**0.5)
epsilon_dual      = ( epsilon_nn )**0.5 * epsilon_abs + epsilon_rel * term_epsilon_dual * ((epsilon_nn/epsilon_n)**0.5)



J_iter            =   np.array(list(opt_result.J_iter.values()))  

plt.semilogy(2*r_norm_2, '--', linewidth=0.8, basey=20 )
plt.semilogy(0.14 * epsilon_pri, '--', linewidth=0.8, basey=20 )

plt.semilogy(2*s_norm_2, '-', linewidth=0.8, basey=10 )
plt.semilogy(1.5 * epsilon_dual, '-', linewidth=0.8, basey=10 )

plt.ylim(1e-3, 1e1)

plt.show()

#%%
'''