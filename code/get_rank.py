
# %%
import matplotlib.pyplot as plt
import pickle
from os.path import dirname, join
import os
import numpy as np
from make_parameter import PreParam

plt.style.use('ggplot')


def get_result(casename, mode, opt_num):
    '''
    load the opt results from saved file
    '''
    path_result = join(dirname(__file__), 'result//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')
    with open(path_result, 'rb') as fp:
        opt_result = pickle.load(fp)
    return opt_result



casename = 'case59_1'
mode = 23
opt_num = 'd_1_2_3_4-reduced'


opt_result =  get_result(casename, mode, opt_num)


matrix_all = list(opt_result.X_lmd.values()) + list(opt_result.X_mdw.values())  + list(opt_result.X_theta.values())

rank_all = list()
for mat in matrix_all:
    rank_all.append( np.linalg.matrix_rank(mat.value, tol = 1e-5) )

print(max(rank_all))


## information of the nummber of cliques and maximal size of cliques
# casename = 'case200'

# path_casedata = join(os.path.abspath('') , 'data//casedata//'+ casename +'.py')

# casedata = PreParam(path_casedata, [(1,1),(2,2),(3,1),(4,2)], 10)

# print(max([len(i) for i in casedata.clique_tree_theta['node'].values()]))

# print(len(casedata.clique_tree_theta['node']) )

