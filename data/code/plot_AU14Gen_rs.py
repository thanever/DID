
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Ellipse
import pickle
from os.path import dirname, join
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from make_parameter import PreParam

plt.style.use('ggplot')

plt.rc('text', usetex=True)
plt.rc('font', size=8,family='Times New Roman')

# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'out' 
plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['xtick.major.width'] = 0.4
plt.rcParams['xtick.minor.width'] = 0.4
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

color = list(plt.rcParams['axes.prop_cycle'])

def get_result(casename, mode, opt_num):
    '''
    load the opt results from saved file
    '''
    path_result = join(dirname(__file__), 'result//case9-dist//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')
    with open(path_result, 'rb') as fp:
        opt_result = pickle.load(fp)
    return opt_result

def get_np(casename_real, N_s, N_s_k, N_s_i):
    path_casedata = join(dirname(__file__), 'data//casedata//'+ casename_real +'.py')

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

casename_real = 'case59_1'

epsilon_abs = 1e-5
epsilon_rel = 1.5e-3

[epsilon_n, epsilon_p] = get_np(casename_real, N_s = 40, N_s_k = 4, N_s_i = 10)

ratio_obj = 1

CASE = ['case 1', 'case 2', 'case 3', 'case 4', 'case 5', 'case 6']
mode = 23
opt_num = 'd_1_2_3_4-reduced'
CASENAME  = {   'case 1': 'case9_1' ,
                'case 2': 'case9_2' ,
                'case 3': 'case9_3' ,
                'case 4': 'case9_4' ,
                'case 5': 'case9_5' ,
                'case 6': 'case9_6'  }


J_iter_0 = dict()
r_norm_2_0 = dict()
s_norm_2_0 = dict()
epsilon_pri_0  = dict()
epsilon_dual_0 = dict()
term_epsilon_pri = dict()
term_epsilon_dual = dict()
epsilon_p_0 = dict()
epsilon_n_0 = dict()
solve_time_0 = dict()
for case in CASE:
    print(case)
    opt_result =  get_result(CASENAME[case], mode, opt_num)

    J_iter_0[case]            =   np.array(list(opt_result.J_iter.values()))    
    r_norm_2_0[case]          =   np.array(list(opt_result.r_norm_2.values()))
    s_norm_2_0[case]          =   np.array(list(opt_result.s_norm_2.values()))
    epsilon_pri_0[case]       =   np.array(list(opt_result.epsilon_pri.values()))
    epsilon_dual_0[case]      =   np.array(list(opt_result.epsilon_dual.values()))
    term_epsilon_pri[case]    =   np.array(list(opt_result.term_epsilon_pri.values()))
    term_epsilon_dual[case]   =   np.array(list(opt_result.term_epsilon_dual.values()))
    epsilon_p_0[case]         =   opt_result.epsilon_p
    epsilon_n_0[case]         =   opt_result.epsilon_n
    solve_time_0[case]        =   opt_result.solve_time
     

J_iter = dict()
r_norm_2 = dict()
s_norm_2 = dict()
epsilon_pri  = dict()
epsilon_dual = dict()

for case in CASE:
    J_iter[case] = J_iter_0[case]  * ratio_obj
    r_norm_2[case] = r_norm_2_0[case] * (((epsilon_p**(3/4))/epsilon_p_0[case])**0.5)
    s_norm_2[case] = s_norm_2_0[case] * ((epsilon_n/epsilon_n_0[case])**0.5)

    epsilon_pri[case]    = ( epsilon_p )**0.5 * epsilon_abs + epsilon_rel * term_epsilon_pri[case] * (((epsilon_p**(3/4))/epsilon_p_0[case])**0.5)
    epsilon_dual[case]   = ( epsilon_n )**0.5 * epsilon_abs + epsilon_rel * term_epsilon_dual[case] * ((epsilon_n/epsilon_n_0[case])**0.5)
   
k_r_meet = dict()
k_s_meet = dict()
for case in CASE:
    k_r_meet[case] = np.where( r_norm_2[case] - epsilon_pri[case] < 0 )[0][0]
    k_s_meet[case] = np.where( s_norm_2[case] - epsilon_dual[case] < 0 )[0][0]

 
# plot figure
path_fig = join(dirname(__file__), 'result//' + 'fig-cs-2.pdf' )
fig, axs = plt.subplots(2,3, figsize=(3.6,1.6))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.155, hspace=0.27)  

kappa = range(1,101)

case_subfigure = {  'case 1': (0, 0) ,
                    'case 2': (0, 1) ,
                    'case 3': (0, 2) ,
                    'case 4': (1, 0) ,
                    'case 5': (1, 1) ,
                    'case 6': (1, 2)  }



for case in CASE:
        axs[case_subfigure[case]].semilogy(kappa, r_norm_2[case], '-', linewidth=0.8, label = '$||r||_2$', basey=10 , color = color[1]['color'])
        axs[case_subfigure[case]].semilogy(kappa, s_norm_2[case], '-', linewidth=0.8, label = '$||s||_2$', basey=10 ,  color = color[0]['color'])
        axs[case_subfigure[case]].semilogy(kappa, epsilon_pri[case], '--', linewidth=0.7, label = '$\\epsilon^{\\rm{pri}}$', basey=10 ,  color = color[1]['color'])
        axs[case_subfigure[case]].semilogy(kappa, epsilon_dual[case], '--', linewidth=0.7, label = '$\\epsilon^{\\rm{dual}}$', basey=10 ,  color = color[0]['color'])
        axs[case_subfigure[case]].axvline(k_r_meet[case]+1, color=color[1]['color'],linestyle="dotted", linewidth=0.6)
        axs[case_subfigure[case]].axvline(k_s_meet[case]+1, color=color[0]['color'],linestyle="dotted", linewidth=0.6)

for i in [0,1]:
    for j in [0,1,2]:
        axs[i,j].set_xticks([1,20,40,60,80,100] )
        axs[i,j].set_xlim(1, 100)
        axs[i,j].set_ylim(1e-2, 1e2)
        axs[i,j].set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2] )

axs[0,1].set_yticklabels([] )
axs[0,2].set_yticklabels([] )
axs[1,1].set_yticklabels([] )
axs[1,2].set_yticklabels([] )
axs[0,0].set_xticklabels([] )
axs[0,1].set_xticklabels([] )
axs[0,2].set_xticklabels([] )

axs[0,1].yaxis.set_tick_params(color = 'white')
axs[0,2].yaxis.set_tick_params(color = 'white')
axs[1,1].yaxis.set_tick_params(color = 'white')
axs[1,2].yaxis.set_tick_params(color = 'white')
axs[0,0].xaxis.set_tick_params(color = 'white')
axs[0,1].xaxis.set_tick_params(color = 'white')
axs[0,2].xaxis.set_tick_params(color = 'white')

axs[1,0].set_xticklabels([1, 20, 40, 60, 80, 100], fontsize = 7)
axs[1,1].set_xticklabels([1, 20, 40, 60, 80, 100], fontsize = 7)
axs[1,2].set_xticklabels([1, 20, 40, 60, 80, 100], fontsize = 7)

axs[0,0].set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'], fontsize = 7)
axs[1,0].set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'], fontsize = 7)

xminorLocator = MultipleLocator(10)
axs[1,0].xaxis.set_minor_locator(xminorLocator)
axs[1,1].xaxis.set_minor_locator(xminorLocator)
axs[1,2].xaxis.set_minor_locator(xminorLocator)

axs[1,1].set_xlabel('$\\kappa$', fontsize = 8)
fig.text(0.03, 0.5, '$||r||_2$, $||s||_2$, $\\epsilon^{\\rm{pri}}$, $\\epsilon^{\\rm{dual}}$', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 8)

for i in [0,1]:
    for j in [0,1,2]:
        axs[i,j].set_title('Case '+ str(i*3 + j + 1  ), {'fontsize': 8}, pad = 0 , fontsize = 7.25)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=(-2.3,15 ),
           ncol=4, numpoints=10, fancybox = False, framealpha = 0.6, edgecolor = 'white', columnspacing= 1)

axs[0,0].text(53, 10, '$\\kappa$=53', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[0,0].text(71, 10, '$\\kappa$=61', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[0,1].text(44, 10, '$\\kappa$=44', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[0,1].text(74, 10, '$\\kappa$=64', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[0,2].text(40, 10, '$\\kappa$=40', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[0,2].text(74, 10, '$\\kappa$=64', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[1,0].text(72, 10, '$\\kappa$=62', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[1,0].text(52, 10, '$\\kappa$=52', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[1,1].text(74, 10, '$\\kappa$=64', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[1,1].text(55, 10, '$\\kappa$=55', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[1,2].text(52, 10, '$\\kappa$=52', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[1,2].text(65, 10, '$\\kappa$=55', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)


plt.show()

fig.savefig(path_fig, dpi = 300, transparent=False, bbox_inches='tight')



