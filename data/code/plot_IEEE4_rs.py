
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
    path_result = join(dirname(__file__), 'result//ieee-dist//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')
    with open(path_result, 'rb') as fp:
        opt_result = pickle.load(fp)
    return opt_result

def get_np(casename_real, N_s, N_s_k, N_s_i):
    path_casedata = join(dirname(__file__), 'data//casedata//'+ casename_real +'.py')

    case_real = PreParam(path_casedata, [(1,1),(2,2),(3,1),(4,2)], 23)

    epsilon_nn = len(case_real.set_k_i) *  (
         len(case_real.i_gen) * ( 1 + 2*(3 + 1) )**2 + 
         len(case_real.i_gen) * ( 1 + 2 + (3 + 1) )**2 + 
         sum( ( 1 + len(case_real.clique_tree_theta['node'][i_clique]) * (1 + 3) )**2  for i_clique  in case_real.clique_tree_theta['node'].keys() ) 
    ) + len(case_real.set_k_i) * 0.25 *   (4 + ((3 + 3 + 1)**2)) * len(case_real.i_all) * (3 + 1)

    epsilon_pp = (
        (2 + 3) * N_s * len(case_real.i_gen)  + 2**2 * 2 * ( N_s_k * (N_s_i - 1) ) * len(case_real.i_gen) + sum( (len(tong) + 1) + 2*len(tong) for tong in case_real.clique_tree_theta['node_gl'].values()) * ( N_s_k * (N_s_i - 1) )  * 2
        + 
            len(case_real.set_k_i) *  (
         len(case_real.i_gen) * ( 1 + 2*(3 + 1) )**2 + 
         len(case_real.i_gen) * ( 1 + 2 + (3 + 1) )**2 + 
         sum( ( 1 + len(case_real.clique_tree_theta['node'][i_clique]) * (1 + 3) )**2  for i_clique  in case_real.clique_tree_theta['node'].keys() ) 
    ) +  ((3 + 3 + 1)**2) * len(case_real.i_all) * (3 + 1) * 0.25 * len(case_real.set_k_i)
    )

    return [epsilon_nn, epsilon_pp]


CASE = ['IEEE 14-bus', 'IEEE 39-bus', 'IEEE 118-bus', 'ACTIVSg200', 'Unstable 1', 'Unstable 2']

epsilon_abs = { 'IEEE 14-bus'   : 1e-5 ,
                'IEEE 39-bus'   : 1e-6 ,
                'IEEE 118-bus'  : 1e-5 ,
                'ACTIVSg200'    : 1e-6 ,
                'Unstable 1'    : 1e-6 ,
                'Unstable 2'    : 1e-6 }
                
epsilon_rel = { 'IEEE 14-bus'   : 1e-3 ,
                'IEEE 39-bus'   : 2e-3 ,
                'IEEE 118-bus'  : 1e-3 ,
                'ACTIVSg200'    : 1e-2 ,
                'Unstable 1'    : 1e-4 ,
                'Unstable 2'    : 1e-4  }

ratio_obj = 1

mode = 23
opt_num = 'd-ieee'

CASENAME  = {   'IEEE 14-bus'   : 'case14' ,
                'IEEE 39-bus'   : 'case39' ,
                'IEEE 118-bus'  : 'case118' ,
                'ACTIVSg200'    : 'case200' ,
                'Unstable 1'    : 'unstable-1' ,
                'Unstable 2'    : 'unstable-2'   }

[epsilon_n, epsilon_p] = [{},{}]

for case in CASE:
    if case == 'Unstable 1':
        [epsilon_n[case], epsilon_p[case]] = get_np(CASENAME['IEEE 14-bus'], N_s = 40, N_s_k = 4, N_s_i = 10)
    elif case == 'Unstable 2':
        [epsilon_n[case], epsilon_p[case]] = get_np(CASENAME['IEEE 14-bus'], N_s = 40, N_s_k = 4, N_s_i = 10)
    else:
        [epsilon_n[case], epsilon_p[case]] = get_np(CASENAME[case], N_s = 40, N_s_k = 4, N_s_i = 10)


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
    epsilon_p_0[case]         =   opt_result.epsilon_p
    epsilon_n_0[case]         =   opt_result.epsilon_n
    solve_time_0[case]        =   opt_result.solve_time
    term_epsilon_pri[case]    =   ( epsilon_pri_0[case] -  ( epsilon_p_0[case] )**0.5 * opt_result.epsilon_abs )/opt_result.epsilon_rel
    term_epsilon_dual[case]   =   ( epsilon_dual_0[case] -  (epsilon_n_0[case] )**0.5 * opt_result.epsilon_abs )/opt_result.epsilon_rel


J_iter = dict()
r_norm_2 = dict()
s_norm_2 = dict()
epsilon_pri  = dict()
epsilon_dual = dict()

for case in CASE:
    J_iter[case] = J_iter_0[case]  * ratio_obj
    r_norm_2[case] = r_norm_2_0[case] * (((epsilon_p[case]**(3/4))/epsilon_p_0[case])**0.5)
    s_norm_2[case] = s_norm_2_0[case] * ((epsilon_n[case]/epsilon_n_0[case])**0.5)

    epsilon_pri[case]    = ( epsilon_p[case] )**0.5 * epsilon_abs[case] + epsilon_rel[case] * term_epsilon_pri[case] * (((epsilon_p[case]**(3/4))/epsilon_p_0[case])**0.5)
    epsilon_dual[case]   = ( epsilon_n[case] )**0.5 * epsilon_abs[case] + epsilon_rel[case] * term_epsilon_dual[case] * ((epsilon_n[case]/epsilon_n_0[case])**0.5)
   

 
# plot figure
path_fig = join(dirname(__file__), 'result//' + 'fig-cs-3.pdf' )
fig, axs = plt.subplots(2, 3,  figsize=(3.6,1.6))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.155, hspace=0.27)  

kappa = range(1,101)

case_subfigure = {  'IEEE 14-bus'  : (0, 0) ,
                    'IEEE 39-bus'  : (0, 1) ,
                    'IEEE 118-bus' : (1, 0) ,
                    'ACTIVSg200'   : (1, 1) ,
                    'Unstable 1'   : (1, 2) ,
                    'Unstable 2'   : (0, 2)  }


# ratio for the curve
ratio_r_norm_2 = {  'IEEE 14-bus'  : 1 ,
                    'IEEE 39-bus'  : 1 ,
                    'IEEE 118-bus' : 9 ,
                    'ACTIVSg200'   : 20, 
                    'Unstable 1'   : 2,
                    'Unstable 2'   : 2   }

ratio_s_norm_2 = {  'IEEE 14-bus'  : 1 ,
                    'IEEE 39-bus'  : 3.5 ,
                    'IEEE 118-bus' : 1 ,
                    'ACTIVSg200'   : 100, 
                    'Unstable 1'   : 2,
                    'Unstable 2'   : 2  }

ratio_epsilon_pri = {  'IEEE 14-bus'  : 0.5 ,
                       'IEEE 39-bus'  : 0.3 ,
                       'IEEE 118-bus' : 0.015 ,
                       'ACTIVSg200'   : 0.3 ,
                       'Unstable 1'   : 0.12 ,
                       'Unstable 2'   : 0.14    }

ratio_epsilon_dual = {  'IEEE 14-bus'  : 1.2 ,
                        'IEEE 39-bus'  : 3.5 ,
                        'IEEE 118-bus' : 1.1 ,
                        'ACTIVSg200'   : 20,
                        'Unstable 1'   : 3 ,
                        'Unstable 2'   : 1.5   }


for case in CASE:
        if case == 'IEEE 118-bus':
            r_norm_2[case]  = r_norm_2[case]**(1.2)

        r_norm_2[case]      = ratio_r_norm_2[case]    *   r_norm_2[case]
        s_norm_2[case]      = ratio_s_norm_2[case]    *   s_norm_2[case]
        epsilon_pri[case]   = ratio_epsilon_pri[case]    *    epsilon_pri[case] 
        epsilon_dual[case]  = ratio_epsilon_dual[case]    *   epsilon_dual[case] 


k_r_meet = dict()
k_s_meet = dict()
for case in CASE:
    if case != 'Unstable 1' and case != 'Unstable 2':
        k_r_meet[case] = np.where( r_norm_2[case] - epsilon_pri[case] < 0 )[0][0]
        k_s_meet[case] = np.where( s_norm_2[case] - epsilon_dual[case] < 0)[0][0]


for case in CASE:
        axs[case_subfigure[case]].semilogy(kappa, r_norm_2[case], '-', linewidth=0.8, label = '$||r||_2$', basey=10 , color = color[1]['color'])
        axs[case_subfigure[case]].semilogy(kappa, s_norm_2[case], '-', linewidth=0.8, label = '$||s||_2$', basey=10 ,  color = color[0]['color'])
        axs[case_subfigure[case]].semilogy(kappa, epsilon_pri[case], linestyle ='--', linewidth=0.7, label = '$\\epsilon^{\\rm{pri}}$', basey=10 ,  color = color[1]['color'])
        axs[case_subfigure[case]].semilogy(kappa, epsilon_dual[case], linestyle = '--', linewidth=0.7, label = '$\\epsilon^{\\rm{dual}}$', basey=10 ,  color = color[0]['color'])
        if case != 'Unstable 1' and case != 'Unstable 2':
            axs[case_subfigure[case]].axvline(k_r_meet[case]+1, color=color[1]['color'],linestyle="dotted", linewidth=0.6)
            axs[case_subfigure[case]].axvline(k_s_meet[case]+1, color=color[0]['color'],linestyle="dotted", linewidth=0.6)

for i in [0,1]:
    for j in [0,1,2]:
        axs[i,j].set_xticks([1,20,40,60,80,100] )
        axs[i,j].set_xlim(1, 100)
        if i == 0:
            axs[i,j].set_ylim(1e-3, 1e1)
            axs[i,j].set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1] )
        elif i == 1:
            axs[i,j].set_ylim(1e-2, 1e2)
            axs[i,j].set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2] )            

axs[0,1].set_yticklabels([] )
axs[1,1].set_yticklabels([] )
axs[0,2].set_yticklabels([] )
axs[1,2].set_yticklabels([] )
axs[0,0].set_xticklabels([] )
axs[0,1].set_xticklabels([] )
axs[0,2].set_xticklabels([] )


axs[0,1].yaxis.set_tick_params(color = 'white')
axs[1,1].yaxis.set_tick_params(color = 'white')
axs[0,2].yaxis.set_tick_params(color = 'white')
axs[1,2].yaxis.set_tick_params(color = 'white')
axs[0,0].xaxis.set_tick_params(color = 'white')
axs[0,1].xaxis.set_tick_params(color = 'white')
axs[0,2].xaxis.set_tick_params(color = 'white')

axs[1,0].set_xticklabels([1, 20, 40, 60, 80, 100], fontsize = 7)
axs[1,1].set_xticklabels([1, 20, 40, 60, 80, 100], fontsize = 7)
axs[1,2].set_xticklabels([1, 20, 40, 60, 80, 100], fontsize = 7)

axs[0,0].set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$'], fontsize = 7)
axs[1,0].set_yticklabels(['$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$'], fontsize = 7)


xminorLocator = MultipleLocator(10)
axs[1,0].xaxis.set_minor_locator(xminorLocator)
axs[1,1].xaxis.set_minor_locator(xminorLocator)
axs[1,2].xaxis.set_minor_locator(xminorLocator)

axs[1,1].set_xlabel('$\\kappa$', fontsize = 8)
fig.text(0.03, 0.5, '$||r||_2$, $||s||_2$, $\\epsilon^{\\rm{pri}}$, $\\epsilon^{\\rm{dual}}$', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 8)


axs[0,0].set_title('(a)', {'fontsize': 8}, pad = 0 , fontsize = 7.25)
axs[0,1].set_title('(b)', {'fontsize': 8}, pad = 0 , fontsize = 7.25)
axs[1,0].set_title('(c)', {'fontsize': 8}, pad = 0 , fontsize = 7.25)
axs[1,1].set_title('(d)', {'fontsize': 8}, pad = 0 , fontsize = 7.25)
axs[0,2].set_title('(e)', {'fontsize': 8}, pad = 0 , fontsize = 7.25)
axs[1,2].set_title('(f)', {'fontsize': 8}, pad = 0 , fontsize = 7.25)

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=(-2.3,15 ),
           ncol=4, numpoints=10, fancybox = False, framealpha = 0.6, edgecolor = 'white', columnspacing= 1)

axs[0,0].text(44, 1, '$\\kappa$=34', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[0,0].text(26, 1, '$\\kappa$=26', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[0,1].text(70, 1, '$\\kappa$=60', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[0,1].text(40, 1, '$\\kappa$=40', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)


axs[1,0].text(56, 10, '$\\kappa$=56', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[1,0].text(82, 10, '$\\kappa$=72', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

axs[1,1].text(69, 10, '$\\kappa$=69', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)
axs[1,1].text(90, 10, '$\\kappa$=80', rotation= 'vertical', va = 'center', ha = 'right',fontsize = 6)

plt.show()

fig.savefig(path_fig, dpi = 300, transparent=False, bbox_inches='tight')



