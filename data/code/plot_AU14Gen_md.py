# -*- coding: utf-8 -*-
"""
plot the figure in case study, showing the dispatch result of AU14Gen power system under 6 operating conditions 

@author: eee
"""

from os.path import dirname, join
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('ggplot')

plt.rc('text', usetex=True)
plt.rc('font', size=8,family='Times New Roman')

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'out' 
plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'black'

COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def get_result(casename, mode, opt_num):
    '''
    load the opt results from saved file
    '''
    path_result = join(dirname(__file__), 'result//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')
    with open(path_result, 'rb') as fp:
        opt_result = pickle.load(fp)
    return opt_result

def get_md(opt_result):

    x_m = dict()
    x_d = dict()
    x_m.update(opt_result.opt_result.m.get_values())
    x_d.update(opt_result.opt_result.d.get_values())

    return {'m':x_m, 'd':x_d}



CaseName = ['case59_1', 'case59_2', 'case59_3', 'case59_4', 'case59_5', 'case59_6']
mode = 10
opt_num = 'd_1_2_3_4'

md = dict()
for casename in CaseName:
    opt_result =  get_result(casename, mode, opt_num)
    md[casename] = get_md(opt_result)

Gen = opt_result.i_gen
m_l = dict(zip(opt_result.casedata['gencontrol'][:,0] , opt_result.casedata['gencontrol'][:,2] ))
m_u = dict(zip(opt_result.casedata['gencontrol'][:,0] , opt_result.casedata['gencontrol'][:,3] ))
d_l = dict(zip(opt_result.casedata['gencontrol'][:,0] , opt_result.casedata['gencontrol'][:,4] ))
d_u = dict(zip(opt_result.casedata['gencontrol'][:,0] , opt_result.casedata['gencontrol'][:,5] ))


path_fig = join(dirname(__file__), 'result//' + 'fig-cs-1.pdf' )
fig, (axs1, axs2) = plt.subplots(2,1, sharex = True, figsize=(3.6,2.34))

x_tick = [1/12 + 1/6 * i for i in range(6)]
color = list(plt.rcParams['axes.prop_cycle'])


Gen_name = [101,201,202,203,204,301,302,401,402,403,404,501,502,503]

for i_gen in range(0, 1):
        for i_case in range(len(CaseName)):
                casename = CaseName[i_case]
                axs1.scatter(i_gen + x_tick[i_case],  md[casename]['m'][Gen[i_gen]] , s=4 , marker='o', c= color[i_case]['color'], label= 'Case '+ str(i_case + 1))
                axs2.scatter(i_gen + x_tick[i_case],  md[casename]['d'][Gen[i_gen]] , s=4 , marker='o', c= color[i_case]['color'])

for i_gen in range(1, len(Gen)):
        for i_case in range(len(CaseName)):
                casename = CaseName[i_case]
                axs1.scatter(i_gen + x_tick[i_case],  md[casename]['m'][Gen[i_gen]] , s=4 , marker='o', c= color[i_case]['color'])
                axs2.scatter(i_gen + x_tick[i_case],  md[casename]['d'][Gen[i_gen]] , s=4 , marker='o', c= color[i_case]['color'])


axs2.set_xlabel('Generator bus', fontsize=8)
axs1.set_ylabel('$m_i$ (p.u.$\\cdot$s$^2/$rad$^2$)' , fontsize=8)
axs2.set_ylabel('$d_i$ (p.u.$\\cdot$s/rad)', fontsize=8)

axs2.set_xticks(range(14))
for i in range(len(Gen_name)): axs2.text(i + 0.1 ,0.35, str(Gen_name[i]), va='center', ma= 'center')
axs1.set_yticks([0,0.5, 1, 1.5])
axs2.set_yticks([1,3,5,7])

axs1.set_xlim(0, 14)
axs1.set_ylim(-0.1, 1.5)
axs2.set_ylim(1, 7.4)

axs1.legend(ncol = 3, fontsize = 'x-small', borderpad = 0.2, handletextpad = 0.2, columnspacing = 0.2)

fig.tight_layout() 
plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
plt.show()
fig.savefig(path_fig,dpi = 300, transparent=False, bbox_inches='tight')



