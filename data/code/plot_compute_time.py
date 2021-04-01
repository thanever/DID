
import matplotlib.pyplot as plt
from os.path import dirname, join
import numpy as np
from matplotlib.ticker import AutoMinorLocator


plt.style.use('default')

plt.rc('text', usetex=True)
plt.rc('font', size=8,family='Times New Roman')

# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'out' 
plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['xtick.major.width'] = 0.4
plt.rcParams['xtick.minor.width'] = 0.4

plt.rcParams['xtick.major.pad']= '0.1'

plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

color = list(plt.rcParams['axes.prop_cycle'])



t_nlp = {'IEEE 14-bus'  :5.989    ,
         'IEEE 39-bus'  :39.743   ,
         'Case 1'       :224.947  ,
         'Case 2'       :228.643  ,
         'Case 3'       :4152.681 ,
         'Case 4'       :129.279  ,
         'Case 5'       :150.299  ,
         'Case 6'       :234.872  ,
         'IEEE 118-bus' :1696.234 ,
         'ACTIVSg200'   :4336.744 }

t_sdp = {'IEEE 14-bus'  :76.5875  ,
         'IEEE 39-bus'  :360.2125 ,
         'Case 1'       :577.632  ,
         'Case 2'       :601.223  ,
         'Case 3'       :581.33   ,
         'Case 4'       :615.32   ,
         'Case 5'       :563.12   ,
         'Case 6'       :572.925  ,
         'IEEE 118-bus' :1981.1396,
         'ACTIVSg200'   :5512.4432}


t_feda = {'IEEE 14-bus' :122.07749,
         'IEEE 39-bus'  :538.05742,
         'Case 1'       :592.23020,
         'Case 2'       :579.76355,
         'Case 3'       :755.08793,
         'Case 4'       :815.45229,
         'Case 5'       :673.64508,
         'Case 6'       :583.75829,
         'IEEE 118-bus' :3456.3729,
         'ACTIVSg200'   :9816.9232,}



ind = np.arange (len(t_nlp))

width = 0.24

# plot figure
path_fig = join(dirname(__file__), 'result//' + 'fig-cs-4.pdf' )
fig, ax = plt.subplots(figsize=(3.6,1.6))

ax.bar(ind - 1.15 * width, t_nlp.values(),  width, label = 'NLP'  ,color = '#5086FF', alpha = 0.9)
ax.bar(ind              , t_sdp.values(),  width, label = 'SDP'   ,color = '#17B12B',  alpha = 0.9)
ax.bar(ind + 1.15 * width, t_feda.values(), width, label = 'FEDA' ,color = '#F35E5A',  alpha = 0.9)


ax.set_ylabel('Time (second)', fontsize = 7)

ax.set_xticks(ind + 0.2 )
ax.xaxis.set_tick_params(color = 'white')
ax.set_xticklabels(('IEEE 14-bus' , 
                    'IEEE 39-bus' , 
                    'Case 1'  ,  
                    'Case 2'  ,  
                    'Case 3'  ,  
                    'Case 4'  ,  
                    'Case 5'  ,  
                    'Case 6'  ,  
                    'IEEE 118-bus' ,
                    'ACTIVSg200'  ),  va = 'top' , ha = 'right', rotation = 30, fontsize = 7)
 

# ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=(-2.3,15 ),
#            ncol=4, numpoints=10, fancybox = False, framealpha = 0.6, edgecolor = 'white', columnspacing= 1)

ax.set_yscale('log')

ax.set_xlim(-0.7,9.7)
ax.set_ylim(1, 1e4)

ax.set_yticks((1, 1e1, 1e2, 1e3, 1e4))

for i in ind:
    ax.text(i - 1.15 * width - 0.07, list(t_nlp.values())[i], str(round(list(t_nlp.values())[i],2)) , rotation= 'vertical', va = 'top', ha = 'left',fontsize = 4)
    ax.text(i                - 0.07, list(t_sdp.values())[i], str(round(list(t_sdp.values())[i],2)) , rotation= 'vertical', va = 'top', ha = 'left',fontsize = 4)
    ax.text(i + 1.15 * width - 0.07, list(t_feda.values())[i], str(round(list(t_feda.values())[i],2)) , rotation= 'vertical', va = 'top', ha = 'left',fontsize = 4)



ax.legend(loc=(0.01,0.87), ncol=3, numpoints=1, fancybox = False, framealpha = 0, edgecolor = 'white', columnspacing= 0.8, borderpad = 0.02, labelspacing=0.1)


ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.grid(axis = 'y', linestyle='dotted', alpha= 0.5)



fig.tight_layout()

plt.show()

fig.savefig(path_fig, dpi = 300, transparent=False, bbox_inches='tight')
