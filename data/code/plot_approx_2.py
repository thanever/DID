import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.colors as colors
from os.path import dirname, join
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


plt.style.use('default')
plt.rc('text', usetex=True)
plt.rc('font', size=8,family='Times New Roman') 

plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['xtick.major.width'] = 0.4

def ff(x_b, x_c):
        
    pi = np.pi
    x_v= (1 - np.sin(x_b))/(x_b - pi/2)**2
    x_1 = np.linspace(-x_c, -x_b, 200 )
    x_2 = np.linspace(-x_b, x_b,200)
    x_3 = np.linspace(x_b, x_c, 200)

    y1 =  x_v * (x_1**2) + pi * x_v * x_1 + ((pi**2)/4)*x_v - 1
    y2 = np.sin(x_b)/x_b * x_2
    y3 = - x_v * (x_3**2) +  pi * x_v * x_3 - ((pi**2)/4)*x_v + 1

    r1 = (np.sin(x_1) - y1)**2
    r2 = (y2- np.sin(x_2))**2
    r3 = (np.sin(x_3) - y3)**2

    r = r1.sum() * (x_c - x_b)/200  + r2.sum() * 2* x_b/200 + r3.sum()*(x_c - x_b)/200
    return r


X, C = np.mgrid[0.000001 : pi/2-0.0001 : 0.001, np.pi/2+0.000001 : np.pi : 0.001]
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = ff(X[i,j], C[i,j])


fig = plt.figure(0, figsize=(3, 2))
ax = fig.gca(projection='3d')

ax.plot_surface(X , C, Z, cmap='nipy_spectral', alpha= 0.8)

ax.plot(X[Z.argmin(axis=0),0], C[0,:], Z.min(axis=0) , c='black', alpha =1, linewidth=1)

ax.set_xlim(0, np.pi/2)
ax.set_xticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.57])
ax.set_xticklabels(['0', '0.3', '0.6', '0.9', '1.2', '$\\pi/2$'], va='baseline',rotation=-50,ha='right')
ax.set_ylim(np.pi/2, np.pi+0.1)
ax.set_yticks([1.57, 1.9, 2.2, 2.5, 2.8, 3.1415])
ax.set_yticklabels(['$\\pi/2$', '1.9', '2.2', '2.5', '2.8', '$~~\\pi$'], va='bottom',rotation=20,ha='left')

ax.set_zticks([0, 0.02, 0.04, 0.06, 0.08])
ax.set_zticklabels([0.00, 0.02, 0.04, 0.06, 0.08], va='top',rotation=0,ha='left')
ax.set_zlim(0,0.085)

ax.view_init(26, -119)
ax.w_xaxis.set_pane_color((1, 1, 1, 1))
ax.w_yaxis.set_pane_color((1, 1, 1, 1))
ax.w_zaxis.set_pane_color((1, 1, 1, 1))

fig.text(0.7,0.08,"$\\theta_b$")
fig.text(0.2,0.18, "$\\theta_c$")
fig.text(0.1,0.5, "$\\varepsilon$")


plt.show()
path_fig= join(dirname(__file__), 'result//'  + 'approximation2_1.pdf')
fig.savefig(path_fig,dpi = 300, transparent=True, bbox_inches='tight')


fig1 = plt.figure(1, figsize=(2.5, 1.2))

host = HostAxes(fig1, [0.15, 0.1, 0.65, 0.8])
par1 = ParasiteAxes(host, sharex=host)
host.parasites.append(par1)

host.axis["right"].set_visible(False)
par1.axis["right"].set_visible(True)

par1.axis["right"].major_ticklabels.set_visible(True)
par1.axis["right"].label.set_visible(True)
fig1.add_axes(host)

host.set_xlim(np.pi/2, np.pi)
host.set_ylim(0.29, 0.61)
host.set_xticks([1.57, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.1415])
host.set_xticklabels(['$\\pi/2$', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8', '3.0', '$\\pi$'])

host.set_xlabel("$\\theta_c$")
host.set_ylabel("$\\theta_b^*$")
par1.set_ylabel("min$_{\\theta_b} \\varepsilon$ $(\\times 10^{-3})$")


p1, = host.plot(C[0,:], X[Z.argmin(axis=0),0], label="$\\theta_b^*$", color = '#1f77b4' )
p2, = par1.plot(C[0,:], Z.min(axis=0) * 1000, label="min$_{\\theta_b} \\varepsilon$", color = '#e6550d')
par1.set_ylim(0, 2.5)


host.legend(loc = 'center left')
host.grid(linestyle='--')
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.show()

path_fig1= join(dirname(__file__), 'result//'  + 'approximation2_2.pdf')
fig1.savefig(path_fig1,dpi = 300, transparent=True, bbox_inches='tight')
