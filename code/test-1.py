
import numpy as np 
import scipy as sp
from sksparse.cholmod import cholesky
import networkx as nx

from os.path import dirname, join
from pypower.api import makeYbus, runpf, loadcase, ext2int, ppoption
from networkx.algorithms import approximation
from copy import deepcopy
from itertools import combinations

casename = 'case200'

path_casedata = join(dirname(__file__), 'data//casedata//'+ casename +'.py')
casedata = loadcase(path_casedata)



####code for generate the extra parameters
# last two columns of the bus data
for i in range(len(casedata['bus'])):
    if casedata['bus'][i, 1] == 2 or casedata['bus'][i, 1] == 3:
        print( '0.01', ',\t' , '0.01', '\t' )
    else:
        if casedata['bus'][i, 2] !=0:
            print( '0', ',\t' , '0.01', '\t' )
        else:
            print( '0', ',\t' , '0', '\t' )

# gen_control data
for i in range(len(casedata['gen'])):
    max_pp = casedata['gen'][i,8]/100
    max_m = round( (2 * 5 * max_pp)/(2 *  np.pi * 50)       , 6)
    min_m = round( 0.01 * max_m     , 6)
    max_d = round( max_pp * 2/(2 * np.pi )   , 6)
    min_d = round( 0.01 * max_d     , 6)
    m =     round( 0.5 * max_m      , 6)
    d =     round( 0.5 * max_d      , 6)
    min_p = round( -3 * max_pp          , 6)
    max_p = round( 3 * max_pp           , 6)
    if np.mod( i + 1, 5) == 1:
        print('[', str(int(casedata['gen'][i,0])).ljust(10), ',', str(3).ljust(10), ',', str(m    ).ljust(10)  ,',', str(m    ).ljust(10) ,',', str(d     ).ljust(10),',', str(d    ).ljust(10) ,',', str(min_p).ljust(10), ',', str(max_p).ljust(10), '],' )
    elif np.mod( i + 1, 5) == 2:
        print('[', str(int(casedata['gen'][i,0])).ljust(10), ',', str(3).ljust(10), ',', str(min_m).ljust(10)  ,',', str(min_m).ljust(10) ,',', str(min_d ).ljust(10),',', str(max_d).ljust(10) ,',', str(min_p).ljust(10), ',', str(max_p).ljust(10), '],' )
    elif np.mod( i + 1, 5) == 3:
        print('[', str(int(casedata['gen'][i,0])).ljust(10), ',', str(3).ljust(10), ',', str(m    ).ljust(10)  ,',', str(m    ).ljust(10) ,',', str(min_d ).ljust(10),',', str(max_d).ljust(10) ,',', str(min_p).ljust(10), ',', str(max_p).ljust(10), '],' )
    elif np.mod( i + 1, 5) == 4:
        print('[', str(int(casedata['gen'][i,0])).ljust(10), ',', str(3).ljust(10), ',', str(min_m).ljust(10)  ,',', str(max_m).ljust(10) ,',', str(d     ).ljust(10),',', str(d    ).ljust(10) ,',', str(min_p).ljust(10), ',', str(max_p).ljust(10), '],' )
    elif np.mod( i + 1, 5) == 0:
        print('[', str(int(casedata['gen'][i,0])).ljust(10), ',', str(3).ljust(10), ',', str(min_m).ljust(10)  ,',', str(max_m).ljust(10) ,',', str(min_d ).ljust(10),',', str(max_d).ljust(10) ,',', str(min_p).ljust(10), ',', str(max_p).ljust(10), '],' )
    
    # if True:
    #     print('[', str(int(casedata['gen'][i,0])).ljust(10), ',', str(3).ljust(10), ',', str(min_m).ljust(10)  ,',', str(max_m).ljust(10) ,',', str(min_d ).ljust(10),',', str(max_d).ljust(10) ,',', str(min_p).ljust(10), ',', str(max_p).ljust(10), '],' )



'''
#code for change the number
new_bus_no = dict(zip(casedata['bus'][:,0], range(1, len(casedata['bus'][:,0]) + 1)  ))

for i in range(len(casedata['bus'][:,0])):
    casedata['bus'][i,0] = new_bus_no[casedata['bus'][i,0]]

for i in range(len(casedata['gen'][:,0])):
    casedata['gen'][i,0] = new_bus_no[casedata['gen'][i,0]]

for i in range(len(casedata['branch'][:,0])):
    casedata['branch'][i,0] = new_bus_no[casedata['branch'][i,0]]
    casedata['branch'][i,1] = new_bus_no[casedata['branch'][i,1]]

for i in range(len(casedata['bus'][:,0])):
    print(int(casedata['bus'][i,0]),'\t,')


for i in range(len(casedata['gen'][:,0])):
    print(int(casedata['gen'][i,0]),'\t,')

for i in range(len(casedata['branch'][:,0])):
    print(int(casedata['branch'][i,0]),'\t,', int(casedata['branch'][i,1]),'\t,')
'''

'''

# code for construct the clique tree
ind_branch = list() # index set [(ind_bus_from, ind_bus_to)] of all branch
for i_b in casedata['branch'][:,[0,1]]:
    ind_branch.append( (int(i_b[0]), int(i_b[1])) )


i_gen = casedata['gen'][:,0].astype(int).tolist()
i_non_0 = (np.where(casedata['bus'][:,2]==0)[0]+1).tolist()
i_non = list(set(i_non_0).difference(set(i_gen)))
i_load_0 = list(set(casedata['bus'][:,0].astype(int).tolist()).difference(set(i_gen)))
i_load = list(set(i_load_0).difference(set(i_non)))
i_gl = i_gen + i_load
# construct the graph of the power grid, and node_array gives the node order in A
G = nx.Graph()
G.add_edges_from(ind_branch)
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
    G_clique_nodes_gl[key] = G_clique_nodes_gl[key].intersection(i_gl)

for key in G_clique_nodes.keys():
    G_clique_nodes[key] = list(G_clique_nodes[key])
    G_clique_nodes_gl[key] = list(G_clique_nodes_gl[key])

casedata["clique_tree"] = {
    'node': G_clique_nodes, 'edge': G_clique_edges, 'node_gl': G_clique_nodes_gl,
    }

'''


'''
# approach 2
# tree decomposition. It is not a strict clique tree.
G_clique = approximation.treewidth.treewidth_min_fill_in(G)[1]

G_clique_nodes = dict(zip(range(1, len(G_clique.nodes)+1), list(G_clique.nodes)))
G_clique_edges = list()
for edge_clique in G_clique.edges:
    G_clique_edges.append( [list(G_clique_nodes.keys())[list(G_clique_nodes.values()).index(edge_clique[0])], 
        list(G_clique_nodes.keys())[list(G_clique_nodes.values()).index(edge_clique[1])]] )

G_clique_nodes_gl = deepcopy(G_clique_nodes)
for key in G_clique_nodes_gl.keys():
    G_clique_nodes_gl[key] = G_clique_nodes_gl[key].intersection(i_gl)

for key in G_clique_nodes.keys():
    G_clique_nodes[key] = list(G_clique_nodes[key])
    G_clique_nodes_gl[key] = list(G_clique_nodes_gl[key])

casedata["clique_tree"] = {
    'node': G_clique_nodes, 'edge': G_clique_edges, 'node_gl': G_clique_nodes_gl,
    }
'''