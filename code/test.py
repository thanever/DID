#%%
import plotly.offline as py
import plotly.graph_objs as go

import networkx as nx
import numpy as np
from pypower.api import runpf, loadcase, ppoption
from os.path import dirname, join
from copy import deepcopy

casename = 'case118'
path_casedata = join(dirname(__file__), 'data//casedata//'+ casename +'.py')
casedata = loadcase(path_casedata)

'''
opt = ppoption(PF_TOL=1e-13, PF_MAX_IT= 10)
s_pf = runpf(casedata, opt)
'''

G = nx.Graph()
G.add_nodes_from(casedata['bus'][:,0].astype(int))
G.add_edges_from(casedata['branch'][:,0:2].astype(int))

pos = casedata['position']
nx.set_node_attributes(G, pos, 'pos')

node_gen = casedata['gen'][:,0].astype(int)

G_copy = deepcopy(G)
edge_removed = [(15, 33), (19, 34), (30, 38), (24, 23), (30, 26), (113, 32), (17, 31), (19, 20), (38, 65), (69, 47),(69, 49), (66, 49), (64, 65), (62, 61), (62, 60), (77, 82), (96, 97), (80, 96), (98, 100), (99, 100)]
for edge in edge_removed: G_copy.remove_edge(edge[0], edge[1])
area_name = ['Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5']
area_node = dict(zip(area_name, list(nx.connected_components(G_copy))))  
area_color = dict(zip(area_name, ['#55efc4', '#fab1a0', '#ffeaa7', '#a29bfe', '#74b9ff']))



edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker= dict(
        showscale=False,
        color=[],
        symbol = [],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=2)))

for node in G.nodes():
    x, y = G.node[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node in G.node():

    if node in node_gen:
        node_trace['marker']['symbol']+= tuple( ['square'] )
    else:
        node_trace['marker']['symbol']+= tuple( ['circle'] )
    
    
    for area in area_name:
        if node in area_node[area]:
            node_info = area + ' | Bus-' + str(node)
            node_trace['text']+=tuple([node_info])
            node_trace['marker']['color']+= tuple( [area_color[area]] )
            break

fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

py.plot(fig, filename='networkx.html')
