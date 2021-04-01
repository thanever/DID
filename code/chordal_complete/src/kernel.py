#!/usr/bin/env python
# -*- coding: utf-8; -*-
#
# Copyright 2017  / Édouard Bonnet, R. B. Sandeep, and Florian Sikora
#
# This file is part of MFI a submission to PACE 2017 Track B: Minimum Fill-in.
#
# MFI is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# MFI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MFI.  If not, see <http://www.gnu.org/licenses/>.

"""kernel.py. Simple kernelization.
"""

import networkx as nx
import pervasives as pv

def kernelize(G):
    """
    """
    for H in nx.biconnected_component_subgraphs(G):
        flag = removeSimplicialUniversal(H)
        if H:
            if flag:
                for J in nx.biconnected_component_subgraphs(H):
                    yield J
            else:
                yield H

def removeSimplicialUniversal(G):
    """Remove all simplicial and universal 
    vertices from G. Note that universal 
    vertices do not affect 'simpliciality' of a 
    vertex. But removing a simplicial vertex
    may introduce new simplicial and universal vertices. 
    """
    n = G.number_of_nodes()
    removeUniversal(G)
    while (removeSimplicial(G)):
        removeUniversal(G)
    return not n == G.number_of_nodes()

def removeSimplicial(G):
    """Remove simplicial vertices.
    """
    l = G.number_of_nodes()
    G.remove_nodes_from([v for v in G.nodes() if simplicial(G,v)])  
    return not l == G.number_of_nodes()
    
def removeUniversal(G):
    """Remove universal vertices.
    """
    n = G.number_of_nodes()
    degrees = G.degree()
    G.remove_nodes_from([k for k, d in degrees if d == n-1])  
    return not n == G.number_of_nodes()

def simplicial(G,v):
    """test if a vertex v is simplicial in G
    """
    neighbors = G.neighbors(v)
    for u in neighbors:
        neighbors = [nnn for nnn in neighbors][1:]
        for w in neighbors:
            if not u in G[w]:
                return False
    return True

