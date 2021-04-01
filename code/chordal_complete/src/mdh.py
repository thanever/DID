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

"""mdh.py - minimum degree heurtistic and other heuristics.
"""

import pervasives as pv

def upperBound(G):
    """Return the best solution of 
    min deg heuristic and good neighborhood heuristic and both with tie breaking.
    """
    sorted_degrees = pv.sort_by_value(G.degree())  #(vertex, degree) tuples
    mdh_soln = minDegreeHeuristic(G, sorted_degrees)
    sorted_nen_counts = get_sorted_nen_counts(G)
    gnh_soln = goodNeighborhoodHeuristic(G, sorted_nen_counts)
    mdh_tb_soln = minDegreeHeuristicTB(G, sorted_degrees, sorted_nen_counts)
    gnh_tb_soln = goodNeighborhoodHeuristicTB(G, sorted_degrees, sorted_nen_counts)
    best_mdh = mdh_soln
    best_gnh = gnh_soln
    if len(mdh_tb_soln) < len(mdh_soln):
        best_mdh = mdh_tb_soln
    if len(gnh_tb_soln) < len(gnh_soln):
        best_gnh = gnh_tb_soln
    if len(best_gnh) < len(best_mdh):
        return best_gnh
    return best_mdh


def get_sorted_nen_counts(G):
    """Obtain the list of number of 
    non-edges in the neighborhood of vertices,
    sorted in non-decreasing order.
    """
    nen_counts = []                         #counts of nonedges in the neighborhoods
    for v in G.nodes():
        N = G.neighbors(v)
        n = len(N)
        m = G.subgraph(N).number_of_edges()
        nen_count = n*(n-1)/2 - m
        nen_counts.append((v, nen_count)) 
    return pv.sort_by_value(nen_counts)

def minDegreeHeuristic(G, sorted_degrees):
    """INPUT: Graph G
    OUTPUT: Return a set of edges (adding which makes G chordal) 
    obtained by minimum degree heuristic.
    """
    addedEdges = []
    H = G.copy()
    sdc = sorted_degrees[:]
    while H:
        (v, deg) = sdc.pop(0)
        new_edges, sdc = update_for_mdh(H, v, sdc)
        addedEdges.extend(new_edges)
    return addedEdges

def update_for_mdh(G, v, sorted_degrees):
    """Make the neighborhood of v a clique in G,
    remove v from G and update sorted_degrees
    accordingly.
    sorted_degrees should not contain the key v.
    """
    N = G.neighbors(v)
    new_edges = pv.cliquifyRet(G,N)
    G.remove_node(v)
    sorted_degrees = update_sorted_degrees(sorted_degrees, N, new_edges)
    return new_edges, sorted_degrees

def minDegreeHeuristicTB(G, sorted_degrees, sorted_nen_counts):
    """Return a set of edges (adding which makes G chordal) 
    obtained by minimum degree heuristic. 
    With tie breaking.
    """
    addedEdges = []
    H = G.copy()
    sdc = sorted_degrees[:]
    snc = sorted_nen_counts[:]
    while H:
        v,sdc,snc = get_best_v_for_mdh_tb(sdc, snc)
        new_edges, sdc, snc = update_for_tb(H, v, sdc, snc)
        addedEdges.extend(new_edges)
    return addedEdges

def get_best_v_for_mdh_tb(sorted_degrees, sorted_nen_counts):
    """
    """
    sub_list = [sorted_degrees[0][0]]
    min_deg = sorted_degrees[0][1]
    for (v, deg) in sorted_degrees[1:]:
        if deg != min_deg:
            break
        sub_list.append(v)
    for v, nen_count in sorted_nen_counts:
        if v in sub_list:
            sorted_degrees.remove((v,min_deg))
            sorted_nen_counts.remove((v,nen_count))
            return v, sorted_degrees, sorted_nen_counts

def update_for_tb(G, v, sorted_degrees, sorted_nen_counts):
    """
    """
    N = G.neighbors(v)
    new_edges = pv.cliquifyRet(G,N)
    G.remove_node(v)
    sorted_degrees = update_sorted_degrees(sorted_degrees, N, new_edges)
    sorted_nen_counts = update_sorted_nen_counts(G, sorted_nen_counts, N, new_edges)
    return new_edges, sorted_degrees, sorted_nen_counts

def goodNeighborhoodHeuristic(G, sorted_nen_counts):
    """INPUT: Graph G, sorted list of counts of nonedges in the neighborhoods.
    OUTPUT: Return a set of edges (adding which makes G chordal) 
    obtained by good neighborhood heuristic: select a vertex with
    minimum number of nonedges in the neighborhood, makes it neighborhood
    a clique, remove it, repeat. 
    """
    addedEdges = []
    H = G.copy()
    snc = sorted_nen_counts[:]
    while H:
        (v, nc) = snc.pop(0)
        new_edges, snc = update_for_gnh(H, v, snc)
        addedEdges.extend(new_edges)
    return addedEdges

def goodNeighborhoodHeuristicTB(G, sorted_degrees, sorted_nen_counts):
    """INPUT: Graph G
    OUTPUT: Return a set of edges (adding which makes G chordal) 
    obtained by good neighborhood heuristic: select a vertex with
    minimum number of nonedges in the neighborhood, makes it neighborhood
    a clique, remove it, repeat. With tie breaking.
    """
    addedEdges = []
    H = G.copy()
    sdc = sorted_degrees[:]
    snc = sorted_nen_counts[:]
    while H:
        v,sdc,snc = get_best_v_for_gnh_tb(sdc, snc)
        new_edges, sdc, snc = update_for_tb(H, v, sdc, snc)
        addedEdges.extend(new_edges)
    return addedEdges

def get_best_v_for_gnh_tb(sorted_degrees, sorted_nen_counts):
    """
    """
    sub_list = [sorted_nen_counts[0][0]]
    min_count = sorted_nen_counts[0][1]
    for (v, count) in sorted_nen_counts[1:]:
        if count != min_count:
            break
        sub_list.append(v)
    for v, deg in sorted_degrees:
        if v in sub_list:
            sorted_degrees.remove((v,deg))
            sorted_nen_counts.remove((v,min_count))
            return v, sorted_degrees, sorted_nen_counts

def update_sorted_degrees(sorted_degrees, N, new_edges):
    """N is the list of vertices (which is a neighborhood of a vertex v)
    which was made a clique;
    new_edges are the new edges among them.
    """
    partial_sd = {}
    temp_list = []
    for (k,v) in sorted_degrees:
        if k in N:
            partial_sd[k] = v-1
            temp_list.append((k,v))
    sorted_degrees = pv.minus(sorted_degrees, temp_list)
    for (u,w) in new_edges:
        partial_sd[u] = partial_sd[u]+1
        partial_sd[w] = partial_sd[w]+1
    return pv.merge(sorted_degrees, pv.sort_by_value(partial_sd))
    
def update_for_gnh(G, v, sorted_nen_counts):
    """Make the neighborhood of v a clique in G,
    remove v from G and update sorted_nen_counts 
    accordingly.
    sorted_nen_counts should not contain the key v.
    """
    N = G.neighbors(v)
    new_edges = pv.cliquifyRet(G,N)
    G.remove_node(v)
    sorted_nen_counts = update_sorted_nen_counts(G, sorted_nen_counts, N, new_edges)
    return new_edges, sorted_nen_counts

def update_sorted_nen_counts(G, sorted_nen_counts, N, new_edges):
    """N is the list of vertices (which is a neighborhood of a vertex v)
    which was made a clique;
    new_edges are the new edges among them.
    G does not contain v.
    Update sorted_nen_counts accordingly.
    """
    partial_nen_counts = {}
    temp_list = []
    set_to_update = set(N)
    for (u,w) in new_edges:
        set_to_update.update(pv.intersection(G.neighbors(u), G.neighbors(w)))
    for (k,val) in sorted_nen_counts:
        if k in set_to_update:
            partial_nen_counts[k] = pv.fillSize(G, G.neighbors(k))
            temp_list.append((k,val)) 
    sorted_nen_counts = pv.minus(sorted_nen_counts, temp_list)
    return pv.merge(sorted_nen_counts, pv.sort_by_value(partial_nen_counts))

