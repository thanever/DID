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

"""bounds.py
The idea of upperBoundDiamond is to add first edges whose absence would
force many edges (by the diamond configuration), where 'many' is quantified
by threshold. When no such edge exists anymore, we run mdh.upperBound on the
remaining graph.
Why is upperBoundDiamond better in some cases? Think of it this way. When,
there are diamonds with really high value then the edge we add is likely to
to be in optimum solutions anyway.
"""

import mdh
import kernel

def upperBoundDiamond(H,threshold=30):
    """
    """
    G = H.copy()
    solution = []
    flag = True
    diamondList = exhaustiveDiamondsList(G,threshold)
    while flag:
        flag = False
        for (diamondValue,(u,v,commonNeighbors)) in diamondList:
            #update the value of the diamond based on the added edges
            updatedValue = diamondValue
            for (a,b) in solution:
                if a in commonNeighbors and b in commonNeighbors:
                    updatedValue -= 1
            #if this value is still above the threshold add (u,v)
            if updatedValue >= threshold:
                G.add_edge(u,v)
                solution = [(u,v)] + solution
                flag = True
        #check if some vertices have become simplicial
        kernel.removeSimplicialUniversal(G)
        #compute the new diamondList
        diamondList = exhaustiveDiamondsList(G,threshold)
    return solution+mdh.upperBound(G)

def exhaustiveDiamondsList(G,threshold=5):
    """A diamond is a set of induced C_4's all containing the same non linked pair u and v
    the value of a diamond is the number of such C_4's.
    This method gives exhaustive list of the diamonds of value more than threshold
    """
    vertices = G.nodes()
    listDiamonds = []
    for u in vertices:
        N = G[u]  #neighbors of u
        l = len(N)
        nonedgeNumber = l*(l-1)/2 - G.subgraph(N).number_of_edges()
        if threshold <= nonedgeNumber: #if the number of nonedges in the neighborhood of u is smaller
                                       #than the threshold, no need to go further
            NN = {}
            for v in N:
                for w in G[v]:
                    if w not in N and w not in NN and u < w:
                        commonNeighbors = [z for z in N if z in G[w]]
                        lc = len(commonNeighbors)
                        NN[w] = lc*(lc-1)/2 - G.subgraph(commonNeighbors).number_of_edges()
                        if threshold <= NN[w]:
                            listDiamonds += [(NN[w], (u,w,commonNeighbors))]
    return sorted(listDiamonds)

