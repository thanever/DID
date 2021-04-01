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

"""pervasives.py
"""

from itertools import combinations

def minus(A,B):
    """difference of two lists
    """
    return filter(lambda x: x not in B, A)

def intersection(A,B):
    """intersection of two lists
    """
    return filter(lambda x: x in B,A)

def isClique(G,S):
    """test if S induces a clique in G 
    """
    GS = G.subgraph(S)
    s = GS.number_of_nodes()
    return GS.number_of_edges() == s*(s-1)/2

def isComplete(G):
    """test if a graph is itself a clique
    """
    return not G or isClique(G,G.nodes())

def cliquify(G,S):
    """Cliquify the vertex set S in G.
    """
    G.add_edges_from(combinations(S,2))
    
def cliquifyRet(G,S):
    """same but cliquify but returns the set of addedEdges
    """
    sav = set(G.edges())
    G.add_edges_from(combinations(S,2))
    return list(set(G.edges()) - sav)

def sort_by_value(D):
    """Sort items in a dictionary/list by value/the second term in the tuple.
    Returns a list.
    The input dictionary/list is not modified.
    """
    if type(D) is dict:
        items = [(v,k) for k,v in D.items()]
    else:
        items = [(v,k) for (k,v) in D]
    items.sort()
    items = [(k,v) for (v,k) in items]
    return items

def merge(list1, list2):
    """Merge two sorted (nondecreasing values) list of tuples,
    as we do in merge sort
    Both lists must be nonempty.
    """
    if not list1:
        return list2
    if not list2:
        return list1
    new_list = []
    size1 = len(list1)
    size2 = len(list2)
    j = 0
    k = 0
    key1, val1 = list1[j]
    key2, val2 = list2[k]
    while True:
        if val2 < val1:
            new_list.append((key2,val2))
            k = k+1
            if k>=size2:
                new_list.extend(list1[j:size1])
                break
            key2, val2 = list2[k]
        else:
            new_list.append((key1,val1))
            j = j+1
            if j>=size1:
                new_list.extend(list2[k:size2])
                break
            key1, val1 = list1[j]
    return new_list

def fillSize(G, S):
    """find the number of nonedges in G[S]
    """
    return len(S)*(len(S)-1)/2 - G.subgraph(S).number_of_edges()

def getBlocks(G, S):
    """Return blocks associated with S in G;
    i.e., a list of tuples (T,C) where C is a 
    a component in G-S and T is N(C).
    """
    U = minus(G.nodes(), S)
    Q = []
    while U:
        C = set()
        T = set()
        Q.append(U.pop(0))
        while Q:
            u = Q.pop(0)
            C.add(u)
            N = G.neighbors(u)
            for v in N:
                if v in S:
                    T.add(v)
                elif not v in C:
                    C.add(v)
                    Q.append(v)
                    U.remove(v)
        yield (frozenset(T),frozenset(C))

def mcs(G):
    """Maximum cardinality search.
    Returns an ordering of the vertices as described in [TY84Simple].
    The ordering is not reversed to make sure than G[V[0:i]] is connected.
    """
    n = G.number_of_nodes()
    sets = []
    size = {}
    ordering = []
    orderingDict = {}
    
    for v in G.nodes():
        sets.append([])
        size[v] = 0
        sets[0].append(v)
    number = n-1
    j = 0
    while number >= 0:
        v = sets[j].pop(0)
        ordering.append(v)
        orderingDict[v] = number
        size[v] = -1
        for w in G.neighbors(v):
            if size[w] >= 0:
                sets[size[w]].remove(w)
                size[w] = size[w]+1
                sets[size[w]].append(w)
        j += 1
        if j == len(sets):
            j -= 1
        while j>=0 and not sets[j]:
            j -= 1
        number -= 1
#    ordering.reverse()
    return ordering

