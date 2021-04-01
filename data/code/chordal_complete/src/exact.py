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

"""exact.py
References
[FKTV08Exact] Fomin, Kratsch, Todinca and Villanger.
    Exact Algorithms for Treewidth and Minimum Fill-in.
    SICOMP 2008.
[BBC00Generating] Anne Berry, Jean-Paul Bordat and Olivier Cogis.
    Generating all the minimal separators of a graph.
    International Journal of Foundations of Computer Science. 2000.
[BT02Listing] Vincent Bouchitte and Ioan Todinca.
    Listing all potential maximal cliques of a graph.
    Theoretical Computer Science. 2002.    
"""

import pervasives as pv
import kernel
import bounds

def minimumFillIn(G):
    """Interface function.
    Applies simple kernelization and
    obtains a minimum fill-in of G.
    """
    sol = []
    for H in kernel.kernelize(G):
        boundSol = bounds.upperBoundDiamond(H, 50)
        boundVal = len(boundSol)
        S, B, bids, Sim = computeSepBlocks(H)
        P = allPMC(H, S, boundVal)
        sol.extend(mfi(H, S, B, bids, P, boundSol, Sim))
    return sol

def mfi(G, S, B, bids, P, boundSol, Sim):
    """Obtains a minimum fill-in of G based on the algorithm in [FKTV08Exact].
    mfis[i] is the mfi size of realization of block i.
    """
    infinity = G.number_of_nodes()*G.number_of_nodes()
    BP, PB = computeBPnPB(G, B, bids, P, S)
    mfis = []
    BPopt = []
    for i in xrange(0, len(bids)):
        mfis.append(-1)
        BPopt.append(-1)                                            
    for i in xrange(0, len(bids)):
        if B[bids[i]][0] in Sim:
            fullSet = S[B[bids[i]][0]][0].union(B[bids[i]][1])
            mfis[bids[i]] = pv.fillSize(G, fullSet)
        else:
            mfis[bids[i]] = infinity
    for i in xrange(0, len(bids)):
        b = bids[i]
        (s, C) = B[b]
        Ss2 = S[s][2]
        for p in BP[b]:
            val = 0
            for pb in PB[(p,b)]:
                val += mfis[pb]
            val += P[p][1] - Ss2
            if val < mfis[b]:
                mfis[b] = val
                BPopt[b] = p
    mfival = infinity
    optsep = -1
    for i in Sim:
        val = S[i][2]
        for j in S[i][1]:
            val += mfis[j]
        if val <= mfival:
            mfival = val
            optsep = i
    if mfival >= len(boundSol):
        return boundSol
    return extractMFI(G, P, S, B, PB, BPopt, optsep)

def extractMFI(G, P, S, B, PB, BPopt, optsep):
    """Extract the actual minimum fill-in.
    """
    mfi = pv.cliquifyRet(G, S[optsep][0])
    optBset = set(S[optsep][1])
    while optBset:
        optb = optBset.pop()
        optpmc = BPopt[optb]
        if optpmc != -1:
            mfi.extend(pv.cliquifyRet(G, P[optpmc][0]))
            for b in PB[(optpmc,optb)]:
                optBset.add(b)
    return mfi

def getBlockId(Sd, Cd, S, B, bids):
    """Find the block index of (Sd, Cd) in B.
    Binary search based on bids - the indices of B
    sorted based on size of the blocks.
    """
    size = len(Sd) + len(Cd)
    lenb = len(bids)
    top = lenb-1
    bottom = 0
    while top >= bottom:
        mid = (top+bottom) / 2
        idx = bids[mid]
        Td = S[B[idx][0]][0]
        Dd = B[idx][1]
        lensc = len(Td) + len(Dd)
        if lensc < size:           #then explore the right side.
            bottom = mid + 1
            continue
        if lensc > size:           #then explore the left side
            top = mid - 1
            continue
        save = mid                 #the sizes are equal. need to explore all blocks with same size. linear search.
        while mid <= top:          #explore the right portion (only same size blocks)
            idx = bids[mid]
            if len(S[B[idx][0]][0]) + len(B[idx][1]) != size:
                break
            if S[B[idx][0]][0] == Sd and B[idx][1] == Cd:
                return idx
            mid += 1
        while save > bottom:       #explore the left portion (only same size blocks)
            idx = bids[save-1]
            if len(S[B[idx][0]][0]) + len(B[idx][1]) != size:
                break
            if S[B[idx][0]][0] == Sd and B[idx][1] == Cd:
                return idx
            save -= 1
        break
    return -1                     

def computeBPnPB(G, B, bids, P, S):
    """Computess BP and PB.
    B is the dictionary of blocks where bids is its indices sorted based
    on the block size, 
    P is the list of pmcs and S is the dictionary of minimal
    separators.  
    BP[i] is the dictionary of pmcs associated with the block B[i].
    PB[(p,b)] is the dictionary of blocks which are associated to P[p]
    and which are subsets of B[b].
    """
    BP = {}
    PB = {}
    for i in xrange(0, len(B)):
        BP[i] = []
    for p in xrange(0, len(P)):
        Pp = P[p][0]
        lenPp = len(Pp)
        b = len(B)-1
        pBlocks = list(pv.getBlocks(G, Pp))
        while b >= 0:
            bb = bids[b]
            (s, C) = B[bb]
            Ss = S[s][0]
            lenSs = len(Ss)
            if lenPp <= lenSs:
                b -= 1
                continue
            if Ss.issubset(Pp) and Pp.issubset(Ss.union(C)):
                BP[bb].append(p)
                PB[(p,bb)] = []
                for (Sd, Cd) in pBlocks:
                    SdCd = Sd.union(Cd)
                    if len(SdCd)>= lenSs+len(C):
                        continue
                    if SdCd.issubset(Ss.union(C)):
                        PB[(p,bb)].append(getBlockId(Sd,Cd,S,B,bids))
            b = b-1
    return BP, PB
        
def computeSepBlocks(G):
    """Computes the following:
    1. S <dictionary>: the minimal separators of G.
        S[i] = (s, blocks, fs), where s is the set of 
        minimal separator, blocks are the ids of blocks 
        associated with s, and fs is the fill-in size of s.
    2. B <dictionary>: contains the full blocks associated with minimal separators.
        B[i] = (s, C) denotes the block where s is the index to S and C is the 
        actual component.
    4. bids <list>: block indices (in B) sorted in nondecreasing order based
        on the number of vertices in the blokcs.
    5. Sim <set>: contains the inclusion-wise minimal separators (indices from S).  
    See Algorithm AllMinSep in [BBC00Generating].
    """
    #INITIALIZATION step in the algorithm.
    S = {}                                   #Minimal separators
    B = {}                                   #Blocks
    S_temp = set()
    sid = 0                                  #separator id.
    bid = 0                                  #full block id.
    Sim = set()
    for v in G.nodes():
        N = set([v])
        N.update(G.neighbors(v))
        for (Sd, Cd) in pv.getBlocks(G, N):
            if Sd in S_temp:
                continue
            fs = pv.fillSize(G, Sd)
            B[bid] = (sid, Cd)
            blocks = [bid]
            bid += 1
            for (Td, Dd) in pv.getBlocks(G, Sd.union(Cd)):
                if Td != Sd:                 #we only need full blocks.
                    continue
                B[bid] = (sid, Dd)
                blocks.append(bid)
                bid += 1
            S[sid] = (Sd, blocks, fs)
            S_temp.add(frozenset(Sd))
            updateSim(Sim, S, Sd, sid)
            sid += 1
    #GENERATION step in the algorithm.
    counter = 0
    while counter != len(S):
        (Sd, blocks_dummy, fs_dummy) = S[counter]
        for x in Sd:
            T = set(G.neighbors(x)).union(Sd)
            for (Td, Dd) in pv.getBlocks(G, T):
                if Td in S_temp:
                    continue
                fs = pv.fillSize(G, Td)
                B[bid] = (sid, Dd)
                blocks = [bid]
                bid += 1
                for Ud, Ed in pv.getBlocks(G, Td.union(Dd)):
                    if Ud != Td:
                        continue
                    B[bid] = (sid, Ed)
                    blocks.append(bid)
                    bid += 1
                S[sid] = (Td, blocks, fs)
                updateSim(Sim, S, Td, sid)
                S_temp.add(frozenset(Td))
                sid += 1
        counter += 1
    bids = sorted([bid for bid in B.keys()], key=lambda x: len(S[B[x][0]][0])+len(B[x][1]))
    return S, B, bids, Sim
        
def updateSim(Sim, S, Sd, sid):
    """S is the current dictionary of minimal separators.
    Sd is a candidate for inclusion-wise minimal minimal separator.
    Update Sim accordingly.
    """
    toAdd = True
    RemoveSet = set()
    for sim in Sim:
        if S[sim][0].issubset(Sd):
            toAdd = False
            break
        if Sd.issubset(S[sim][0]):
            RemoveSet.add(sim)
    if toAdd:
        Sim.add(sid)
    Sim.difference_update(RemoveSet)        


def allPMC(G, S, boundVal):
    """Return a set of all potential maximal cliques of G.
    S is the set of all minimal separators of G.
    boundVal is an upper bound on the minimum fill-in of G.
    See Algorithm 'main program' in [BT02Listing].
    """
    n = G.number_of_nodes()
    V = pv.mcs(G)
    P_cur = set()
    P_cur.add((frozenset([V[0]]), 0))
    S_cur = set()
    U = set([V[0]])
    H = G.subgraph(U)
    for i in xrange(1, G.number_of_nodes()):
        u = V[i]
        H.add_edges_from([(u, v) for v in pv.intersection(G.neighbors(u), U)])
        U.add(u)
        S_next = computeAllMSforSG(H, S, U)
        P_next = oneMoreVertex(H, u, S_cur, S_next, P_cur, boundVal)
        P_cur = P_next
        S_cur = S_next
    return list(P_next)

def oneMoreVertex(G, a, S_cur, S_next, P_cur, boundVal):
    """See Algorithm ONE_MORE_VERTEX in [BT02Listing].
    """
    P_next = set()
    P_next_temp = set()
    for (pmc, dummy) in P_cur:
        if pmc in P_next_temp:
            continue
        ispmc, fs = isPMC(pmc, G, boundVal)
        if ispmc:
            P_next.add((pmc, fs))
            P_next_temp.add(pmc)
        else:
            pmca = frozenset(pmc.union(set([a])))
            if pmca in P_next_temp:
                continue
            ispmc, fs = isPMC(pmca, G, boundVal)
            if ispmc:
                P_next.add((pmca, fs))
                P_next_temp.add(pmca)
    for s in S_next:
        sa = frozenset(s.union(set([a])))
        if not sa in P_next_temp:
            ispmc, fs = isPMC(sa, G, boundVal)
            if ispmc:
                P_next.add((sa, fs))
                P_next_temp.add(sa)
        if a not in s and not s in S_cur:
            blocks = list(pv.getBlocks(G, s))
            for t in S_next:
                for (Sd, C) in blocks:
                    if Sd == s:
                        stc = frozenset(s.union(t.intersection(C)))
                        if stc in P_next_temp:
                            continue
                        ispmc, fs = isPMC(stc, G, boundVal)
                        if ispmc:
                            P_next.add((stc, fs))
                            P_next_temp.add(stc)
    return P_next 
    
def computeAllMSforSG(H, S, U):
    """Compute the set of all minimal separators of G[U].
    S is the set of all minimal separators of G. 
    See page 30 of [BT02Listing].
    """
    T = set()
    for i in xrange(0, len(S)):
        s, blocks, fs = S[i]
        t = s.intersection(U)
        if t in T:
            continue
        if isMinSep(t, H):
            T.add(frozenset(t))
    return T
      
def isMinSep(S, G):
    """Check whether S is a minimal separator of G.
    A set of vertices S is a minimal separator of G
    if and only if there are at least two full components
    associated with S in G.
    See Lemma 1 of [BT02Listing].
    """
    n_of_full_comps = 0
    for (Sd, C) in pv.getBlocks(G, S):
        if Sd == S:
            n_of_full_comps += 1
            if n_of_full_comps == 2:
                return True
    return False

def isPMC(P, G, boundVal):
    """Check whether P is a pmc in G.
    If yes, the number of nonedges in G[P]
    is also returned.
    See Theorem 8 of [BT02Listing].
    """
    nj = len(P)
    J = G.subgraph(P)
    fs = nj*(nj-1)/2 - J.number_of_edges()
    if  fs > boundVal:
        return False, -1
    for (N, C) in pv.getBlocks(G, P):
        if N == P:
            return False, -1
        pv.cliquify(J, N)
    if nj*(nj-1) == 2*J.number_of_edges():    #checking whether J became a clique.
        return True, fs
    return (False, -1)
    
