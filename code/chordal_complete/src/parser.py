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

"""parser.py: I/O functions
"""
import networkx as nx


def parseStdInput():
    """Obtain a graph by parsing the standard input 
    as per the format specified in the PACE Challange.
    """
    edges = [(1,2),(2,3),(3,4),(4,1)]

    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def output(solution):
    """Print solution in the given format:
    each edge of the solution in a line and
    vertices of an edge separated by a white space.
    """
    for (x,y) in solution:
        print(x,y)
