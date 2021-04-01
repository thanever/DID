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

"""main.py
Finds a chordal completion of the input graph.
Input from standard input and output to standard output.
"""

import parser
import exact
    

G = parser.parseStdInput()
sol = exact.minimumFillIn(G)
print(sol)
parser.output(sol)


