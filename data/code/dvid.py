# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:41:27 2018

@author: eee

"""
from make_parameter import Param
from optimization_sin import OptSin
from optimization_lin import OptLin
from optimization_qcq import OptQcq

import numpy as np
from pyomo.opt import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.environ import *
import pickle
from copy import deepcopy, copy
from collections import Counter
import multiprocessing  as mp
import time
from solver_single import solve_single
import cvxpy as cp
from itertools import combinations

class Dvid(Param):

    def __init__(self, path_casedata, set_disturb):
        self.path_casedata = path_casedata
        self.set_disturb = set_disturb

    def dispatch(self, mode):

        self.mode = mode

        if   self.mode >= 10 and self.mode < 20:  self.opt = OptSin(self.path_casedata, self.set_disturb,  self.mode)
        elif self.mode >= 20 and self.mode < 30:  self.opt = OptLin(self.path_casedata, self.set_disturb,  self.mode)
        elif self.mode >= 30 and self.mode < 40:  self.opt = OptQcq(self.path_casedata, self.set_disturb,  self.mode)
        else: print("Mode error!")
        
    def save(self, path_result):

        self.path_result = path_result
        with open(self.path_result, 'wb') as fpr:
            pickle.dump(self.opt, fpr)
        print('|==== Successfully save results ====|')

