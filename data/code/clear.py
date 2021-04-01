# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 21:06:29 2018

@author: Tong
"""

from os.path import dirname, join
import os
#%%

if __name__ == "__main__":
	path_current = dirname(__file__)

	for i,j,k in os.walk(path_current, topdown=False):
		print(i,j,k)

	all_files = k

	for p in all_files:
		if ('.py' in p) or ('.mo' in p) or ('.exe' in p) or ('.dll' in p) or ('.md' in p):
			pass
		else:
			os.remove(join(path_current, p))
		if '.pyomo' in p:
			os.remove(join(path_current, p))
		if ('.exe' in p) and ('ipopt.exe' not in p):
			os.remove(join(path_current, p))



#%%
