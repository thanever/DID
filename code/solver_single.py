from pyomo.opt import SolverFactory
  
def solve_single(opt_x_s):
    '''
    solve one opt_x[s]
    '''
    solver = SolverFactory('ipopt')
    solver.set_options('constr_viol_tol=1e-10')
    solver.solve(opt_x_s, tee=False)

    return opt_x_s