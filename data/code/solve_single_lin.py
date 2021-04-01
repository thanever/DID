def solve_single_lin(opt_x_s):
    '''
    solve one opt_x[s]
    '''

    opt_x_s.solve(solver='MOSEK', verbose = True)

    return opt_x_s