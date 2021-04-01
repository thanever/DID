# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 9 bus, 3 generator case.
"""

from numpy import array

def case9_1():
    """Power flow data for 9 bus, 3 generator case.
    Please see L{caseformat} for details on the case file format.

    Based on data from Joe H. Chow's book, p. 70.

    @return: Power flow data for 9 bus, 3 generator case.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin; origin inertia of gen 1 and gen 3: 7.1470, 3.1619
    ppc["bus"] = array([
        [1, 3, 2.3 * 0,   2.3 *  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0.01 ,   0.01 ],
        [2, 2, 2.3 * 0,   2.3 *  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0.01 ,   0.01 ],
        [3, 2, 2.3 * 0,   2.3 *  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0.01 ,   0.01 ],
        [4, 1, 2.3 * 0,   2.3 *  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0 ,      0 ],
        [5, 1, 2.3 * 90,  2.3 * 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0 ,      0.01 ],
        [6, 1, 2.3 * 0,   2.3 *  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0 ,      0 ],
        [7, 1, 2.3 * 100, 2.3 * 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0 ,      0.01 ],
        [8, 1, 2.3 * 0,   2.3 *  0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0 ,      0 ],
        [9, 1, 2.3 * 125, 2.3 * 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9, 0 ,      0.01 ]
    ])


# 5.9152, 2
# 3.1619, 2

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 2.3 * 0,   0, 300, -300, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2.3 * 163, 0, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 2.3 * 85,  0, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 4, 0,      0.0576, 0,     250, 250, 250, 0, 0, 1, -135, 135],
        [4, 5, 0,      0.092,  0.158, 250, 250, 250, 0, 0, 1, -135, 135],
        [5, 6, 0,      0.17,   0.358, 150, 150, 150, 0, 0, 1, -135, 135],
        [3, 6, 0,      0.0586, 0,     300, 300, 300, 0, 0, 1, -135, 135],
        [6, 7, 0,      0.1008, 0.209, 150, 150, 150, 0, 0, 1, -135, 135],
        [7, 8, 0,      0.072,  0.149, 250, 250, 250, 0, 0, 1, -135, 135],
        [8, 2, 0,      0.0625, 0,     250, 250, 250, 0, 0, 1, -135, 135],
        [8, 9, 0,      0.161,  0.306, 250, 250, 250, 0, 0, 1, -135, 135],
        [9, 4, 0,      0.085,  0.176, 250, 250, 250, 0, 0, 1, -135, 135]
    ])

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 5]
    ])

    ##----- Generator bus control data  -----##
    # type of generator or inverter, min_m, max_m, min_d, max_d, min_p (p.u.), max_p (p.u.)
	#  type of generator or inverter: 9 - conventional generator, 0 - inverter with no optimize, 1 - inverter optimizing d, 2 - inverter optimizing m, 3 - inverters optimizing d and m
    ppc["gencontrol"] = array([
        [ 1          , 3          , 0.000796   , 0.079577   , 0.007958   , 0.795775   , -7.5       , 7.5        ],
        [ 2          , 3          , 0.000955   , 0.095493   , 0.009549   , 0.95493    , -9.0       , 9.0        ],
        [ 3          , 3          , 0.000859   , 0.085944   , 0.008594   , 0.859437   , -8.1       , 8.1        ]
    ])
    
    
    ###----- disturbances data -----###
    #1 - power-step: num, buses, start time, amplitude(\times p_0)
    #2 - power ramp: num, buses, start time, duration time, height (\times p_0)
    #3 - power fluctuation: num, buses, start time, time interval, date num
    #4 - 3psc: num, branch-f_bus,  branch-t_bus, nearest bus, short-circuit conductance (p.u.), start time, clearing time(\Delta t with second)
    #9: uniform distribution date on [-0.2, 0.2]

    ppc["disturbance"] = {
    1: array([
        [1,     3,      0,      -0.5, 0.15],
        [2,     9,      0,      -0.5, 0.15]
    ]),
    2: array([
        [1,    1,      0,      5,      -0.5, 0.15],
        [2,    7,      0,      5,      -0.5, 0.15]
    ]),
    3: array([
        [1,    2,      0,      0.5,      9, 0.6],
        [2,    5,      0,      0.5,      9, 0.6]
    ]),
    4: array([
        [1,    5,      6,       999,      999,    0,      0.1, 0.1],
        [2,    8,      9,       999,      999,    0,      0.1, 0.1]
    ]),
    9: array([
        -0.077752022,-0.049592684,-0.094386402,-0.027777255,0.106175442,0.07760974,0.010639525,0.145379902,-0.113145199,-0.173299214,0.027856593,-0.153232617,-0.123096728,-0.077041254,-0.152101278,-0.059589093,-0.06009897,-0.121105119,-0.017018885,0.072363903,0.170954442,-0.034889518,-0.060918945,0.037836957,0.170201299,-0.1546673,0.1972234,-0.071938968,-0.107488659,-0.088976472,-0.068018895,-0.06587186,-0.003419552,0.152372651,0.112508103,0.142640044,-0.049963844,-0.149540361,0.139890352,0.13228229,-0.075472623,0.055259011,-0.017222513,-0.13241881,0.012423058,-0.000117756,0.191749308,-0.080003032,-0.166717849,-0.149633579,-0.166748818,-0.179088528,0.199144351,-0.101607438,-0.105415065,0.149580196,-0.005620803,0.089868514,0.022076414,-0.094829629,0.111238813,0.133720447,0.121662029,0.141913883,-0.111726178,-0.14110605,0.032269891,0.120886645,0.009050625,-0.011489962,0.016302734,0.12436742,0.11313339,-0.092537767,0.147695992,0.174444751,0.15505781,-0.193334631,0.159925609,-0.099949322,0.099644356,0.079428698,-0.048152161,0.06028194,0.126261189,0.15150098,-0.147034132,-0.058805963,-0.094758118,0.051548502,0.020517378,-0.195080454,0.092211274,0.089238641,-0.091390137,-0.088318354,0.030346984,0.13823391,-0.160389818,-0.160731705
    ])
    }

    # parameters of discretization
    # time_ele_d1 - number of time elements for disturbance 1;
    # time_ele_d2_d - number of time element during the disturbance for disturbance 2;
    # t_f - assume t_0 = 0 for all disturbances
    # order - order of collocation
    ppc["param_disc"] = {
    'time_ele':{1: 20, 2: 20, 3: 20, 4: 20},
    'time_ele_d':{2: 4, 4: 4},
    'order':{1: 3, 2: 3, 3: 3, 4: 3},
    't_f':{1: 30, 2: 30, 3: 30, 4: 30},
    'colloc_point_radau': {1:(0, 1.0),2:(0, 0.333333, 1.0),3:(0, 0.155051, 0.644949, 1.0),4:(0, 0.088588, 0.409467, 0.787659, 1.0),5:(0, 0.057104, 0.276843, 0.583590, 0.860240, 1.0)}
    }

    ppc["freq_band"] = {
    1:{(0, 15):(49.5, 50.5), (15, 300):(49.85, 50.15)},
    2:{(0, 15):(49.5, 50.5), (15, 300):(49.85, 50.15)},
    3:{(0, 300):(49.85, 50.15)},
    4:{(0, 15):(49, 51), (15, 60):(49.5, 50.5), (60, 300):(49.85, 50.15)}
    }


    return ppc
