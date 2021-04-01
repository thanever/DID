# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 39 bus New England system.
"""

from numpy import array

def case39():
    """Power flow data for 39 bus New England system.
    Please see L{caseformat} for details on the case file format.

    Data taken from [1] with the following modifications/additions:

        - renumbered gen buses consecutively (as in [2] and [4])
        - added C{Pmin = 0} for all gens
        - added C{Qmin}, C{Qmax} for gens at 31 & 39 (copied from gen at 35)
        - added C{Vg} based on C{V} in bus data (missing for bus 39)
        - added C{Vg, Pg, Pd, Qd} at bus 39 from [2] (same in [4])
        - added C{Pmax} at bus 39: C{Pmax = Pg + 100}
        - added line flow limits and area data from [4]
        - added voltage limits, C{Vmax = 1.06, Vmin = 0.94}
        - added identical quadratic generator costs
        - increased C{Pmax} for gen at bus 34 from 308 to 508
          (assumed typo in [1], makes initial solved case feasible)
        - re-solved power flow

    Notes:
        - Bus 39, its generator and 2 connecting lines were added
          (by authors of [1]) to represent the interconnection with
          the rest of the eastern interconnect, and did not include
          C{Vg, Pg, Qg, Pd, Qd, Pmin, Pmax, Qmin} or C{Qmax}.
        - As the swing bus, bus 31 did not include and Q limits.
        - The voltages, etc in [1] appear to be quite close to the
          power flow solution of the case before adding bus 39 with
          it's generator and connecting branches, though the solution
          is not exact.
        - Explicit voltage setpoints for gen buses are not given, so
          they are taken from the bus data, however this results in two
          binding Q limits at buses 34 & 37, so the corresponding
          voltages have probably deviated from their original setpoints.
        - The generator locations and types are as follows:
            - 1   30      hydro
            - 2   31      nuke01
            - 3   32      nuke02
            - 4   33      fossil02
            - 5   34      fossil01
            - 6   35      nuke03
            - 7   36      fossil04
            - 8   37      nuke04
            - 9   38      nuke05
            - 10  39      interconnection to rest of US/Canada

    This is a solved power flow case, but it includes the following
    violations:
        - C{Pmax} violated at bus 31: C{Pg = 677.87, Pmax = 646}
        - C{Qmin} violated at bus 37: C{Qg = -1.37,  Qmin = 0}

    References:

    [1] G. W. Bills, et.al., I{"On-Line Stability Analysis Study"}
    RP90-1 Report for the Edison Electric Institute, October 12, 1970,
    pp. 1-20 - 1-35.
    prepared by
      - E. M. Gulachenski - New England Electric System
      - J. M. Undrill     - General Electric Co.
    "...generally representative of the New England 345 KV system, but is
    not an exact or complete model of any past, present or projected
    configuration of the actual New England 345 KV system."

    [2] M. A. Pai, I{Energy Function Analysis for Power System Stability},
    Kluwer Academic Publishers, Boston, 1989.
    (references [3] as source of data)

    [3] Athay, T.; Podmore, R.; Virmani, S., I{"A Practical Method for the
    Direct Analysis of Transient Stability,"} IEEE Transactions on Power
    Apparatus and Systems , vol.PAS-98, no.2, pp.573-584, March 1979.
    U{http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4113518&isnumber=4113486}
    (references [1] as source of data)

    [4] Data included with TC Calculator at
    U{http://www.pserc.cornell.edu/tcc/} for 39-bus system.

    @return: Power flow data for 39 bus New England system.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 1, 97.6, 44.2, 0, 0, 2, 1.0393836, -13.536602, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [2, 1, 0, 0, 0, 0, 2, 1.0484941, -9.7852666, 345, 1, 1.06, 0.94           ,0 ,      0 ],
        [3, 1, 322, 2.4, 0, 0, 2, 1.0307077, -12.276384, 345, 1, 1.06, 0.94       ,0 ,      0.01 ],
        [4, 1, 500, 184, 0, 0, 1, 1.00446, -12.626734, 345, 1, 1.06, 0.94         ,0 ,      0.01 ],
        [5, 1, 0, 0, 0, 0, 1, 1.0060063, -11.192339, 345, 1, 1.06, 0.94           ,0 ,      0 ],
        [6, 1, 0, 0, 0, 0, 1, 1.0082256, -10.40833, 345, 1, 1.06, 0.94            ,0 ,      0 ],
        [7, 1, 233.8, 84, 0, 0, 1, 0.99839728, -12.755626, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [8, 1, 522, 176.6, 0, 0, 1, 0.99787232, -13.335844, 345, 1, 1.06, 0.94    ,0 ,      0.01 ],
        [9, 1, 6.5, -66.6, 0, 0, 1, 1.038332, -14.178442, 345, 1, 1.06, 0.94      ,0 ,      0.01 ],
        [10, 1, 0, 0, 0, 0, 1, 1.0178431, -8.170875, 345, 1, 1.06, 0.94           ,0 ,      0 ],
        [11, 1, 0, 0, 0, 0, 1, 1.0133858, -8.9369663, 345, 1, 1.06, 0.94          ,0 ,      0 ],
        [12, 1, 8.53, 88, 0, 0, 1, 1.000815, -8.9988236, 345, 1, 1.06, 0.94       ,0 ,      0.01 ],
        [13, 1, 0, 0, 0, 0, 1, 1.014923, -8.9299272, 345, 1, 1.06, 0.94           ,0 ,      0 ],
        [14, 1, 0, 0, 0, 0, 1, 1.012319, -10.715295, 345, 1, 1.06, 0.94           ,0 ,      0 ],
        [15, 1, 320, 153, 0, 0, 3, 1.0161854, -11.345399, 345, 1, 1.06, 0.94      ,0 ,      0.01 ],
        [16, 1, 329, 32.3, 0, 0, 3, 1.0325203, -10.033348, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [17, 1, 0, 0, 0, 0, 2, 1.0342365, -11.116436, 345, 1, 1.06, 0.94          ,0 ,      0 ],
        [18, 1, 158, 30, 0, 0, 2, 1.0315726, -11.986168, 345, 1, 1.06, 0.94       ,0 ,      0.01 ],
        [19, 1, 0, 0, 0, 0, 3, 1.0501068, -5.4100729, 345, 1, 1.06, 0.94          ,0 ,      0 ],
        [20, 1, 680, 103, 0, 0, 3, 0.99101054, -6.8211783, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [21, 1, 274, 115, 0, 0, 3, 1.0323192, -7.6287461, 345, 1, 1.06, 0.94      ,0 ,      0.01 ],
        [22, 1, 0, 0, 0, 0, 3, 1.0501427, -3.1831199, 345, 1, 1.06, 0.94          ,0 ,      0 ],
        [23, 1, 247.5, 84.6, 0, 0, 3, 1.0451451, -3.3812763, 345, 1, 1.06, 0.94   ,0 ,      0.01 ],
        [24, 1, 308.6, -92.2, 0, 0, 3, 1.038001, -9.9137585, 345, 1, 1.06, 0.94   ,0 ,      0.01 ],
        [25, 1, 224, 47.2, 0, 0, 2, 1.0576827, -8.3692354, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [26, 1, 139, 17, 0, 0, 2, 1.0525613, -9.4387696, 345, 1, 1.06, 0.94       ,0 ,      0.01 ],
        [27, 1, 281, 75.5, 0, 0, 2, 1.0383449, -11.362152, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [28, 1, 206, 27.6, 0, 0, 3, 1.0503737, -5.9283592, 345, 1, 1.06, 0.94     ,0 ,      0.01 ],
        [29, 1, 283.5, 26.9, 0, 0, 3, 1.0501149, -3.1698741, 345, 1, 1.06, 0.94   ,0 ,      0.01 ],
        [30, 2, 0, 0, 0, 0, 2, 1.0499, -7.3704746, 345, 1, 1.06, 0.94             ,0.01 ,   0.01 ],
        [31, 3, 9.2, 4.6, 0, 0, 1, 0.982, 0, 345, 1, 1.06, 0.94                   ,0.01 ,   0.01 ],
        [32, 2, 0, 0, 0, 0, 1, 0.9841, -0.1884374, 345, 1, 1.06, 0.94             ,0.01 ,   0.01 ],
        [33, 2, 0, 0, 0, 0, 3, 0.9972, -0.19317445, 345, 1, 1.06, 0.94            ,0.01 ,   0.01 ],
        [34, 2, 0, 0, 0, 0, 3, 1.0123, -1.631119, 345, 1, 1.06, 0.94              ,0.01 ,   0.01 ],
        [35, 2, 0, 0, 0, 0, 3, 1.0494, 1.7765069, 345, 1, 1.06, 0.94              ,0.01 ,   0.01 ],
        [36, 2, 0, 0, 0, 0, 3, 1.0636, 4.4684374, 345, 1, 1.06, 0.94              ,0.01 ,   0.01 ],
        [37, 2, 0, 0, 0, 0, 2, 1.0275, -1.5828988, 345, 1, 1.06, 0.94             ,0.01 ,   0.01 ],
        [38, 2, 0, 0, 0, 0, 3, 1.0265, 3.8928177, 345, 1, 1.06, 0.94              ,0.01 ,   0.01 ],
        [39, 2, 1104, 250, 0, 0, 1, 1.03, -14.535256, 345, 1, 1.06, 0.94          ,0.01 ,   0.01 ]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [30, 250, 161.762, 400, 140, 1.0499, 100, 1, 1040, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [31, 677.871, 221.574, 300, -100, 0.982, 100, 1, 646, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [32, 650, 206.965, 300, 150, 0.9841, 100, 1, 725, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [33, 632, 108.293, 250, 0, 0.9972, 100, 1, 652, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [34, 508, 166.688, 167, 0, 1.0123, 100, 1, 508, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [35, 650, 210.661, 300, -100, 1.0494, 100, 1, 687, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [36, 560, 100.165, 240, 0, 1.0636, 100, 1, 580, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [37, 540, -1.36945, 250, 0, 1.0275, 100, 1, 564, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [38, 830, 21.7327, 300, -150, 1.0265, 100, 1, 865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [39, 1000, 78.4674, 300, -100, 1.03, 100, 1, 1100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 2,      0,      0.0411, 0.6987, 600, 600, 600, 0, 0, 1, -360, 360],
        [1, 39,     0,      0.025, 0.75, 1000, 1000, 1000, 0, 0, 1, -360, 360],
        [2, 3,      0,      0.0151, 0.2572, 500, 500, 500, 0, 0, 1, -360, 360],
        [2, 25,     0,      0.0086, 0.146, 500, 500, 500, 0, 0, 1, -360, 360],
        [2, 30,     0,  0.0181, 0, 900, 900, 2500, 1.025, 0, 1, -360, 360],
        [3, 4,      0,      0.0213, 0.2214, 500, 500, 500, 0, 0, 1, -360, 360],
        [3, 18,     0,       0.0133, 0.2138, 500, 500, 500, 0, 0, 1, -360, 360],
        [4, 5,      0,      0.0128, 0.1342, 600, 600, 600, 0, 0, 1, -360, 360],
        [4, 14,     0,       0.0129, 0.1382, 500, 500, 500, 0, 0, 1, -360, 360],
        [5, 6,      0,      0.0026, 0.0434, 1200, 1200, 1200, 0, 0, 1, -360, 360],
        [5, 8,      0,      0.0112, 0.1476, 900, 900, 900, 0, 0, 1, -360, 360],
        [6, 7,      0,      0.0092, 0.113, 900, 900, 900, 0, 0, 1, -360, 360],
        [6, 11,     0,       0.0082, 0.1389, 480, 480, 480, 0, 0, 1, -360, 360],
        [6, 31,     0,  0.025, 0, 1800, 1800, 1800, 1.07, 0, 1, -360, 360],
        [7, 8,      0,      0.0046, 0.078, 900, 900, 900, 0, 0, 1, -360, 360],
        [8, 9,      0,      0.0363, 0.3804, 900, 900, 900, 0, 0, 1, -360, 360],
        [9, 39,     0,      0.025, 1.2, 900, 900, 900, 0, 0, 1, -360, 360],
        [10, 11,    0, 0.0043, 0.0729, 600, 600, 600, 0, 0, 1, -360, 360],
        [10, 13,    0, 0.0043, 0.0729, 600, 600, 600, 0, 0, 1, -360, 360],
        [10, 32,    0,   0.02, 0, 900, 900, 2500, 1.07, 0, 1, -360, 360],
        [12, 11,    0, 0.0435, 0, 500, 500, 500, 1.006, 0, 1, -360, 360],
        [12, 13,    0, 0.0435, 0, 500, 500, 500, 1.006, 0, 1, -360, 360],
        [13, 14,    0, 0.0101, 0.1723, 600, 600, 600, 0, 0, 1, -360, 360],
        [14, 15,    0, 0.0217, 0.366, 600, 600, 600, 0, 0, 1, -360, 360],
        [15, 16,    0, 0.0094, 0.171, 600, 600, 600, 0, 0, 1, -360, 360],
        [16, 17,    0, 0.0089, 0.1342, 600, 600, 600, 0, 0, 1, -360, 360],
        [16, 19,    0, 0.0195, 0.304, 600, 600, 2500, 0, 0, 1, -360, 360],
        [16, 21,    0, 0.0135, 0.2548, 600, 600, 600, 0, 0, 1, -360, 360],
        [16, 24,    0, 0.0059, 0.068, 600, 600, 600, 0, 0, 1, -360, 360],
        [17, 18,    0, 0.0082, 0.1319, 600, 600, 600, 0, 0, 1, -360, 360],
        [17, 27,    0, 0.0173, 0.3216, 600, 600, 600, 0, 0, 1, -360, 360],
        [19, 20,    0, 0.0138, 0, 900, 900, 2500, 1.06, 0, 1, -360, 360],
        [19, 33,    0, 0.0142, 0, 900, 900, 2500, 1.07, 0, 1, -360, 360],
        [20, 34,    0, 0.018, 0, 900, 900, 2500, 1.009, 0, 1, -360, 360],
        [21, 22,    0, 0.014, 0.2565, 900, 900, 900, 0, 0, 1, -360, 360],
        [22, 23,    0, 0.0096, 0.1846, 600, 600, 600, 0, 0, 1, -360, 360],
        [22, 35,    0,   0.0143, 0, 900, 900, 2500, 1.025, 0, 1, -360, 360],
        [23, 24,    0, 0.035, 0.361, 600, 600, 600, 0, 0, 1, -360, 360],
        [23, 36,    0, 0.0272, 0, 900, 900, 2500, 1, 0, 1, -360, 360],
        [25, 26,    0, 0.0323, 0.531, 600, 600, 600, 0, 0, 1, -360, 360],
        [25, 37,    0, 0.0232, 0, 900, 900, 2500, 1.025, 0, 1, -360, 360],
        [26, 27,    0, 0.0147, 0.2396, 600, 600, 600, 0, 0, 1, -360, 360],
        [26, 28,    0, 0.0474, 0.7802, 600, 600, 600, 0, 0, 1, -360, 360],
        [26, 29,    0, 0.0625, 1.029, 600, 600, 600, 0, 0, 1, -360, 360],
        [28, 29,    0, 0.0151, 0.249, 600, 600, 600, 0, 0, 1, -360, 360],
        [29, 38,    0, 0.0156, 0, 1200, 1200, 2500, 1.025, 0, 1, -360, 360]
    ])

    ppc["gencontrol"] = array([
        [ 30         , 3          , 0.165521   , 0.165521   , 1.655212   , 1.655212   , -31.2      , 31.2       ],
        [ 31         , 3          , 0.002056   , 0.002056   , 0.020563   , 2.056282   , -19.38     , 19.38      ],
        [ 32         , 3          , 0.115388   , 0.115388   , 0.023077   , 2.307747   , -21.75     , 21.75      ],
        [ 33         , 3          , 0.002075   , 0.207538   , 1.03769    , 1.03769    , -19.56     , 19.56      ],
        [ 34         , 3          , 0.001617   , 0.161701   , 0.01617    , 1.617014   , -15.24     , 15.24      ],
        [ 35         , 3          , 0.10934    , 0.10934    , 1.093394   , 1.093394   , -20.61     , 20.61      ],
        [ 36         , 3          , 0.001846   , 0.001846   , 0.018462   , 1.846197   , -17.4      , 17.4       ],
        [ 37         , 3          , 0.089764   , 0.089764   , 0.017953   , 1.795268   , -16.92     , 16.92      ],
        [ 38         , 3          , 0.002753   , 0.275338   , 1.37669    , 1.37669    , -25.95     , 25.95      ],
        [ 39         , 3          , 0.003501   , 0.350141   , 0.035014   , 3.501409   , -33.0      , 33.0       ]
    ])

    
    ###----- disturbances data -----###
    #1 - power-step: num, buses, start time, amplitude(\times p_0)
    #2 - power ramp: num, buses, start time, duration time, height (\times p_0)
    #3 - power fluctuation: num, buses, start time, time interval, date num
    #4 - 3psc: num, branch-f_bus,  branch-t_bus, nearest bus, short-circuit conductance (p.u.), start time, clearing time(\Delta t with second)
    #9: uniform distribution date on [-0.2, 0.2]

    ppc["disturbance"] = {
    1: array([
        [1,     32,      0,      -0.5, 0.15],
        [2,     3,      0,      -0.5,  0.15]
    ]),
    2: array([
        [1,    35,      0,      5,      -0.5, 0.15],
        [2,    8,      0,      5,      -0.5,  0.15]
    ]),
    3: array([
        [1,    39,      0,      0.5,      9, 0.6],
        [2,    21,      0,      0.5,      9, 0.6]
    ]),
    4: array([
        [1,    4,      14,       999,      999,    0,      0.1, 0.1],
        [2,    17,     27,       999,      999,    0,      0.1, 0.1]
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
