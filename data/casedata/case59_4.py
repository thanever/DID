# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for IEEE 14 bus test case.
"""

from numpy import array

def case59_4():
    """Power flow data for IEEE 14 bus test case.
    Please see L{caseformat} for details on the case file format.

    This data was converted from IEEE Common Data Format
    (ieee14cdf.txt) on 20-Sep-2004 by cdf2matp, rev. 1.11

    Converted from IEEE CDF file from:
    U{http://www.ee.washington.edu/research/pstca/}

    08/19/93 UW ARCHIVE           100.0  1962 W IEEE 14 Bus Test Case

    @return: Power flow data for IEEE 14 bus test case.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1       ,	2,	0,	0,	0,	0,	1,	1,	-21.6109,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [2       ,	1,	270,	30,	0,	0,	1,	1.03505,	-21.6109,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [3       ,	2,	0,	0,	0,	0,	2,	1,	5.8891,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [4       ,	2,	0,	0,	0,	0,	2,	1,	1.5272,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [5       ,	2,	0,	0,	0,	0,	2,	1,	1.1932,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [6       ,	3,	0,	0,	0,	0,	2,	1,	0,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [7       ,	2,	235,	25+39.367,	0,	0,	2,	1.045,	-10.7006,	330,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [8       ,	1,	80,	10,	0,	0,	2,	1.0157,	-1.4416,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [9       ,	1,	1130,	120,	0,	0,	2,	1.00308,	-7.4573,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [10      ,	1,	125,	15,	0,	0,	2,	1.00744,	-7.0143,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [11      ,	1,	0,	0,	0,	0,	2,	1.00944,	-6.0141,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [12      ,	1,	0,	0,	0,	0,	2,	1.04585,	-8.2698,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [13      ,	1,	1060,	110,	0,	0,	2,	1.01444,	-12.533,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [14      ,	1,	1000,	110,	0,	400,	2,	1.02324,	-12.8034,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [15      ,	1,	0,	0,	0,	0,	2,	1.04484,	-10.6555,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [16      ,	1,	0,	0,	0,	0,	2,	1.03741,	-12.901,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [17      ,	1,	290,	30,	0,	0,	2,	1.01499,	-5.4176,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [18      ,	1,	1105,	120,	0,	300,	2,	1.03448,	-14.6753,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [19      ,	1,	750,	80,	0,	0,	2,	1.03223,	-17.4582,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [20      ,	2,	0,	0,	0,	0,	3,	1,	-6.3752,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [21      ,	2,	0,	0,	0,	0,	3,	1,	-15.3751,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [22      ,	1,	0,	0,	0,	0,	3,	1.00488,	-13.9924,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [23      ,	1,	0,	0,	0,	0,	3,	1.00542,	-22.8954,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [24      ,	1,	0,	0,	0,	0,	3,	1.0163,	-24.0067,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [25      ,	1,	900,	90,	0,	0,	3,	1.0183,	-25.5714,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [26      ,	1,	470,	50,	0,	0,	3,	1.02074,	-25.6044,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [27      ,	1,	620,	100,	0,	0,	3,	1.0441,	-32.9496,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [28      ,	1,	140,	15,	0,	0,	3,	1.04184,	-23.2471,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [29      ,	1,	0,	0,	0,	0,	3,	1.00776,	-24.165,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [30      ,	1,	0,	0,	0,	0,	3,	1.03746,	-23.6261,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [31      ,	1,	92,	10,	0,	0,	3,	1.01132,	-22.6716,	220,	1,	1.1,	0.9,        0 ,      0.01 ],
        [32      ,	2,	1625,	165-86.68,	0,	0,	3,	1.015,	-28.3633,	220,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [33      ,	1,	180,	20,	0,	0,	3,	1.01615,	-27.4327,	220,	1,	1.1,	0.9,        0 ,      0.01 ],
        [34      ,	1,	0,	0,	0,	0,	3,	1.04266,	-34.3703,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [35      ,	2,	0,	0,	0,	0,	4,	1,	-13.1037,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [36      ,	2,	0,	0,	0,	0,	4,	1,	2.2181,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [37      ,	2,	0,	0,	0,	0,	4,	1,	1.9537,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [38      ,	2,	0,	0,	0,	0,	4,	1,	-4.726,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [39      ,	1,	730,	75,	0,	0,	4,	1.00788,	-11.03,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [40      ,	1,	540,	55,	0,	0,	4,	0.99929,	-6.1997,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [41      ,	1,	0,	0,	0,	0,	4,	1.00106,	-4.249,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [42      ,	1,	110,	10,	0,	0,	4,	1.01207,	-6.1851,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [43      ,	1,	190,	20,	0,	60,	4,	1.03289,	-21.2613,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [44      ,	1,	390,	40,	0,	0,	4,	1.01319,	-19.2319,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [45      ,	1,	420,	45,	0,	30,	4,	1.00629,	-25.8081,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [46      ,	2,	922,	100+53.018,	0,	0,	4,	1,	-25.9326,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [47      ,	1,	0,	0,	0,	0,	4,	1.04649,	-16.8203,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [48      ,	1,	0,	0,	0,	-30,	4,	1.03316,	-16.5543,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [49      ,	1,	0,	0,	0,	-60,	4,	1.04457,	-15.3269,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [50      ,	1,	0,	0,	0,	-60,	4,	1.05436,	-13.0593,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [51      ,	2,	0,	0,	0,	0,	5,	1,	-31.4249,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [52      ,	2,	0,	0,	0,	0,	5,	1,	-32.988,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [53      ,	2,	0,	0,	0,	0,	5,	1,	-32.4534,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [54      ,	1,	180,	20,	0,	0,	5,	1.03667,	-39.3425,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [55      ,	1,	0,	0,	0,	0,	5,	1.00777,	-39.5519,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [56      ,	1,	0,	0,	0,	0,	5,	1.00901,	-41.0027,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [57      ,	2,	640,	65+4.023,	0,	0,	5,	1.01,	-41.5366,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [58      ,	1,	490,	50,	0,	0,	5,	1.00469,	-41.6212,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [59      ,	2,	122,	15+109.263,	0,	0,	5,	1.03,	-36.9831,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ]
    ])


    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf  # p_g of 3 to 8 is all change from 0 to 40
    ppc["gen"] = array([
        [1  ,	0,	-194.719,	290.564,	-290.564,	1,	666.6,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [3  ,	2160,	-123.271,	1162.432,	-1162.432,	1,	2666.8,	1,	2400,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [4  ,	1380,	-7.371,	726.54,	-726.54,	1,	1666.8,	1,	1500,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [5  ,	1410,	28.229,	726.54,	-726.54,	1,	1666.8,	1,	1500,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [6  ,	1597.149,	-174.217,	1162.432,	-1162.432,	1,	2666.8,	1,	2400,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [20 ,	3330,	99.731,	1743.648,	-1743.648,	1,	4000.2,	1,	3600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [21 ,	760,	-18.509,	387.418,	-387.418,	1,	888.8,	1,	800,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [35 ,	960,	-65.659,	581.127,	-581.127,	1,	1333.2,	1,	1200,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [36 ,	580,	-4.706,	290.564,	-290.564,	1,	666.6,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [37 ,	960,	42.633,	581.127,	-581.127,	1,	1333.2,	1,	1200,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [38 ,	651,	-10.518,	435.846,	-435.846,	1,	999.9,	1,	900,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [51 ,	560,	-105.051,	290.564,	-290.564,	1,	666.6,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [52 ,	540,	-5.462,	450,	-450,	1,	750,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [53 ,	150,	2.203,	72.663,	-72.663,	1,	166.7,	1,	150,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [2       , 19    ,	0,      0.0667,	0.817,	0,	0,	0,	0,	0,	1,	-360,	360],
        [2       , 19    ,	0,      0.0667,	0.817,	0,	0,	0,	0,	0,	1,	-360,	360],
        [2       , 19    ,	0,      0.062,	0.76,	0,	0,	0,	0,	0,	1,	-360,	360],
        [2       , 19    ,	0,      0.062,	0.76,	0,	0,	0,	0,	0,	1,	-360,	360],
        [2       , 28    ,	0,      0.0356,	0.437,	0,	0,	0,	0,	0,	1,	-360,	360],
        [2       , 28    ,	0,      0.0356,	0.437,	0,	0,	0,	0,	0,	1,	-360,	360],
        [2       , 28    ,	0,      0.0868,	0.76,	0,	0,	0,	0,	0,	1,	-360,	360],
        [7       , 8     ,	0,      0.076,	0.931,	0,	0,	0,	0,	0,	1,	-360,	360],
        [7       , 8     ,	0,      0.076,	0.931,	0,	0,	0,	0,	0,	1,	-360,	360],
        [7       , 50    ,	0,      0.046,	0.73,	0,	0,	0,	0,	0,	1,	-360,	360],
        [7       , 50    ,	0,      0.046,	0.73,	0,	0,	0,	0,	0,	1,	-360,	360],
        [8       , 9     ,	0,      0.0356,	0.437,	0,	0,	0,	0,	0,	1,	-360,	360],
        [8       , 9     ,	0,      0.0356,	0.437,	0,	0,	0,	0,	0,	1,	-360,	360],
        [8       , 14    ,	0,      0.0527,	0.646,	0,	0,	0,	0,	0,	1,	-360,	360],
        [8       , 14    ,	0,      0.0527,	0.646,	0,	0,	0,	0,	0,	1,	-360,	360],
        [8       , 17    ,	0,      0.0527,	0.646,	0,	0,	0,	0,	0,	1,	-360,	360],
        [8       , 17    ,	0,      0.0527,	0.646,	0,	0,	0,	0,	0,	1,	-360,	360],
        [9       , 10    ,	0,      0.014,	0.171,	0,	0,	0,	0,	0,	1,	-360,	360],
        [9       , 10    ,	0,      0.014,	0.171,	0,	0,	0,	0,	0,	1,	-360,	360],
        [9       , 11    ,	0,      0.0062,	0.076,	0,	0,	0,	0,	0,	1,	-360,	360],
        [10      , 13    ,	0,      0.0248,	0.304,	0,	0,	0,	0,	0,	1,	-360,	360],
        [10      , 13    ,	0,      0.0248,	0.304,	0,	0,	0,	0,	0,	1,	-360,	360],
        [10      , 13    ,	0,      0.0248,	0.304,	0,	0,	0,	0,	0,	1,	-360,	360],
        [11      , 14    ,	0,      0.0356,	0.437,	0,	0,	0,	0,	0,	1,	-360,	360],
        [12      , 15    ,	0,      0.0145,	1.54,	0,	0,	0,	0,	0,	1,	-360,	360],
        [12      , 15    ,	0,      0.0145,	1.54,	0,	0,	0,	0,	0,	1,	-360,	360],
        [13      , 14    ,	0,      0.0108,	0.133,	0,	0,	0,	0,	0,	1,	-360,	360],
        [13      , 14    ,	0,      0.0108,	0.133,	0,	0,	0,	0,	0,	1,	-360,	360],
        [13      , 16    ,	0,      0.0155,	0.19,	0,	0,	0,	0,	0,	1,	-360,	360],
        [14      , 19    ,	0,      0.0558,	0.684,	0,	0,	0,	0,	0,	1,	-360,	360],
        [16      , 18    ,	0,      0.0077,	0.095,	0,	0,	0,	0,	0,	1,	-360,	360],
        [16      , 19    ,	0,      0.0388,	0.475,	0,	0,	0,	0,	0,	1,	-360,	360],
        [17      , 18    ,	0,      0.0403,	0.494,	0,	0,	0,	0,	0,	1,	-360,	360],
        [17      , 18    ,	0,      0.0403,	0.494,	0,	0,	0,	0,	0,	1,	-360,	360],
        [17      , 19    ,	0,      0.0574,	0.703,	0,	0,	0,	0,	0,	1,	-360,	360],
        [17      , 19    ,	0,      0.0574,	0.703,	0,	0,	0,	0,	0,	1,	-360,	360],
        [18      , 19    ,	0,      0.0403,	0.494,	0,	0,	0,	0,	0,	1,	-360,	360],
        [22      , 23    ,	0,      0.028,	0.74,	0,	0,	0,	0,	0,	1,	-360,	360],
        [22      , 23    ,	0,      0.028,	0.74,	0,	0,	0,	0,	0,	1,	-360,	360],
        [22      , 24    ,	0,      0.016,	1.7,	0,	0,	0,	0,	0,	1,	-360,	360],
        [22      , 24    ,	0,      0.016,	1.7,	0,	0,	0,	0,	0,	1,	-360,	360],
        [23      , 24    ,	0,      0.004,	0.424,	0,	0,	0,	0,	0,	1,	-360,	360],
        [24      , 25    ,	0,      0.003,	0.32,	0,	0,	0,	0,	0,	1,	-360,	360],
        [24      , 26    ,	0,      0.0045,	0.447,	0,	0,	0,	0,	0,	1,	-360,	360],
        [24      , 26    ,	0,      0.0045,	0.447,	0,	0,	0,	0,	0,	1,	-360,	360],
        [25      , 26    ,	0,      0.0012,	0.127,	0,	0,	0,	0,	0,	1,	-360,	360],
        [26      , 27    ,	0,      0.0325,	3.445,	0,	0,	0,	0,	0,	1,	-360,	360],
        [26      , 27    ,	0,      0.0325,	3.445,	0,	0,	0,	0,	0,	1,	-360,	360],
        [28      , 29    ,	0,      0.10695,	0.58267,	0,	0,	0,	0,	0,	1,	-360,	360],
        [28      , 29    ,	0,      0.10695,	0.58267,	0,	0,	0,	0,	0,	1,	-360,	360],
        [28      , 29    ,	0,      0.10695,	0.58267,	0,	0,	0,	0,	0,	1,	-360,	360],
        [29      , 30    ,	0,      -0.0337,	0,	0,	0,	0,	0,	0,	1,	-360,	360],
        [29      , 30    ,	0,      -0.0337,	0,	0,	0,	0,	0,	0,	1,	-360,	360],
        [31      , 32    ,	0,      0.045,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [31      , 32    ,	0,      0.045,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [31      , 32    ,	0,      0.045,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [32      , 33    ,	0,      0.01,	0.26,	0,	0,	0,	0,	0,	1,	-360,	360],
        [32      , 33    ,	0,      0.01,	0.26,	0,	0,	0,	0,	0,	1,	-360,	360],
        [34      , 59    ,	0,      0.05,	0.19,	0,	0,	0,	0,	0,	1,	-360,	360],
        [34      , 59    ,	0,      0.05,	0.19,	0,	0,	0,	0,	0,	1,	-360,	360],
        [39      , 40    ,	0,      0.0475,	0.381,	0,	0,	0,	0,	0,	1,	-360,	360],
        [39      , 40    ,	0,      0.0475,	0.381,	0,	0,	0,	0,	0,	1,	-360,	360],
        [39      , 42    ,	0,      0.05,	0.189,	0,	0,	0,	0,	0,	1,	-360,	360],
        [39      , 43    ,	0,      0.122,	0.79,	0,	0,	0,	0,	0,	1,	-360,	360],
        [39      , 43    ,	0,      0.122,	0.79,	0,	0,	0,	0,	0,	1,	-360,	360],
        [39      , 43    ,	0,      0.122,	0.79,	0,	0,	0,	0,	0,	1,	-360,	360],
        [40      , 41    ,	0,      0.0076,	0.062,	0,	0,	0,	0,	0,	1,	-360,	360],
        [40      , 41    ,	0,      0.0076,	0.062,	0,	0,	0,	0,	0,	1,	-360,	360],
        [41      , 42    ,	0,      0.0513,	0.412,	0,	0,	0,	0,	0,	1,	-360,	360],
        [42      , 44    ,	0,      0.192,	0.67333,	0,	0,	0,	0,	0,	1,	-360,	360],
        [42      , 44    ,	0,      0.192,	0.67333,	0,	0,	0,	0,	0,	1,	-360,	360],
        [42      , 44    ,	0,      0.192,	0.67333,	0,	0,	0,	0,	0,	1,	-360,	360],
        [43      , 45    ,	0,      0.0709,	0.46,	0,	0,	0,	0,	0,	1,	-360,	360],
        [43      , 45    ,	0,      0.0709,	0.46,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 45    ,	0,      0.0532,	0.427,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 46    ,	0,      0.0532,	0.427,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 46    ,	0,      0.0532,	0.427,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 46    ,	0,      0.0532,	0.427,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 46    ,	0,      0.0532,	0.427,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 47    ,	0,      0.0494,	0.4,	0,	0,	0,	0,	0,	1,	-360,	360],
        [44      , 47    ,	0,      0.0494,	0.4,	0,	0,	0,	0,	0,	1,	-360,	360],
        [45      , 46    ,	0,      0.0152,	0.122,	0,	0,	0,	0,	0,	1,	-360,	360],
        [45      , 46    ,	0,      0.0152,	0.122,	0,	0,	0,	0,	0,	1,	-360,	360],
        [48      , 49    ,	0,      0.025,	0.39,	0,	0,	0,	0,	0,	1,	-360,	360],
        [48      , 49    ,	0,      0.025,	0.39,	0,	0,	0,	0,	0,	1,	-360,	360],
        [49      , 50    ,	0,      0.046,	0.73,	0,	0,	0,	0,	0,	1,	-360,	360],
        [49      , 50    ,	0,      0.046,	0.73,	0,	0,	0,	0,	0,	1,	-360,	360],
        [54      , 57    ,	0,      0.15,	0.56,	0,	0,	0,	0,	0,	1,	-360,	360],
        [54      , 57    ,	0,      0.15,	0.56,	0,	0,	0,	0,	0,	1,	-360,	360],
        [54      , 58    ,	0,      0.019,	0.87,	0,	0,	0,	0,	0,	1,	-360,	360],
        [54      , 58    ,	0,      0.019,	0.87,	0,	0,	0,	0,	0,	1,	-360,	360],
        [55      , 57    ,	0,      0.017,	0.03,	0,	0,	0,	0,	0,	1,	-360,	360],
        [55      , 57    ,	0,      0.017,	0.03,	0,	0,	0,	0,	0,	1,	-360,	360],
        [55      , 58    ,	0,      0.028,	0.17,	0,	0,	0,	0,	0,	1,	-360,	360],
        [56      , 57    ,	0,      0.017,	0.03,	0,	0,	0,	0,	0,	1,	-360,	360],
        [56      , 57    ,	0,      0.017,	0.03,	0,	0,	0,	0,	0,	1,	-360,	360],
        [56      , 58    ,	0,      0.028,	0.14,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 58    ,	0,      0.019,	0.09,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 59    ,	0,      0.66,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 59    ,	0,      0.66,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 59    ,	0,      0.66,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 59    ,	0,      0.66,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 59    ,	0,      0.66,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [57      , 59    ,	0,      0.66,	0.3,	0,	0,	0,	0,	0,	1,	-360,	360],
        [1       , 2     ,	0,      0.018,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [3       , 8     ,	0,      0.006,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [4       , 11    ,	0,      0.0096,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [5       , 10    ,	0,      0.0102,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [6       , 17    ,	0,      0.006,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [20      , 22    ,	0,      0.004,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [21      , 31    ,	0,      0.0169,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [23      , 32    ,	0,      0.032,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [23      , 32    ,	0,      0.032,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 33    ,	0,      0.024,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 33    ,	0,      0.024,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [27      , 34    ,	0,      0.027,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [27      , 34    ,	0,      0.027,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [35      , 44    ,	0,      0.01127,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [36      , 42    ,	0,      0.0255,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [37      , 41    ,	0,      0.01127,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [38      , 39    ,	0,      0.017,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1.01502233,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1.01502233,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1.01502233,	0,	1,	-360,	360],
        [51      , 54    ,	0,      0.0255,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [52      , 55    ,	0,      0.02133,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [53      , 56    ,	0,      0.1,	0,	0,	0,	0,	1,	0,	1,	-360,	360]
    ])

     ##----- Generator bus control data  -----##
    # type of generator or inverter, min_m, max_m, min_d, max_d, min_p (p.u.), max_p (p.u.)
	#  type of generator or inverter: 9 - conventional generator, 0 - inverter with no optimize, 1 - inverter optimizing d, 2 - inverter optimizing m, 3 - inverters optimizing d and m
    ppc["gencontrol"] = array([
            [ 1          , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ],
            [ 3          , 3          , 0.015279   , 1.527887   , 0.076394   , 7.639437   , -72.0      , 72.0       ],
            [ 4          , 3          , 0.009549   , 0.95493    , 0.047746   , 4.774648   , -45.0      , 45.0       ],
            [ 5          , 3          , 0.009549   , 0.95493    , 0.047746   , 4.774648   , -45.0      , 45.0       ],
            [ 6          , 3          , 0.015279   , 1.527887   , 0.076394   , 7.639437   , -72.0      , 72.0       ],
            [ 20         , 3          , 0.022918   , 2.291831   , 0.114592   , 11.459156  , -108.0     , 108.0      ],
            [ 21         , 3          , 0.005093   , 0.509296   , 0.025465   , 2.546479   , -24.0      , 24.0       ],
            [ 35         , 3          , 0.007639   , 0.763944   , 0.038197   , 3.819719   , -36.0      , 36.0       ],
            [ 36         , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ],
            [ 37         , 3          , 0.007639   , 0.763944   , 0.038197   , 3.819719   , -36.0      , 36.0       ],
            [ 38         , 3          , 0.00573    , 0.572958   , 0.028648   , 2.864789   , -27.0      , 27.0       ],
            [ 51         , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ],
            [ 52         , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ],
            [ 53         , 3          , 0.000955   , 0.095493   , 0.004775   , 0.477465   , -4.5       , 4.5        ]
    ])


    ###----- disturbances data -----###
    #1 - power-step: num, buses, start time, amplitude(\times p_0)
    #2 - power ramp: num, buses, start time, duration time, height (\times p_0)
    #3 - power fluctuation: num, buses, start time, time interval, date num
    #4 - 3psc: num, branch-f_bus,  branch-t_bus, nearest bus, short-circuit conductance (p.u.), start time, clearing time(\Delta t with second)
    #9: uniform distribution date on [-0.2, 0.2]

    ppc["disturbance"] = {
    1: array([
        [1,     5,      0,      -0.5, 0.09],
        [2,     2,      0,      -0.5, 0.09]
    ]),
    2: array([
        [1,    21,      0,      5,      -0.5, 0.09],
        [2,    58,      0,      5,      -0.5, 0.09]
    ]),
    3: array([
        [1,    38,      0,     0.5,    9, 0.3],
        [2,    35,     0,     0.5,      9, 0.3]
    ]),
    4: array([
        [1,    44,     45,       999,      999,    0,      0.1, 0.01],
        [2,    14,     19,       999,      999,    0,      0.1, 0.01]
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
    'time_ele':{1: 10, 2: 10, 3: 20, 4: 10},
    'time_ele_d':{2: 4, 4: 4},
    'order':{1: 3, 2: 3, 3: 3, 4: 3},
    't_f':{1: 10, 2: 10, 3: 30, 4: 10},
    'colloc_point_radau': {1:(0, 1.0),2:(0, 0.333333, 1.0),3:(0, 0.155051, 0.644949, 1.0),4:(0, 0.088588, 0.409467, 0.787659, 1.0),5:(0, 0.057104, 0.276843, 0.583590, 0.860240, 1.0)}
    }

    ppc["freq_band"] = {
    1:{(0, 15):(49.5, 50.5), (15, 300):(49.85, 50.15)},
    2:{(0, 15):(49.5, 50.5), (15, 300):(49.85, 50.15)},
    3:{(0, 300):(49.85, 50.15)},
    4:{(0, 15):(49, 51), (15, 60):(49.5, 50.5), (60, 300):(49.85, 50.15)}
    }

    return ppc