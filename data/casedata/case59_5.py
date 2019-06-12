# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for IEEE 14 bus test case.
"""

from numpy import array

def case59_5():
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
        [1       ,	2,	0,	0,	0,	0,	1,	1,	-34.8374,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [2       ,	1,	340,	35,	0,	0,	1,	1.01192,	-30.7573,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [3       ,	2,	0,	0,	0,	0,	2,	1,	6.6377,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [4       ,	2,	0,	0,	0,	0,	2,	1,	-1.6055,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [5       ,	2,	0,	0,	0,	0,	2,	1,	-5.2144,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [6       ,	3,	0,	0,	0,	0,	2,	1,	0,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [7       ,	2,	290,	30+118.315,	0,	0,	2,	1.045,	-0.1466,	330,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [8       ,	1,	100,	10,	0,	0,	2,	1.02926,	-0.6459,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [9       ,	1,	1410,	145,	0,	0,	2,	1.00881,	-11.4949,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [10      ,	1,	160,	20,	0,	0,	2,	1.01454,	-12.9559,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [11      ,	1,	0,	0,	0,	0,	2,	1.02004,	-9.1664,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [12      ,	1,	0,	0,	0,	0,	2,	1.05053,	-11.99,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [13      ,	1,	1275,	130,	0,	0,	2,	1.0113,	-18.1414,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [14      ,	1,	1245,	125,	0,	400,	2,	1.01666,	-17.8022,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [15      ,	1,	0,	0,	0,	0,	2,	1.04309,	-14.9884,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [16      ,	1,	0,	0,	0,	0,	2,	1.03141,	-17.8606,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [17      ,	1,	360,	40,	0,	0,	2,	1.02469,	-6.9801,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [18      ,	1,	1380,	140,	0,	300,	2,	1.02452,	-19.7406,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [19      ,	1,	940,	95,	0,	0,	2,	1.01231,	-23.9397,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [20      ,	2,	0,	0,	0,	0,	3,	1,	-2.6394,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [21      ,	2,	0,	0,	0,	0,	3,	1,	-14.3288,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [22      ,	1,	0,	0,	0,	0,	3,	1.02758,	-9.7342,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [23      ,	1,	0,	0,	0,	0,	3,	1.00906,	-21.097,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [24      ,	1,	0,	0,	0,	0,	3,	1.0191,	-22.5877,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [25      ,	1,	1085,	110,	0,	0,	3,	1.02056,	-24.3762,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [26      ,	1,	580,	60,	0,	0,	3,	1.02321,	-24.3805,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [27      ,	1,	580,	60,	0,	0,	3,	1.05049,	-31.7609,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [28      ,	1,	170,	20,	0,	0,	3,	1.02293,	-29.1694,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [29      ,	1,	0,	0,	0,	0,	3,	1.01212,	-21.5711,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [30      ,	1,	0,	0,	0,	0,	3,	1.03772,	-25.0417,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [31      ,	1,	105,	15,	0,	0,	3,	1.03181,	-20.5127,	220,	1,	1.1,	0.9,        0 ,      0.01 ],
        [32      ,	2,	2130,	220-54.861,	0,	0,	3,	1.015,	-28.0344,	220,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [33      ,	1,	222,	25,	0,	0,	3,	1.01675,	-26.8631,	220,	1,	1.1,	0.9,        0 ,      0.01 ],
        [34      ,	1,	0,	0,	0,	0,	3,	1.04707,	-33.5134,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [35      ,	2,	0,	0,	0,	0,	4,	1,	20.8208,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [36      ,	2,	0,	0,	0,	0,	4,	1,	49.8063,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [37      ,	2,	0,	0,	0,	0,	4,	1,	53.5283,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [38      ,	2,	0,	0,	0,	0,	4,	1,	47.0741,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [39      ,	1,	990,	100,	0,	0,	4,	1.03397,	39.733,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [40      ,	1,	740,	75,	0,	0,	4,	1.03583,	45.1429,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [41      ,	1,	0,	0,	0,	0,	4,	1.04084,	47.4921,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [42      ,	1,	150,	15,	0,	0,	4,	1.03688,	42.2693,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [43      ,	1,	260,	30,	0,	60,	4,	1.00362,	19.7786,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [44      ,	1,	530,	55,	0,	0,	4,	1.0287,	14.6049,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [45      ,	1,	575,	60,	0,	30,	4,	0.99742,	7.8867,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [46      ,	2,	1255,	130-22.828,	0,	0,	4,	1,	6.9244,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [47      ,	1,	0,	0,	0,	0,	4,	1.05247,	10.2137,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [48      ,	1,	0,	0,	0,	-30,	4,	1.03942,	9.7541,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [49      ,	1,	0,	0,	0,	-60,	4,	1.04779,	7.5743,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [50      ,	1,	0,	0,	0,	-60,	4,	1.05445,	3.673,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [51      ,	2,	0,	0,	0,	0,	5,	1,	-33.4105,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [52      ,	2,	0,	0,	0,	0,	5,	1,	-33.0178,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [53      ,	2,	0,	0,	0,	0,	5,	1,	-37.2042,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [54      ,	1,	225,	25,	0,	0,	5,	1.04224,	-41.1683,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [55      ,	1,	0,	0,	0,	0,	5,	1.0122,	-39.883,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [56      ,	1,	0,	0,	0,	0,	5,	1.01517,	-42.0475,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [57      ,	2,	750,	75-13.793,	0,	0,	5,	1.015,	-42.5626,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [58      ,	1,	600,	60,	0,	0,	5,	1.01056,	-43.0584,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [59      ,	2,	150,	15+123.834,	0,	0,	5,	1.03,	-36.7437,	275,	1,	1.1,	0.9,        0.01 ,   0.01]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf  # p_g of 3 to 8 is all change from 0 to 40
    ppc["gen"] = array([
        [1  ,	-600,	-77.957,	435.846,	-435.846,	1,	999.9,	1,	900,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [3  ,	2800,	193.445,	1453.04,	-1453.04,	1,	3333.5,	1,	3000,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [4  ,	1920,	268.884,	968.72,	-968.72,	1,	2222.4,	1,	2000,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [5  ,	920,	166.124,	484.36,	-484.36,	1,	1111.2,	1,	1000,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [6  ,	2137.635,	220.967,	1162.432,	-1162.432,	1,	2666.8,	1,	2400,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [20 ,	4400,	704.745,	2324.864,	-2324.864,	1,	5333.6,	1,	4800,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [21 ,	1026,	131.429,	581.127,	-581.127,	1,	1333.2,	1,	1200,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [35 ,	1384,	339.579,	774.836,	-774.836,	1,	1777.6,	1,	1600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [36 ,	840,	136.25,	435.846,	-435.846,	1,	999.9,	1,	900,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [37 ,	1360,	185.293,	774.836,	-774.836,	1,	1777.6,	1,	1600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [38 ,	1360,	252.042,	726.41,	-726.41,	1,	1666.5,	1,	1500,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [51 ,	560,	-70.361,	290.564,	-290.564,	1,	666.6,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [52 ,	760,	0.532,	600,	-600,	1,	1000,	1,	800,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [53 ,	174,	7.05,	145.326,	-145.326,	1,	333.4,	1,	300,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
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
        [1       , 2     ,	0,      0.012,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [3       , 8     ,	0,      0.0048,	0,	0,	0,	0,	0.9709,	0,	1,	-360,	360],
        [4       , 11    ,	0,      0.0072,	0,	0,	0,	0,	0.9709,	0,	1,	-360,	360],
        [5       , 10    ,	0,      0.0153,	0,	0,	0,	0,	0.9709,	0,	1,	-360,	360],
        [6       , 17    ,	0,      0.006,	0,	0,	0,	0,	0.9709,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [20      , 22    ,	0,      0.003,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [21      , 31    ,	0,      0.01127,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [23      , 32    ,	0,      0.032,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [23      , 32    ,	0,      0.032,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 33    ,	0,      0.024,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 33    ,	0,      0.024,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [27      , 34    ,	0,      0.027,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [27      , 34    ,	0,      0.027,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [35      , 44    ,	0,      0.00845,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [36      , 42    ,	0,      0.017,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [37      , 41    ,	0,      0.00845,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [38      , 39    ,	0,      0.0102,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1.015,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1.015,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1.015,	0,	1,	-360,	360],
        [51      , 54    ,	0,      0.0255,	0,	0,	0,	0,	0.9852,	0,	1,	-360,	360],
        [52      , 55    ,	0,      0.016,	0,	0,	0,	0,	0.995,	0,	1,	-360,	360],
        [53      , 56    ,	0,      0.05,	0,	0,	0,	0,	0.9852,	0,	1,	-360,	360]
    ])

     ##----- Generator bus control data  -----##
    # type of generator or inverter, min_m, max_m, min_d, max_d, min_p (p.u.), max_p (p.u.)
	#  type of generator or inverter: 9 - conventional generator, 0 - inverter with no optimize, 1 - inverter optimizing d, 2 - inverter optimizing m, 3 - inverters optimizing d and m
    ppc["gencontrol"] = array([
            [ 1          , 3          , 0.00573    , 0.572958   , 0.028648   , 2.864789   , -27.0      , 27.0       ],
            [ 3          , 3          , 0.019099   , 1.909859   , 0.095493   , 9.549297   , -90.0      , 90.0       ],
            [ 4          , 3          , 0.012732   , 1.27324    , 0.063662   , 6.366198   , -60.0      , 60.0       ],
            [ 5          , 3          , 0.006366   , 0.63662    , 0.031831   , 3.183099   , -30.0      , 30.0       ],
            [ 6          , 3          , 0.015279   , 1.527887   , 0.076394   , 7.639437   , -72.0      , 72.0       ],
            [ 20         , 3          , 0.030558   , 3.055775   , 0.152789   , 15.278875  , -144.0     , 144.0      ],
            [ 21         , 3          , 0.007639   , 0.763944   , 0.038197   , 3.819719   , -36.0      , 36.0       ],
            [ 35         , 3          , 0.010186   , 1.018592   , 0.05093    , 5.092958   , -48.0      , 48.0       ],
            [ 36         , 3          , 0.00573    , 0.572958   , 0.028648   , 2.864789   , -27.0      , 27.0       ],
            [ 37         , 3          , 0.010186   , 1.018592   , 0.05093    , 5.092958   , -48.0      , 48.0       ],
            [ 38         , 3          , 0.009549   , 0.95493    , 0.047746   , 4.774648   , -45.0      , 45.0       ],
            [ 51         , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ],
            [ 52         , 3          , 0.005093   , 0.509296   , 0.025465   , 2.546479   , -24.0      , 24.0       ],
            [ 53         , 3          , 0.00191    , 0.190986   , 0.009549   , 0.95493    , -9.0       , 9.0        ]
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
