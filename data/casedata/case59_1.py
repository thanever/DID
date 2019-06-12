# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for IEEE 14 bus test case.
"""

from numpy import array

def case59_1():
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
        [1       ,	3,	0,	0,	0,	0,	1,	1,	0,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [2       ,	1,	450,	45,	0,	0,	1,	1.03895,	-1.4021,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [3       ,	2,	0,	0,	0,	0,	2,	1,	48.9004,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [4       ,	2,	0,	0,	0,	0,	2,	1,	37.8999,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [5       ,	2,	0,	0,	0,	0,	2,	1,	32.241,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [6       ,	2,	0,	0,	0,	0,	2,	1,	39.211,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [7       ,	2,	390,	39+68.262,	0,	0,	2,	1.055,	44.0174,	330,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [8       ,	1,	130,	13,	0,	0,	2,	1.0472,	41.4466,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [9       ,	1,	1880,	188,	0,	0,	2,	1.0221,	27.6908,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [10      ,	1,	210,	21,	0,	0,	2,	1.03244,	26.2505,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [11      ,	1,	0,	0,	0,	0,	2,	1.03793,	30.4145,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [12      ,	1,	0,	0,	0,	0,	2,	1.05939,	26.6789,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [13      ,	1,	1700,	170,	0,	0,	2,	1.00806,	18.8693,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [14      ,	1,	1660,	166,	0,	400,	2,	1.00839,	19.1254,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [15      ,	1,	0,	0,	0,	0,	2,	1.04263,	22.6683,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [16      ,	1,	0,	0,	0,	0,	2,	1.02565,	18.7505,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [17      ,	1,	480,	48,	0,	0,	2,	1.04329,	33.114,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [18      ,	1,	1840,	184,	0,	300,	2,	1.01299,	16.0964,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [19      ,	1,	1260,	126,	0,	0,	2,	1.00207,	9.3905,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [20      ,	2,	0,	0,	0,	0,	3,	1,	-3.3932,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [21      ,	2,	0,	0,	0,	0,	3,	1,	-18.0204,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [22      ,	1,	0,	0,	0,	0,	3,	1.0417,	-10.8515,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [23      ,	1,	0,	0,	0,	0,	3,	1.00933,	-21.7495,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [24      ,	1,	0,	0,	0,	0,	3,	1.0164,	-22.7619,	500,	1,	1.1,	0.9,        0 ,      0 ],
        [25      ,	1,	1230,	123,	0,	0,	3,	1.01598,	-24.9219,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [26      ,	1,	650,	65,	0,	0,	3,	1.01823,	-24.9732,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [27      ,	1,	655,	66,	0,	0,	3,	1.03586,	-35.4136,	500,	1,	1.1,	0.9,        0 ,      0.01 ],
        [28      ,	1,	195,	20,	0,	0,	3,	1.02913,	-9.3431,	330,	1,	1.1,	0.9,        0 ,      0.01 ],
        [29      ,	1,	0,	0,	0,	0,	3,	1.01981,	-24.8071,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [30      ,	1,	0,	0,	0,	0,	3,	1.01358,	-17.6649,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [31      ,	1,	115,	12,	0,	0,	3,	1.0383,	-23.5946,	220,	1,	1.1,	0.9,        0 ,      0.01 ],
        [32      ,	2,	2405,	240-71.399,	0,	0,	3,	1.015,	-30.2295,	220,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [33      ,	1,	250,	25,	0,	0,	3,	1.01578,	-28.5167,	220,	1,	1.1,	0.9,        0 ,      0.01 ],
        [34      ,	1,	0,	0,	0,	0,	3,	1.03763,	-39.0132,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [35      ,	2,	0,	0,	0,	0,	4,	1,	74.3354,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [36      ,	2,	0,	0,	0,	0,	4,	1,	107.9391,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [37      ,	2,	0,	0,	0,	0,	4,	1,	113.3369,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [38      ,	2,	0,	0,	0,	0,	4,	1,	106.526,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [39      ,	1,	990,	99,	0,	0,	4,	1.03118,	99.5376,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [40      ,	1,	740,	74,	0,	0,	4,	1.03417,	104.7857,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [41      ,	1,	0,	0,	0,	0,	4,	1.03927,	107.1131,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [42      ,	1,	150,	15,	0,	0,	4,	1.03014,	100.3798,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [43      ,	1,	260,	26,	0,	60,	4,	0.98553,	76.613,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [44      ,	1,	530,	53,	0,	0,	4,	1.03012,	68.1448,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [45      ,	1,	575,	58,	0,	30,	4,	0.99305,	62.2465,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [46      ,	2,	1255,	126-58.162,	0,	0,	4,	1,	60.9735,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [47      ,	1,	0,	0,	0,	0,	4,	1.03981,	60.9511,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [48      ,	1,	0,	0,	0,	-30,	4,	1.04183,	60.1987,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [49      ,	1,	0,	0,	0,	-60,	4,	1.04738,	56.657,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [50      ,	1,	0,	0,	0,	-60,	4,	1.05524,	50.2682,	330,	1,	1.1,	0.9,        0 ,      0 ],
        [51      ,	2,	0,	0,	0,	0,	5,	1,	-55.315,	20,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [52      ,	2,	0,	0,	0,	0,	5,	1,	-53.9517,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [53      ,	2,	0,	0,	0,	0,	5,	1,	-56.283,	15,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [54      ,	1,	300,	60,	0,	0,	5,	1.04785,	-63.3086,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [55      ,	1,	0,	0,	0,	0,	5,	1.02206,	-60.8729,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [56      ,	1,	0,	0,	0,	0,	5,	1.02043,	-62.1823,	275,	1,	1.1,	0.9,        0 ,      0 ],
        [57      ,	2,	1000,	200-22.648,	0,	0,	5,	1.015,	-63.4538,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ],
        [58      ,	1,	800,	160,	0,	0,	5,	1.01043,	-64.6602,	275,	1,	1.1,	0.9,        0 ,      0.01 ],
        [59      ,	2,	200,	40-10.554,	0,	0,	5,	1.03,	-45.7446,	275,	1,	1.1,	0.9,        0.01 ,   0.01 ]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf  # p_g of 3 to 8 is all change from 0 to 40
    ppc["gen"] = array([
        [1 ,	300.808,	311.463,	581.128,	-581.128,	1,	1333.2,	1,	1200,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [3 ,	3600,	573.64,	1743.648,	-1743.648,	1,	4000.2,	1,	3600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [4 ,	2500,	663.381,	1210.9,	-1210.9,	1,	2778,	1,	2500,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [5 ,	1500,	531.183,	968.72,	-968.72,	1,	2222.4,	1,	2000,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [6 ,	2950.2,	734.155,	1743.648,	-1743.648,	1,	4000.2,	1,	3600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [20,	4200,	996.438,	2034.256,	-2034.256,	1,	4666.9,	1,	4200,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [21,	939.9,	154.576,	581.127,	-581.127,	1,	1333.2,	1,	1200,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [35,	1400,	514.88,	774.836,	-774.836,	1,	1777.6,	1,	1600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [36,	837,	177.852,	435.846,	-435.846,	1,	999.9,	1,	900,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [37,	1400,	209.191,	774.836,	-774.836,	1,	1777.6,	1,	1600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [38,	1549.8,	326.804,	871.692,	-871.692,	1,	1999.8,	1,	1800,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [51,	600,	50.693,	290.564,	-290.564,	1,	666.6,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [52,	800,	160.267,	600,	-600,	1,	1000,	1,	800,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
        [53,	436,	100.855,	290.652,	-290.652,	1,	666.8,	1,	600,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
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
        [1       , 2     ,	0,      0.009,	0,	0,	0,	0,	0.939,	0,	1,	-360,	360],
        [3       , 8     ,	0,      0.004,	0,	0,	0,	0,	0.9434,	0,	1,	-360,	360],
        [4       , 11    ,	0,      0.00576,	0,	0,	0,	0,	0.939,	0,	1,	-360,	360],
        [5       , 10    ,	0,      0.00765,	0,	0,	0,	0,	0.939,	0,	1,	-360,	360],
        [6       , 17    ,	0,      0.004,	0,	0,	0,	0,	0.939,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [11      , 12    ,	0,      0.0272,	0,	0,	0,	0,	0.9756,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [15      , 16    ,	0,      0.0272,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [20      , 22    ,	0,      0.00343,	0,	0,	0,	0,	0.939,	0,	1,	-360,	360],
        [21      , 31    ,	0,      0.01127,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [23      , 32    ,	0,      0.032,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [23      , 32    ,	0,      0.032,	0,	0,	0,	0,	0.9615,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 30    ,	0,      0.036,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 33    ,	0,      0.024,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [24      , 33    ,	0,      0.024,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [27      , 34    ,	0,      0.027,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [27      , 34    ,	0,      0.027,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [35      , 44    ,	0,      0.00845,	0,	0,	0,	0,	0.939,	0,	1,	-360,	360],
        [36      , 42    ,	0,      0.017,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [37      , 41    ,	0,      0.00845,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [38      , 39    ,	0,      0.0085,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [47      , 48    ,	0,      0.008,	0,	0,	0,	0,	1,	0,	1,	-360,	360],
        [51      , 54    ,	0,      0.0255,	0,	0,	0,	0,	0.9524,	0,	1,	-360,	360],
        [52      , 55    ,	0,      0.016,	0,	0,	0,	0,	0.9622,	0,	1,	-360,	360],
        [53      , 56    ,	0,      0.025,	0,	0,	0,	0,	0.9622,	0,	1,	-360,	360]
    ])

     ##----- Generator bus control data  -----##
    # type of generator or inverter, min_m, max_m, min_d, max_d, min_p (p.u.), max_p (p.u.)
	#  type of generator or inverter: 9 - conventional generator, 0 - inverter with no optimize, 1 - inverter optimizing d, 2 - inverter optimizing m, 3 - inverters optimizing d and m
    ppc["gencontrol"] = array([
            [ 1          , 3          , 0.007639   , 0.763944   , 0.038197   , 3.819719   , -36.0      , 36.0       ],
            [ 3          , 3          , 0.022918   , 2.291831   , 0.114592   , 11.459156  , -108.0     , 108.0      ],
            [ 4          , 3          , 0.015915   , 1.591549   , 0.079577   , 7.957747   , -75.0      , 75.0       ],
            [ 5          , 3          , 0.012732   , 1.27324    , 0.063662   , 6.366198   , -60.0      , 60.0       ],
            [ 6          , 3          , 0.022918   , 2.291831   , 0.114592   , 11.459156  , -108.0     , 108.0      ],
            [ 20         , 3          , 0.026738   , 2.673803   , 0.13369    , 13.369015  , -126.0     , 126.0      ],
            [ 21         , 3          , 0.007639   , 0.763944   , 0.038197   , 3.819719   , -36.0      , 36.0       ],
            [ 35         , 3          , 0.010186   , 1.018592   , 0.05093    , 5.092958   , -48.0      , 48.0       ],
            [ 36         , 3          , 0.00573    , 0.572958   , 0.028648   , 2.864789   , -27.0      , 27.0       ],
            [ 37         , 3          , 0.010186   , 1.018592   , 0.05093    , 5.092958   , -48.0      , 48.0       ],
            [ 38         , 3          , 0.011459   , 1.145916   , 0.057296   , 5.729578   , -54.0      , 54.0       ],
            [ 51         , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ],
            [ 52         , 3          , 0.005093   , 0.509296   , 0.025465   , 2.546479   , -24.0      , 24.0       ],
            [ 53         , 3          , 0.00382    , 0.381972   , 0.019099   , 1.909859   , -18.0      , 18.0       ]
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