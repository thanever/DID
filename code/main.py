
from dvid import Dvid
from os.path import dirname, join
from optimization_sin import OptSin
import time

def multi_dvid(casename):
    t1 = time.time()
    set_disturb =  [(1,1),(2,2),(3,1), (4,2)]  # [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2)]
    mode = 23
    opt_num = 'd_1_2_3_4'

    path_casedata = join(dirname(__file__), 'data//casedata//'+ casename +'.py')
    path_result = join(dirname(__file__), 'result//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')

    did = Dvid(path_casedata, set_disturb)
    did.dispatch(mode)
    did.save(path_result)
    t2 = time.time()

    print(t2-t1)

if __name__ == "__main__":
    t1 = time.time()
    set_disturb = [(1,1),(2,2),(3,1),(4,2)]#, (4,3)] # [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2)]
    casename = 'case14'
    mode =  22
    opt_num = 'd_1_2_3_4'

    path_casedata = join(dirname(__file__), 'data//casedata//'+ casename +'.py')
    path_result = join(dirname(__file__), 'result//' + 'result-' + casename + '-mode_' + str(mode) + '-opt_num_' + str(opt_num) +  '.p')

    did = Dvid(path_casedata, set_disturb)
    did.dispatch(mode)
    # did.save(path_result)
    t2 = time.time()

    print(t2-t1)





# if __name__ == "__main__": 
#     CASENAME  = ['case59_1',
#                 'case59_2',
#                 'case59_3',
#                 'case59_4',
#                 'case59_5',
#                 'case59_6']

#     for casename in CASENAME:
#         multi_dvid(casename)

'''
mode == 10: self.opt_sin()
mode == 11: self.opt_sin_admm()
mode == 20: self.opt_lin()
mode == 21: self.opt_lin_sdp_large()
mode == 22: self.opt_lin_sdp_small()
mode == 23: self.opt_lin_sdp_small_admm()
mode == 24: self.opt_lin_sdp_small_admm_fs()
mode == 30: self.opt_qq()
mode == 31: self.opt_qq_sdp_large()
mode == 32: self.opt_qq_sdp_small()
mode == 33: self.opt_qq_sdp_small_admm()
mode == 34: self.opt_qq_sdp_small_admm_fs()
'''

