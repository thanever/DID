rawfile_name = 'LF_Case06_R4_S.raw';
mpc_name = 'LF_Case06_R4_S.m';
mpc = psse2mpc(rawfile_name, mpc_name);
%fname_out = savecase(fname, varargin)

runpf(mpc_name)