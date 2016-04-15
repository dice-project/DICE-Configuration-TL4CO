function [gps, desc] = covMTKernelFactory(setting, t, d)
% Set of commonly used covariance kernels and their hyperparameters for
% multi-task GPs
%
% Input
%   setting: numerical selection of options
%   t: number of tasks
%   d: input dimension

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
% Copyright (C) 2016 Pooyan Jamshidi, Imperial College London


%import bo4co.*

maxOption = 7;
if nargin < 1
    gps = maxOption;
    return
end

% init values for hyperparameter
gps.per_hyp = 6;
gps.ell = 1;
gps.sf=1;
gps.cc_hyp = [1 0 1 0 0 1 0 0 0 1];     % assumes that all signals are independent
gps.sn = 0.1;
gps.noise_hyp = 0.001;
num_cc_hyp = sum(1:t);

% mean function
gps.meanfunc = @meanConst;
gps.hyp.mean =0;
%gps.meanfunc = {@meanSum, {@meanLinear, @meanConst}}; %hyp.mean = [0.5; 1];
%gps.meanfunc = {@meanPoly,2};

%gps.hyp.mean = [ones(d+1,1); 0];

switch setting
    case 1
        desc='Covariance Function: K = CC(l) x (SE(t))';
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covSEiso'}};
        gps.hyp.cov(1:num_cc_hyp) = gps.cc_hyp(1:num_cc_hyp);
        gps.hyp.cov(num_cc_hyp+1) = log(sqrt(gps.ell));
        gps.hyp.cov(num_cc_hyp+2) = log(sqrt(gps.sf));
    case 2
        desc='Covariance Function: K = CC(l) x (Per_UU(t)*SE_U(t)) + Noise(t)';
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','MTGP_covPeriodicisoUU','MTGP_covSEisoU'}};
        covfunc = {'MTGP_covSum',{covfunc,'MTGP_covNoise'}};
        gps.hyp.cov(1:num_cc_hyp) = gps.cc_hyp(1:num_cc_hyp);
        gps.hyp.cov(num_cc_hyp+1) = log(gps.per_hyp);
        gps.hyp.cov(num_cc_hyp+2) = log(sqrt(gps.ell));
        gps.hyp.cov(num_cc_hyp+3:num_cc_hyp+2+t) = log(sqrt(gps.noise_hyp));
    case 3
        desc='Covariance Function: K = CC(l) x (Matern(t))';
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'MTGP_covMaterniso',5}}};
        gps.hyp.cov(1:num_cc_hyp) = gps.cc_hyp(1:num_cc_hyp);
        gps.hyp.cov(num_cc_hyp+1) = log(sqrt(gps.ell));
        gps.hyp.cov(num_cc_hyp+2) = log(sqrt(gps.sf));
    case 4
        desc='Covariance Function: K = CC(l) x (covCat(t))';
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','covCategorical'}};
        gps.hyp.cov(1:num_cc_hyp) = gps.cc_hyp(1:num_cc_hyp);
        gps.hyp.cov(num_cc_hyp+1) = log(sqrt(gps.ell));
        gps.hyp.cov(num_cc_hyp+2) = log(sqrt(gps.sf));
    case 5
        desc='Covariance Function: K = CC(l) x (SEard(t))';
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD','covSEardMT'}};
        gps.hyp.cov(1:num_cc_hyp) = gps.cc_hyp(1:num_cc_hyp);
        gps.hyp.cov(num_cc_hyp+1:num_cc_hyp+d) = log(sqrt(gps.ell*ones(d,1)));
        gps.hyp.cov(num_cc_hyp+1+d)= log(sqrt(gps.sf));
     case 6
        desc='Covariance Function: K = CC(l) x (Maternard(t))';
        covfunc = {'MTGP_covProd',{'MTGP_covCC_chol_nD',{'covMaternardMT',5}}};
        gps.hyp.cov(1:num_cc_hyp) = gps.cc_hyp(1:num_cc_hyp);
        gps.hyp.cov(num_cc_hyp+1:num_cc_hyp+d) = log(gps.ell*ones(d,1));
        gps.hyp.cov(num_cc_hyp+1+d)= log(gps.sf);
end

gps.covfunc = covfunc;

% likelihood function
gps.likfunc = @likGauss;
gps.hyp.lik = log(gps.sn);
