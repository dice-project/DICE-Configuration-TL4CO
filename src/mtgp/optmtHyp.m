function [gps, fv1] = optmtHyp(gps, x, y)
% Evidence optimization routine for hyperparamter selection.
% Try previous hyper-parameter as well as a grid of hyper-parameters
% Input
%   gps: GP structure from previous step (used as initial state for the update)
%	gps.hyp: hyper parameter structure
%	gps.meanfunc: mean function
%	gps.covfunc: covariance function
%	gps.likfunc: likelihood function
%   x: (N x d) N observed x
%   y: (N x 1) corresponding y to x
%
% Output
%   gps: optimized parameter set
%   fv1: negative log evidence over iterations of conjugate gradient
%
% Later we may want multiple covariances (TODO)

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
% Copyright (C) 2016 Pooyan Jamshidi, Imperial College London

opt.init_num_opt = 100;
opt.num_rep=1;

% optimize hyperparameters
for cnt_rep = 1:opt.num_rep
    disp(['Number of rep: ',num2str(cnt_rep)]);

    % optimize hyperparameter
    [results.hyp{cnt_rep}] = minimize(gps.hyp, @MTGP, -opt.init_num_opt, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y);

    % training
    results.nlml(cnt_rep) = MTGP(results.hyp{cnt_rep}, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y);
end
% find best  nlml
[results.nlml, best_hyp] =min(results.nlml);
gps.hyp = results.hyp{best_hyp};
fv1=results.nlml;