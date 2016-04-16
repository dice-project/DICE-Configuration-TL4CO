function [nextX, gps, xTest, m, s, z, ef, h, et] = mtLCB(xRange, observedX, observedY, gps)
% this is the acquision function for multi-task GP

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
% Copyright (C) 2016 Pooyan Jamshidi, Imperial College London

%import bo4co.*

global istestfun nMinGridPoints;

%% kappa function
kappaf=@(s,r,e,t) sqrt(2*log(s*zeta(r)*t.^2/e));

%% process input
d = size(xRange, 1); % input space dimension
N = size(observedX{1}, 1);
t= size(observedX,2); % number of tasks
assert(d == size(observedX{1}, 2));
assert(N == numel(observedY{1})); %observedY = observedY(:);

%% Parameters of the algorithm
nMinGridPoints = 1e5;
isSmoothGrid = true;
kappa=1; %default value

%% generate training and test data with labels 

if istestfun
    [xTest, xTestDiff, nTest, nTestPerDim] = makeGrid(xRange, nMinGridPoints);
else
    [xTest, xTestDiff, nTest, nTestPerDim] = makeDGrid(xRange);
end

nCandidateSample = nTest; % candidate x_{n+1} to be drawn at every step

x_test = [xTest ones(size(xTest,1),1)];

x = [observedX{1}, ones(size(observedX{1},1),1)];
y = [observedY{1}];

for i_task = 2:t
    x = [x; observedX{i_task}, ones(size(observedX{i_task},1),1)*i_task];
    y = [y; observedY{i_task}];
    
    x_test=[x_test; xTest ones(size(xTest,1),1)*i_task];
end

tic;
%% Bayesian model selection to find hyperparameter 
gps = optmtHyp(gps, x, y);

%% Perform GP on the test grid (calculate posterior)
[m, s2, fmu, fs2] = MTGP(gps.hyp, @MTGP_infExact, gps.meanfunc, gps.covfunc, gps.likfunc, x, y, x_test);


% reshape of results
m = reshape(m,[size(xTest,1) t]);
s2 = reshape(s2,[size(xTest,1) t]);
s = sqrt(s2);

mt1=m(:,1);
st1=s(:,1);

% compute the entropy of the minimizer distribution
h=0;
%h=compute_entropy(mt1,st1);
%% Estimate the current min f
[z, zi] = min(mt1);

%% Estimate LCB (UCB)
%ef = m+kappa*s; % for UCB

% adapt kappa
kappa=kappaf(length(xTest),2,0.001,length(observedY{1}));

ef = mt1-kappa*st1;
%idxs = arrayfun(@(x)find(xTest==x,1),observedX);
[sharedVals,idxsIntoA] = intersect(xTest,observedX{1},'rows');
[dummy, mefi] = min(ef(setdiff(1:end,idxsIntoA)));
idxNext = mefi;

xTestnotObserved=xTest(setdiff(1:end,idxsIntoA),:);
nextX = xTestnotObserved(idxNext, :);

et=toc;

end