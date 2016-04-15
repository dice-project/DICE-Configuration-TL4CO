function K = covCategoricalMT(hyp, x, z,i)
% Covariance for categorical variables

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
% Copyright (C) 2016 Pooyan Jamshidi, Imperial College London


if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end

dg = strcmp(z,'diag');
xeqz = isempty(z);                       % determine mode

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

if xeqz
    z = x;
end                                % make sure we have a valid z

if dg                                                               % vector kxx
    k_pre = zeros(size(x(:,1:end-1),1),1);
else
    if size(z(:,1:end-1), 1) == 1
        k_pre = sum((abs(bsxfun(@minus, x(:,1:end-1), z(:,1:end-1))) > 0), 2);
    else
        dim = size(x(:,1:end-1), 2);
        k_pre = zeros(size(x(:,1:end-1), 1), size(z(:,1:end-1), 1));
        for d = 1:dim            
            k_pre = k_pre + bsxfun(@ne, x(:, d), z(:, d)');
        end
    end
end

if nargin<4                                                        % covariances
  K = sf2*exp(-k_pre/ell);
else                                                               % derivatives
  if i==1
    K = sf2*exp(-k_pre/ell).*k_pre;
  elseif i==2
    K = 2*sf2*exp(-k_pre/ell);
  else
    error('Unknown hyperparameter')
  end
end
end