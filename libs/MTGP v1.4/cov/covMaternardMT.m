function K = covMaternardMT(d, hyp, x, z, i)
% Based on the covMaterniso.m function of the GPML Toolbox - 
%   with the following changes:
%       - only elements of x(:,1:end-1)/z(:,1:end-1) will be analyzed, 
%       - x(:,end)/z(:,end) will be ignored, as it contains only the label information
%       - independent of the label all x values will have the same hyp

% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% Imperial College London, H2020 DICE project


if nargin<3, K = '(D)'; return; end              % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

[n,D] = size(x(:,1:end-1));
ell = exp(hyp(1:D));
sf2 = exp(2*hyp(D+1));
if all(d~=[1,3,5]), error('only 1, 3 and 5 allowed for d'), end         % degree

switch d
  case 1, f = @(t) 1;               df = @(t) 1./t;     % df(t) = (f(t)-f'(t))/t
  case 3, f = @(t) 1 + t;           df = @(t) 1;
  case 5, f = @(t) 1 + t.*(1+t/3);  df = @(t) (1+t)/3;
end
          m = @(t,f) f(t).*exp(-t); dm = @(t,f) df(t).*exp(-t);

% precompute distances
if dg                                                               % vector kxx
  K = zeros(size(x(:,1:end-1),1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = sq_dist(diag(sqrt(d)./ell)*x(:,1:end-1)');
  else                                                   % cross covariances Kxz
    K = sq_dist(diag(sqrt(d)./ell)*x(:,1:end-1)',diag(sqrt(d)./ell)*z(:,1:end-1)');
  end
end

if nargin<5                                                        % covariances
  K = sf2*m(sqrt(K),f);
else                                                               % derivatives
  if i<=D                                               % length scale parameter
    if dg
      Ki = zeros(size(x(:,1:end-1),1),1);
    else
      if xeqz
        Ki = sq_dist(sqrt(d)/ell(i)*x(:,i)');
      else
        Ki = sq_dist(sqrt(d)/ell(i)*x(:,i)',sqrt(d)/ell(i)*z(:,i)');
      end
    end
    K = sf2*dm(sqrt(K),f).*Ki;
    K(Ki<1e-12) = 0;                                    % fix limit case for d=1
  elseif i==D+1                                            % magnitude parameter
    K = 2*sf2*m(sqrt(K),f);
  else
    error('Unknown hyperparameter')
  end
end