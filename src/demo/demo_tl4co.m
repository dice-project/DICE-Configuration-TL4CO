% The demo file for TL4CO
% Authors: Pooyan Jamshidi (pooyan.jamshidi@gmail.com)
% The code is released under the FreeBSD License.
% Copyright (C) 2016 Pooyan Jamshidi, Imperial College London


%% import m-files
%import bo4co.*
close all;
clear variables;
clc;
warning off;

global istestfun;
init;

expData = []; % init matrix saving experimental data
expElapsed=[];
expEntropy=[];

% multi-task initialization
T = 2;    % Number of tasks
irank = T;    % rank for Kf (1, ... M). irank=M -> Full rank

Fmotion = @(x) alpha(max(min(x,1),0)); % for transparent fill and fun!

%% this is the analytical function we want to find it's minimum
% domain represents the domain of the function
if istestfun
    [f, domain, trueMinLoc] = testFunctionFactory('branin'); %f11 e4
else
    [f, domain, trueMinLoc] = testConfigurationFactory('cass20-109'); %f11 e4
end
d = size(domain, 1); % dimension of the space

%% create the grid
[xTest, xTestDiff, nTest, nTestPerDim] = makeGrid(domain, nMinGridPoints);
%[xTest, xTestDiff, nTest, nTestPerDim] = makeDGrid(domain);
n=size(xTest,1);

%% initialize the prior
gps = covMTKernelFactory(6,T,d);

%% observations from other tasks
for t=2:T
    switch t
        case 2
            g=@(x) f(x)+randn*f(x);
        case 3
            g=@(x) f(x)+rand*f(x);
        case 4
            g=@(x) f(x)+rand^3*f(x);
%         case 4
%             g=@(x) f(x)+0.3*f(x);
    end
    v=randperm(n);
    idx_train = v(1:maxIter);
    X=xTest(idx_train,:);
    observedX{t}=X;
    for k=1:length(X)
        Y(k,:)=g(X(k,:));
    end
    observedY{t}=Y;
end

%% initial samples from the Latin Hypercube design
% (this section should be genralized in a way to replace any DoE that gives initial design such as random, etc...)
nInit = 3*d;
maxIter=maxIter-nInit;
% this loop is only for replicating optimization loop to get a mean and CI for each optimization criteria
for exp=1:maxExp
    fprintf('experiment number %d is running now!!!',exp);
    %obsX = lhsdesign(d, nInit)';
    obsX = lhsdesign4grid(d, nInit, domain);
    %obsX=xTest(v(1:nInit),:);
    %obsX = unirnddesign4grid(d, nInit, domain);
    obsY = zeros(size(obsX, 1), 1);
    
    for k = 1:size(obsX, 1)
        obsY(k) = f(obsX(k, :));
    end
    
    tElapsed=[];
    tEntropy=[];
    %% Bayesian optimization loop (for locating minimizer)
    for k = 1:maxIter
        % criterial to evaluate in order to find where to sample next
        
        observedX{1} = obsX;
        observedY{1} = obsY;
        [nextX, gpslearned, xTest, m, s, z, ef, h, et] = mtLCB(domain, observedX, observedY, gps);
        
        tEntropy = [tEntropy;h];     
        tElapsed = [tElapsed;et];        
        ms(:,k)=m(:,1); % saving mean prediction for saving the evolution of gp models over time
        mst2(:,k+(k-1)*(T-2):k+k*(T-2))=m(:,2:T); 
        
        % evaluate at the suggested point
        nextY = f(nextX);
        
        % save the measurement pair and CIs
        obsX = [obsX; nextX];
        obsY = [obsY; nextY];
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % compute resulting K_f matrix
        vec_dim = [1:T];
        L = zeros(T,T);
        for cnt_dim = 1:T
            L(cnt_dim,1:vec_dim(cnt_dim)) = [gpslearned.hyp.cov(sum(vec_dim(1:cnt_dim-1))+1:sum(vec_dim(1:cnt_dim-1))+vec_dim(cnt_dim))];
        end
        K_f =  L*L';
        
        % MTGP correlation coefficients - not normalized
        MTGP_cc(k,1) = K_f(2,1);
        %MTGP_cc(k,2) = K_f(3,1);
        %MTGP_cc(k,3) = K_f(3,2);
        est_hyp(k,:) = gpslearned.hyp.cov(1:sum(1:t));
        
        % normalization of K_f matrix
        [a Kc_n]= normalize_Kc(est_hyp(k,:),T);
        % print results on console:
        disp(['Estimated cross correlation covariance Kc_n:']);
        Kc_n
        
        % MTGP correlation coefficients - normalized
        MTGP_cc_n(k,1) = Kc_n(2,1);
        %MTGP_cc_n(count,2) = Kc_n(3,1);
        %MTGP_cc_n(count,3) = Kc_n(3,2);
        
        % Pearsons correlation coefficient of the output function
        a = corrcoef(m(:,1),m(:,2));
        Pear_cc_output(k,1) = a(2);
        %a = corrcoef(results.m(:,1),results.m(:,3));
        %Pear_cc_output(count,2) = a(2);
        %a = corrcoef(results.m(:,2),results.m(:,3));
        %Pear_cc_output(count,3) = a(2);
                
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % visualize and update
        if visualize>0
            if d==1
                % visualize true function 1D
                %m=m(:,1);
                %s=s(:,1);
                obsX=observedX{1};
                obsY=observedY{1};

                y=f(xTest);
                h1=figure(1);
                plot(xTest,y,'LineWidth',2);
                hold on; plot(trueMinLoc,f(trueMinLoc),'+');
                % observations
                fu = [m+2*s; flip(m-2*s,1)];
                
                for kk=2:10
                    fu1(:,kk)=[m+2*s/(10-kk+1); flip(m+2*s/(10-kk+2),1)];
                    fu1(:,kk+10)=[m-2*s/(10-kk+2); flip(m-2*s/(10-kk+1),1)];
                end
                fu1(:,1)=[m+2*s/10; flip(m,1)];
                fu1(:,11)=[m; flip(m-2*s/10,1)];
                
                for kk=1:10
                    fu2(:,kk)=[m+2*kk*s/10; flip(m+2*(kk-1)*s/10,1)];
                    fu2(:,kk+10)=[m-2*(kk-1)*s/10; flip(m-2*kk*s/10,1)];
                end
                
                for kk=1:10
                    hold on; h2=fill([xTest; flip(xTest,1)], fu2(:,kk), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk)); % [7 7 7]/8
                    h3=fill([xTest; flip(xTest,1)], fu2(:,kk+10), [7 7 7]/8,'FaceColor','r','FaceAlpha',10/(10*kk));
                    %set(gcf,'windowbuttonmotionfcn','Fmotion( ([1 0]*get(gca,''currentp'')*[0;1;0] - min(ylim)) / diff(ylim) )');
                    set(h2,'Linestyle','none')
                    set(h3,'Linestyle','none')
                end
                
                
                hold on; plot(xTest, m); plot(obsX, obsY, '*');
                % current estimate
                hold on; plot(nextX, nextY,'o');
                saveas(gcf,strcat('gp-',num2str(k),'.fig'));
                
                %h3=figure(2); plot(z); % if you use boTS
                %disp('paused');
                %pause(5);% pause for a while
                close(h1);
            end
            
            if d==2
                h1=figure(1);                
                plot(obsX(:,1),obsX(:,2), 'r+');
                for idx=1:size(obsX,1)
                text(obsX(idx,1)+0.2,obsX(idx,2)+0.2, num2str(idx),...
                    'FontWeight', 'bold',...
                    'FontSize',8,...
                    'HorizontalAlignment','center');
                end
                hold on; plot(trueMinLoc(:,1),trueMinLoc(:,2),'d');
                hold on; plot(nextX(:,1),nextX(:,2),'o');
                hold on; contour(unique(xTest(:,1)),unique(xTest(:,2)),reshape(f(xTest),size(unique(xTest(:,1)),1),size(unique(xTest(:,2)),1)));
                set(h1,'ShowText','on','TextStep',get(h,'LevelStep'))
                % true function 2D
                %h2=figure(2);surf(unique(xTest(:,1)),unique(xTest(:,2)),reshape(y,size(unique(xTest(:,1)),1),size(unique(xTest(:,2)),1)));
                h2=figure(2);surfc(unique(xTest(:,1)),unique(xTest(:,2)),reshape(m,size(unique(xTest(:,1)),1),size(unique(xTest(:,2)),1)));
                shading interp
                disp('paused');
                pause(5); % pause for a while
                close(h1);
                close(h2);
            end
        end
        
    end
    
    % saving the replication data
    
    expData=[expData obsX obsY];
    expElapsed=[expElapsed tElapsed];
    expEntropy=[expEntropy tEntropy];
    
    % Pearsons correlation coefficient of the training data
    %a = corrcoef(observedY{1},observedY{2});
    %Pear_cc_input(exp,1) = a(2);
    %a = corrcoef(y1,y3);
    %Pear_cc_input(count,2) = a(2);
    %a = corrcoef(y2,y3);
    %Pear_cc_input(count,3) = a(2);
    
end
%% report what has been found
[mv, mloc] = min(obsY);
fprintf('Minimum value: %f found at:\n', mv);
disp(obsX(mloc, :));
fprintf('True minimum value: %f at:\n', f(trueMinLoc));
disp(trueMinLoc);