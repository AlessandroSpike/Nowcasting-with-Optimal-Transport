clc;clearvars;close all
rng('default')
%% parameter
nn=[10 50]; %num serie
TT=[50 100]; %lenght simul
Num_Sim=100; % num simul
PrcNan=[1 2 3 4]; %percentuale nan
parametri = combvec(nn,TT,PrcNan);

maxiter=50;% num iteration
% set parameters for classical nowcast
r=1; % number of factors
p=1; % lags in factor VAR for DFM
modelSpec.r=r;
modelSpec.p=p;

% set sinkhorn param
M1=.4;% size batches
learnRate = .001;
sqGradDecay = .95;
iterOt=100; % num iter sink
iterRSMpro=10; % num iter rms  update
init_epsilon=1;% entropy init param

%% main loop
parfor AA=1:size(parametri,2)
    pp=parametri(:,AA);
    n=pp(1);
    T=pp(2);
    prcNan1=pp(3);

    simulazioniBasicExpandingFunzione(n,T,prcNan1,maxiter,r,p,M1,learnRate,sqGradDecay,iterOt,iterRSMpro,init_epsilon,Num_Sim)

end
