clc;clearvars;close all;
rng('shuffle')
%% parameter
nn=[10 50]; %num serie
TT=[50 100]; %lenght simul
Num_Sim=100; % num simul
PrcNan=[0.5 0.2 0.1]; %percentuale nan
parametri = combvec(nn,TT,PrcNan);

maxiter=50;% num iteration
% set parameters for classical nowcast
r=1; % number of factors
p=1; % lags in factor VAR for DFM
s=1; % lags in ido
modelSpec.r=r;
modelSpec.p=p;

% set sinkhorn param
M=.4;% size batches
learnRate = .001;
sqGradDecay = .95;
iterOt=100; % num iter sink
iterRSMpro=10; % num iter rms  update
init_epsilon=1;% entropy init param

%% main loop
for AA=1:size(parametri,2)
    pp=parametri(:,AA);
    n=pp(1);
    T=pp(2);
    prcNan=pp(3);
    simulazioniISExpandingFunzione(n,T,prcNan,maxiter,r,p,M,learnRate,sqGradDecay,iterOt,iterRSMpro,init_epsilon,Num_Sim,s)

           
end