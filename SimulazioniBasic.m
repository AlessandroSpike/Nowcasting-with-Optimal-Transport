clc;clearvars;close all;
rng('default')
%% parameter
nn=[50]; %num serie
TT=[100]; %lenght simul
Num_Sim=500; % num simul
PrcNan=[0.5 0.2 0.1]; %percentuale nan
maxiter=20;% num iteration
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
for N=1:length(nn)
    n=nn(N);
    for tt=1:length(TT)
        T=TT(tt);
        for prnan=1:length(PrcNan)
            prcNan=PrcNan(prnan);
            %% container
            LL_DFM=cell(Num_Sim,1);
            LL_OT=cell(Num_Sim,1);
            A_DFM_RMSE=nan(Num_Sim,1);
            A_OT_RMSE=nan(Num_Sim,1);
            L_DFM_RMSE=nan(Num_Sim,1);
            L_OT_RMSE=nan(Num_Sim,1);
            X_DFM_RMSE=nan(Num_Sim,1);
            X_OT_RMSE=nan(Num_Sim,1);
            F_DFM_R2=nan(Num_Sim,1);
            F_OT_R2=nan(Num_Sim,1);
            %%  main loop
            for i=1:Num_Sim
                missingVal=randi(n*T,n*T*prcNan,1);
                
                f_0=randn(r,1);
                A=rand(1)*eye(r,r);
                Lambda=randn(n,r);
                
                X=zeros(n,T);
                F=zeros(r,T);
                for t=1:T
                    e=randn(n,1);
                    u=randn(r,1);
                    if t==1
                        F(:,t)=f_0;
                    else
                        F(:,t)=A*F(:,t-1) +u;
                    end
                    X(:,t)=Lambda*F(:,t) + e;  
                end
                X=X';
                X1=X;
                X(missingVal)=nan;
                disp(i)
                disp('Dati Creati Fatto')
               
                [LL,SystemMatrices_end,StateSpaceP_new,SystemMatrices_end_Collect]=funzioneNowcast(X,r,p,maxiter,0,M1,learnRate,...
                sqGradDecay,iterOt,iterRSMpro,init_epsilon);
                disp('Classico Fatto')

                tic
                [LL_Sink,SystemMatrices_end_Sink,StateSpaceP_new_Sink,SystemMatrices_end_Collect_Sink]=funzioneNowcast(X,r,p,maxiter,1,M1,learnRate,...
                sqGradDecay,iterOt,iterRSMpro,init_epsilon);
                disp('OT Fatto')
                toc
            
                LL_DFM{i}=SystemMatrices_end_Collect_Sink;
                LL_OT{i}=SystemMatrices_end_Collect;
            
                A_DFM_RMSE(i) = sqrt((A-SystemMatrices_end.A).^2);
                A_OT_RMSE(i) = sqrt((A-SystemMatrices_end_Sink.A).^2);
            
                L_DFM_RMSE(i) = sqrt(mean((Lambda-SystemMatrices_end.C).^2));
                L_OT_RMSE(i) = sqrt(mean((Lambda-SystemMatrices_end_Sink.C).^2));

                X_DFM_RMSE(i) = sqrt(mean(X1(isnan(X))-StateSpaceP_new.Xhat(isnan(X))).^2);
                X_OT_RMSE(i) = sqrt(mean(X1(isnan(X))-StateSpaceP_new_Sink.Xhat(isnan(X))).^2);
                
                Fx_dfm=StateSpaceP_new.States;
                Fx_ot=StateSpaceP_new_Sink.States;
                F=F';
                F_DFM_R2(i)=trace((F'*Fx_dfm)*((Fx_dfm'*Fx_dfm)^-1)*(Fx_dfm'*F))/trace(F'*F);
                F_OT_R2(i)=trace((F'*Fx_ot)*((Fx_ot'*Fx_ot)^-1)*(Fx_ot'*F))/trace(F'*F);
            
                disp(['RMSE A dfm: ',num2str(mean(A_DFM_RMSE(1:i))),' -- A ot: ',num2str(mean(A_OT_RMSE(1:i)))])
                disp(['RMSE L dfm: ',num2str(mean(L_DFM_RMSE(1:i))),' -- L ot: ',num2str(mean(L_OT_RMSE(1:i)))])
                disp(['RMSE X dfm: ',num2str(mean(X_DFM_RMSE(1:i))),' -- X ot: ',num2str(mean(X_OT_RMSE(1:i)))])
                disp(['R2 F dfm: ',num2str(mean(F_DFM_R2(1:i))),' -- F ot: ',num2str(mean(F_OT_R2(1:i)))])
            
            end
            clc;
            filename=(['AfterJBES_RisSimul_NumVar',num2str(n),'_LungSerie',num2str(T),'_PercNan',num2str(10*prcNan),'_r',num2str(r),'_p',num2str(p),'.mat']);
            save(filename,"LL_OT","LL_DFM","A_DFM_RMSE","A_OT_RMSE","L_DFM_RMSE","L_OT_RMSE","F_OT_R2","F_DFM_R2","X_DFM_RMSE","X_OT_RMSE")
        end
    end
end
