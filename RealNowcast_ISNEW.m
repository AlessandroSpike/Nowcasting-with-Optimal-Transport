clc;clearvars; close all
rng(1,'philox')
%% addpath dati
addpath("Data\")
%% load data
load DatiUSGiannone
Xtot1=X;
Xgdp=Xtot1(:,3);
Xgdp2=Xtot1(:,21);
Xtot1=[Xtot1(:,[1:2,4:20,22:end]),Xgdp2,Xgdp];
%% tolgo o no recessiono
Covid=0;
if Covid==0
    Xtot1(421:432,:)=[]; % periodo covid da togliere
    Dati(421:432,:)=[]; % periodo covid da togliere
    fine=450;
    lung=34*7;
else
    fine=462;
    lung=38*7;
end
%% EM param
r=1; % number of factors
p=1; % lags in factor VAR for DFM
nM=23; %number of monthly variables
nQ=2; %number of quarterly variables
%loadings restrictions (restrictions are written as R1*LAMBDA=R2)
R1          =[2 -1 0 0 0; 3 0 -1 0 0; 2 0 0 -1 0; 1 0 0 0 -1];
R2          =zeros(4,1);
%detect GDP releases
isgdprelease=zeros(nM+nQ,1);
isgdprelease=logical(isgdprelease);
modelSpec.isGDPrelease =isgdprelease;
blocks=ones(nM+nQ,1); % common factors
modelSpec.r=r;
modelSpec.p=p;
modelSpec.blocks =blocks;
modelSpec.nM=nM;
modelSpec.nQ=nQ;
modelSpec.R1=R1;
modelSpec.R2=R2;
thresh =1E-3; %for likelihood convergence
nmaxit =20; %max number of iterations
%% sinkhorn nowcast
% set sinkhorn param
M1=.4;% size batches
learnRate = .001;
sqGradDecay = .95;
iterOt=200; % num iter sink
iterRSMpro=10; % num iter rms  update
init_epsilon=1;
%% container
NowcastNewOT=nan(lung,1);
NowcastNewIS=nan(lung,1);
NowcastNewAR=nan(lung,1);
Contatore=nan(lung,1);
TarNew=nan(lung,1);

%% main
ii=1;
for t=351:3:fine
    %% transform & standardize
    Target=Xtot1(t,end);
    for i=1:7
        tic
        if i==1
           XNew=Xtot1(73:t+1,:);
           XNew(end-1,end)=nan;
           XNew(end-1,end-1)=nan;
        elseif i==2
           XNew=Xtot1(73:t,:);
           XNew(end,end)=nan;
           XNew(end,end-1)=nan;
        else
           XNew=Xtot1(73:t,:);
           XNew(end-i+3:end,:)=nan;
        end
        mX=nanmean(XNew); vX=nanstd(XNew);
        XNew=bsxfun(@minus,XNew,mX); XNew=bsxfun(@rdivide,XNew,vX);
        %% new part
        % initialize sink
        x=XNew;
        T=size(XNew,1);
        dove=isnan(x);
        optNaN.method   =1;                 
        optNaN.k        =3;
        media=remNaNs_spline(XNew,optNaN);
        x(dove)=media(dove);
        % initialize classic
        [S_init,P_init,C_init,R_init,A_init,Q_init]=initialize_EM_MQ(x,modelSpec,1);    
        %load structure
        SystemMatrices_init.S =S_init; 
        SystemMatrices_init.P =P_init;
        SystemMatrices_init.C =C_init; 
        SystemMatrices_init.R =R_init;
        SystemMatrices_init.A =A_init; 
        SystemMatrices_init.Q =Q_init;


        S=nan(38,nmaxit);
        P=nan(38,38,nmaxit);
        C=nan(25,38,nmaxit);
        R=nan(25,25,nmaxit);
        Q=nan(38,38,nmaxit);
        A=nan(38,38,nmaxit);
        % main 
        it=1; llconverged=0; loglklhd_old=0;
        x_init=x;
        while it<=nmaxit && llconverged==0             
            x_end=nan(size(x_init));       
            xaus=x_init;
            M=ceil(M1*size(xaus,1));
            rifaccio=1;
            epsilon=init_epsilon;
            while rifaccio==1   
                [x_endaus,S_divergence]=OT_imputerCONDITIONAL(xaus,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon);
                if sum(sum(isnan(x_endaus)))>0
                   epsilon=epsilon+0.1;
                   disp('occhio')
                else
                   x_end=x_endaus;
                   rifaccio=0;
               end
            end            
           [SystemMatrices_end,loglklhd,x_init]=EMalgorithm_MQ(x_end,modelSpec,SystemMatrices_init,dove,1); 
            %update matrices
            S(:,it)=SystemMatrices_end.S;
            P(:,:,it)=SystemMatrices_end.P;
            C(:,:,it)=SystemMatrices_end.C;
            R(:,:,it)=SystemMatrices_end.R;
            Q(:,:,it)=SystemMatrices_end.Q;
            A(:,:,it)=SystemMatrices_end.A;
       

            SystemMatrices_init.S =nanmean(S,2);
            SystemMatrices_init.P =nanmean(P,3);
    
            SystemMatrices_init.C =nanmean(C,3);
            SystemMatrices_init.R =nanmean(R,3);
            SystemMatrices_init.A =nanmean(A,3);
            SystemMatrices_init.Q =nanmean(Q,3);
            %check likelihood convergence
            llconverged=check_convergence(loglklhd,loglklhd_old,thresh);
            loglklhd_old =loglklhd;   
            it=it+1;
        end
        % last run of kalman filter & smoother 
        x_end=nan(size(x));   
        xaus=x;
        M=ceil(M1*size(xaus,1));
        rifaccio=1;
        epsilon=init_epsilon;
        while rifaccio==1   
            [x_endaus,S_divergence]=OT_imputer(xaus,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon);            
            if sum(sum(isnan(x_endaus)))>0
           epsilon=epsilon+0.1;
               disp('occhio')
            else
                x_end=x_endaus;
               rifaccio=0;
           end
        end
        KalmanFilterOutputSink      =kalman_filter(x_end,SystemMatrices_end);
        KalmanSmootherOutputSink    =kalman_smoother(KalmanFilterOutputSink);
        
        %load results
        StateSpaceP_newSink.S       =KalmanSmootherOutputSink.S_smooth(:,1); %S_{0|0}
        StateSpaceP_newSink.P       =SystemMatrices_end.P; %P_{0|0}
        
        StateSpaceP_newSink.C       =SystemMatrices_end.C;
        StateSpaceP_newSink.R       =SystemMatrices_end.R;
        StateSpaceP_newSink.A       =SystemMatrices_end.A;
        StateSpaceP_newSink.Q       =SystemMatrices_end.Q;
        
        
        StateSpaceP_newSink.States  =KalmanSmootherOutputSink.S_smooth'; %states
        StateSpaceP_newSink.Xhat    =((KalmanSmootherOutputSink.S_smooth(1:5,2:end)'*...
                                 SystemMatrices_end.C(:,1:5)').*kron(vX,ones(T,1)))+kron(mX,ones(T,1)); %estimated 

       toc
        %% classic
        x=XNew;
        T=size(XNew,1);
        dove=isnan(x);
        % initialize classic
        [S_init,P_init,C_init,R_init,A_init,Q_init]=initialize_EM_MQ(x,modelSpec,0);    
        %load structure
        SystemMatrices_init.S =S_init; 
        SystemMatrices_init.P =P_init;
        SystemMatrices_init.C =C_init; 
        SystemMatrices_init.R =R_init;
        SystemMatrices_init.A =A_init; 
        SystemMatrices_init.Q =Q_init;
        % main 
        it=0; llconverged=0; loglklhd_old=0;
        %remove leading and ending nans for the estimation
        optNaN.method   =1;                
        optNaN.k        =3;
        y_est   =remNaNs_spline(x,optNaN);
        x_end = y_est;


        while it<=nmaxit && llconverged==0
           [SystemMatrices_end,loglklhd,x_init]=EMalgorithm_MQ(x_end,modelSpec,SystemMatrices_init,dove,0);
               
            %update matrices
            SystemMatrices_init.S =SystemMatrices_end.S;
            SystemMatrices_init.P =SystemMatrices_end.P;
            
            SystemMatrices_init.C =SystemMatrices_end.C;
            SystemMatrices_init.R =SystemMatrices_end.R;
            SystemMatrices_init.A =SystemMatrices_end.A;
            SystemMatrices_init.Q =SystemMatrices_end.Q;
            %check likelihood convergence
            llconverged=check_convergence(loglklhd,loglklhd_old,thresh);
            loglklhd_old =loglklhd ;   
            it=it+1;
        end
        % last run of kalman filter & smoother 
        KalmanFilterOutput      =kalman_filter(XNew,SystemMatrices_end);
        KalmanSmootherOutput    =kalman_smoother(KalmanFilterOutput);
        
        %load results
        StateSpaceP_new.S       =KalmanSmootherOutput.S_smooth(:,1); %S_{0|0}
        StateSpaceP_new.P       =SystemMatrices_end.P; %P_{0|0}
        
        StateSpaceP_new.C       =SystemMatrices_end.C;
        StateSpaceP_new.R       =SystemMatrices_end.R;
        StateSpaceP_new.A       =SystemMatrices_end.A;
        StateSpaceP_new.Q       =SystemMatrices_end.Q;
        
    
        StateSpaceP_new.States  =KalmanSmootherOutput.S_smooth(:,2:end)'; %states
        StateSpaceP_new.Xhat    =((KalmanSmootherOutput.S_smooth(1:5,2:end)'*...
                             SystemMatrices_end.C(:,1:5)').*kron(vX,ones(T,1)))+kron(mX,ones(T,1)); %estimated 
        %% ar classico
        x=XNew;
        aicCont=zeros(12,1);
        for u=1:12
            Mdl = arima(u,0,0);
            EstMdl = estimate(Mdl,x(1:end-1,end),'Display','off');
            results = summarize(EstMdl);
            aicCont(u)=results.AIC;
        end
        [minAic,posAic]=min(aicCont);
        Mdl = arima(posAic,0,0);
        EstMdl = estimate(Mdl,x(1:end-1,end),'Display','off');
        ForeAR=forecast(EstMdl,1);

        %% colleziono
        TarNew(ii)=Target;
        if i>1
            NowcastNewOT(ii)=StateSpaceP_newSink.Xhat(end,end);
            NowcastNewIS(ii)=StateSpaceP_new.Xhat(end,end);
            NowcastNewAR(ii)=ForeAR(end,end);
        else
            NowcastNewOT(ii)=StateSpaceP_newSink.Xhat(end-1,end);
            NowcastNewIS(ii)=StateSpaceP_new.Xhat(end-1,end);
            NowcastNewAR(ii)=ForeAR(end,end);
        end

        disp([datestr(Dati.Date(t)),' OT: ', num2str(sqrt(mean((TarNew(1:ii)-NowcastNewOT(1:ii)).^2))),' DFM: ',...
                num2str(sqrt(mean((TarNew(1:ii)-NowcastNewIS(1:ii)).^2))),' AR: '...
                num2str(sqrt(mean((TarNew(1:ii)-NowcastNewAR(1:ii)).^2)))])

        Contatore(ii)=i;
        ii=ii+1;
    end
end
%% save
clc
filename=(['AfterJBES_NEW_RisRealiIS_Covid',num2str(Covid),'.mat']);
save(filename,"TarNew",'NowcastNewIS','NowcastNewOT','NowcastNewAR','Contatore')