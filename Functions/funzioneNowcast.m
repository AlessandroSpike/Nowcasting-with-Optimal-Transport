function [LL,SystemMatrices_end,StateSpaceP_new,SystemMatrices_end_Collect]=funzioneNowcast(X,r,p,nmaxit,isSink,M1,learnRate,...
    sqGradDecay,iterOt,iterRSMpro,init_epsilon)

SystemMatrices_end_Collect=cell(1,6);

S=nan(1,nmaxit);
P=nan(1,nmaxit);
C=nan(size(X,2),nmaxit);
R=nan(size(X,2),size(X,2),nmaxit);
Q=nan(1,nmaxit);
A=nan(1,nmaxit);


modelSpec.r=r;
modelSpec.p=p;
M=min(round(M1*size(X,1)),20);
LL=nan(nmaxit,1);
% transform & standardize
[T,N]=size(X); 
mX=nanmean(X); vX=nanstd(X); x=bsxfun(@minus,X,mX); x=bsxfun(@rdivide,x,vX);
% initialize sink
dove=isnan(x);
media=.1*randn(size(x))+repmat(nanmean(x),size(x,1),1);
if isSink==1
    x(dove)=media(dove);
end

% initialize classic
[S_init,P_init,C_init,R_init,A_init,Q_init]=initialize_EM(x,r,p,isSink);

%load structure
SystemMatrices_init.S =S_init; 
SystemMatrices_init.P =P_init;
SystemMatrices_init.C =C_init; 
SystemMatrices_init.R =R_init;
SystemMatrices_init.A =A_init; 
SystemMatrices_init.Q =Q_init;


%% main 
it=1;

%remove leading and ending nans for the estimation
optNaN.method   =5;                
optNaN.k        =3;
y_est   =remNaNs_spline(x,optNaN);
if isSink==1
    x_init = y_est;
else
    x_end = y_est;
end

while it<=nmaxit
   if isSink==1
        rifaccio=1;
        epsilon=init_epsilon;
        while rifaccio==1   
            [x_end,divergence]=OT_imputerCONDITIONAL(x_init,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon);
            if sum(sum(isnan(x_end)))>0
               epsilon=epsilon+0.1;
               disp('occhio')
            else
               rifaccio=0;
            end
        end
   end
   [SystemMatrices_end,loglklhd,x_init]=EMalgorithm(x_end,modelSpec,SystemMatrices_init,dove,isSink);
    S(it)=SystemMatrices_end.S;
    P(it)=SystemMatrices_end.P;
    C(:,it)=SystemMatrices_end.C;
    R(:,:,it)=SystemMatrices_end.R;
    Q(it)=SystemMatrices_end.Q;
    A(it)=SystemMatrices_end.A;
    
%    update matrices
    if isSink==1
        SystemMatrices_init.S =nanmean(S,2);
        SystemMatrices_init.P =nanmean(P,2);

        SystemMatrices_init.C =nanmean(C,2);
        SystemMatrices_init.R =nanmean(R,3);
        SystemMatrices_init.A =nanmean(A,2);
        SystemMatrices_init.Q =nanmean(Q,2);
    else
         SystemMatrices_init.S= SystemMatrices_end.S;
        SystemMatrices_init.P =SystemMatrices_end.P;
        
        SystemMatrices_init.C =SystemMatrices_end.C;
        SystemMatrices_init.R =SystemMatrices_end.R;
        SystemMatrices_init.A =SystemMatrices_end.A;
        SystemMatrices_init.Q =SystemMatrices_end.Q;
    end


    
    LL(it)=loglklhd;
    it=it+1;
end
SystemMatrices_end_Collect{1}=S;
SystemMatrices_end_Collect{2}=P;
SystemMatrices_end_Collect{3}=C;
SystemMatrices_end_Collect{4}=R;
SystemMatrices_end_Collect{5}=A;
SystemMatrices_end_Collect{6}=Q;
% last run of kalman filter & smoother 
%@ iteration j states are estimated using system matrices @ iteration (j-1))
if isSink==1
    rifaccio=1;
    epsilon=init_epsilon;
    while rifaccio==1   
        [x_end,divergence]=OT_imputerCONDITIONAL(x_init,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon);
        if sum(sum(isnan(x_end)))>0
           epsilon=epsilon+0.1;
           disp('occhio')
        else
           rifaccio=0;
        end
    end
    KalmanFilterOutput      =kalman_filter(x_end,SystemMatrices_init);
else
    KalmanFilterOutput      =kalman_filter(x,SystemMatrices_end);
end

KalmanSmootherOutput    =kalman_smoother(KalmanFilterOutput);

%load results
StateSpaceP_new.S       =KalmanSmootherOutput.S_smooth(:,1); %S_{0|0}
StateSpaceP_new.P       =SystemMatrices_end.P; %P_{0|0}

StateSpaceP_new.C       =SystemMatrices_end.C;
StateSpaceP_new.R       =SystemMatrices_end.R;
StateSpaceP_new.A       =SystemMatrices_end.A;
StateSpaceP_new.Q       =SystemMatrices_end.Q;


StateSpaceP_new.States  =KalmanSmootherOutput.S_smooth(:,2:end)'; %states
if isSink==1
    StateSpaceP_new.Xhat    =((KalmanSmootherOutput.S_smooth(:,2:end)'*...
                         SystemMatrices_end.C').*kron(nanstd(x_end),ones(T,1)))+...
        kron(nanmean(x_end),ones(T,1));         %estimated X
    % StateSpaceP_new.Xhat   =x_end;

else

    StateSpaceP_new.Xhat    =((KalmanSmootherOutput.S_smooth(:,2:end)'*...
                         SystemMatrices_end.C').*kron(nanstd(x),ones(T,1)))+...
        kron(nanmean(x),ones(T,1));         %estimated X
end
StateSpaceP_new.mX      =mX; 
StateSpaceP_new.vX      =vX; 



