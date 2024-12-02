function [LL,SystemMatrices_end,StateSpaceP_new]=funzioneNowcastClassic(X,r,p,nmaxit)

modelSpec.r=r;
modelSpec.p=p;
LL=nan(nmaxit,1);
% transform & standardize
[T,N]=size(X); 
mX=nanmean(X); vX=nanstd(X); x=bsxfun(@minus,X,mX); x=bsxfun(@rdivide,x,vX);

% initialize classic
[S_init,P_init,C_init,R_init,A_init,Q_init]=initialize_EM_Classic(x,r,p);

%load structure
SystemMatrices_init.S =S_init; 
SystemMatrices_init.P =P_init;
SystemMatrices_init.C =C_init; 
SystemMatrices_init.R =R_init;
SystemMatrices_init.A =A_init; 
SystemMatrices_init.Q =Q_init;


%% main 
it=0;

%remove leading and ending nans for the estimation
optNaN.method   =5;                
optNaN.k        =3;
y_est   =remNaNs_spline(x,optNaN);



while it<=nmaxit
 
   [SystemMatrices_end,loglklhd]=EMalgorithmClassico(y_est,modelSpec,SystemMatrices_init);
       
    %update matrices
    SystemMatrices_init.S =SystemMatrices_end.S;
    SystemMatrices_init.P =SystemMatrices_end.P;
    
    SystemMatrices_init.C =SystemMatrices_end.C;
    SystemMatrices_init.R =SystemMatrices_end.R;
    SystemMatrices_init.A =SystemMatrices_end.A;
    SystemMatrices_init.Q =SystemMatrices_end.Q;

    it=it+1;
    LL(it)=loglklhd;
end

% last run of kalman filter & smoother 
%@ iteration j states are estimated using system matrices @ iteration (j-1))

KalmanFilterOutput      =kalman_filter(x,SystemMatrices_end);


KalmanSmootherOutput    =kalman_smoother(KalmanFilterOutput);

%load results
StateSpaceP_new.S       =KalmanSmootherOutput.S_smooth(:,1); %S_{0|0}
StateSpaceP_new.P       =SystemMatrices_end.P; %P_{0|0}

StateSpaceP_new.C       =SystemMatrices_end.C;
StateSpaceP_new.R       =SystemMatrices_end.R;
StateSpaceP_new.A       =SystemMatrices_end.A;
StateSpaceP_new.Q       =SystemMatrices_end.Q;


StateSpaceP_new.States  =KalmanSmootherOutput.S_smooth(:,2:end)'; %states
StateSpaceP_new.Xhat    =((KalmanSmootherOutput.S_smooth(:,2:end)'*...
                         SystemMatrices_end.C').*kron(nanstd(x),ones(T,1)))+...
        kron(nanmean(x),ones(T,1));         %estimated X
StateSpaceP_new.mX      =mX; 
StateSpaceP_new.vX      =vX; 



