function [SystemMatrices_end,loglklhd,x_init]=EMalgorithm(x,modelSpec,SystemMatrices_init,dove,isSink)
    % EM algo for DFM: use smoothed states and their variance to compute
    % sufficient statistics & model parameters
    
    [T,N]   =size(x);
    
    
    %unpack model specification
    r       =modelSpec.r;               %number of factors
    p       =modelSpec.p;               %number of lags in factor VAR
    
    % *     *     *     *   E X P E C T A T I O N   *     *     *     *

    KalmanFilterOutput          =kalman_filter(x,SystemMatrices_init);
    KalmanSmootherOutput        =kalman_smoother(KalmanFilterOutput);

    
    %smoothed states and their variance 
    S_smooth                    =KalmanSmootherOutput.S_smooth;
    P_smooth                    =KalmanSmootherOutput.P_smooth;
    PP_smooth                   =KalmanSmootherOutput.PP_smooth;
    
    
    %log likelihood
    loglklhd                    =KalmanFilterOutput.loglklhd;
    
   
    
    % sufficient statistics
    P   =zeros(r*p,r*p); 
    Pl  =zeros(r*p,r*p); 
    PPl =zeros(r*p,r*p);
    
    for t=1:T
        
        P   =P+P_smooth{t+1};    %sum(var(S_{t})
        Pl  =Pl+P_smooth{t};     %sum(var(S_{t-1})
        PPl =PPl+PP_smooth{t+1}; %sum(cov(S_{t}S_{t-1})
    end
    
    E_FF   =S_smooth(1:r,2:end)*S_smooth(1:r,2:end)'+P(1:r,1:r);        %E(F_{t}F_{t}')
    E_FlFl =S_smooth(1:r,1:end-1)*S_smooth(1:r,1:end-1)'+Pl(1:r,1:r);   %E(F_{t-1}F_{t-1}')
    E_FFl  =S_smooth(1:r,2:end)*S_smooth(1:r,1:end-1)'+PPl(1:r,1:r);    %E(F_{t}F_{t-1}')
    
    % update model parameters
    A_end =zeros(r*p,r*p); A_end(r+1:end,1:r*(p-1))=eye(r*(p-1));
    Q_end =zeros(r*p,r*p); 
    C_end =zeros(N,r*p);
    
    tempA =E_FFl/E_FlFl;
    tempQ =(1/T)*(E_FF-tempA*E_FFl');

    MC1     =zeros(N*r,N*r); 
    MC2     =zeros(N,r);
   

    for t=1:T            
        %track missing values
        Wx =x(t,:)';     W =diag(~isnan(Wx));     Wx(isnan(Wx)) =0;
        MC1 =MC1 + kron(S_smooth(1:r,t+1)*S_smooth(1:r,t+1)'+...
                 P_smooth{t+1}(1:r,1:r),W);
        %
            MC2 =MC2 + Wx*S_smooth(1:r,t+1)';
        
    end
    %update C matrix for monthly variables
    vecC    =MC1\MC2(:); 
    tempC   =reshape(vecC,N,r);
  

    Wx2 =x;     Wx2(isnan(Wx2)) =0;
    ee    =(Wx2'-tempC*S_smooth(1:r,2:end))*(Wx2'-tempC*S_smooth(1:r,2:end))';
    tempR =(1/T)*(ee+tempC*P(1:r,1:r)*tempC'); 
    
    A_end(1:r,1:r) =tempA; 
    Q_end(1:r,1:r) =tempQ; 
    C_end(:,1:r)   =tempC; 
    R_end          =diag(diag(tempR));
   
    P_end=P;

    %load structure with results
    SystemMatrices_end.S =S_smooth(:,1); 
    SystemMatrices_end.P =P_end;
    SystemMatrices_end.C =C_end; 
    SystemMatrices_end.R =R_end;
    SystemMatrices_end.A =A_end; 
    SystemMatrices_end.Q =Q_end;


    %x_state per sinkhorn
    if isSink==1
        x_aus= (S_smooth(:,2:end)'*C_end');        %estimated X
        x_init=x;
        x_init(dove)=x_aus(dove);
    else
        x_init=x;
    end
end
