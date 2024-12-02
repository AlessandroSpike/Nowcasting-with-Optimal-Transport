function KalmanSmootherOutput = kalman_smoother(KalmanFilterOutput)
    % model in state-space form:
    % X_{t}=C S_{t} + e_{t}
    % S_{t}=A S_{t-1} + u_{t}
    % e_{t}~N(0,R); u_{t}~N(0,Q); P_{t}=var(S_{t});
    
    % INPUT: (=kalman_filter output)
    % S_filter=S_{t|t} t=0,..,T [contains also initial value]
    % P_filter=P_{t|t} t=0,..,T [contains also initial value]
    % S_forecast=S_{t|t-1} t=1,...,T
    % P_forecast=P_{t|t-1} t=1,...,T
    % KL_struc=[Tx2]structure with intermediate output:
    %          K_{t}=P_{t|t-1}C'[CP_{t|t-1}C'+R]^{-1} t=1,...,T (gain on 1st dim)
    %          L_{t-1}=P_{t-1|t-1}A'[P_{t|t-1}]^{-1} t=0,...,T (on 2nd dim)
    % C,R,A,Q=system matrices
    
    % OUTPUT:
    % S_smooth=S_{t|T} t=1,...,T
    % P_smooth=P_{t|T}=E[(S_{t}-S_{t|T})(S_{t}-S_{t|T})']
    % PP_smooth=P_{t,t-1|T}=E[(S_{t}-S_{t|T})(S_{t-1}-S_{t-1|T})']
    % C,R,A,Q=system matrices

    %unpack structures
    SystemMatrices  =KalmanFilterOutput.SystemMatrices; 
    C               =SystemMatrices.CbigT; 
    A               =SystemMatrices.A; 
    
    S_filter        =KalmanFilterOutput.S_filter; 
    P_filter        =KalmanFilterOutput.P_filter; 
    S_forecast      =KalmanFilterOutput.S_forecast; 
    P_forecast      =KalmanFilterOutput.P_forecast;

    KL_struc       =KalmanFilterOutput.KL; 

    
    [nS,T]=size(S_filter); 
   
    S_smooth  =NaN(nS,T); 
    P_smooth  =cell(T,1); 
    PP_smooth =cell(T,1);
    
    K =KL_struc(:,1); L =KL_struc(:,2);
    
    % t=T
    S_smooth(:,T) =S_filter(:,T); 
    P_smooth{T}   =P_filter{T};
    PP_smooth{T}  =(eye(nS)-K{end}*C)*A*P_filter{T}; %Shumway(1988)
    
    % t<T
    for t=T-1:-1:1
    
        S_smooth(:,t) =S_filter(:,t)+L{t}*(S_smooth(:,t+1)-S_forecast(:,t));
        P_smooth{t}   =P_filter{t}-L{t}*(P_forecast{t}-P_smooth{t+1})*L{t}';
    
        if t>1 
            
            PP_smooth{t} =P_filter{t}*L{t-1}'+...
                L{t}*(PP_smooth{t+1}-A*P_filter{t})*L{t-1}'; %Shumway(1988)
        end
    end

    KalmanSmootherOutput.S_smooth       =S_smooth; 
    KalmanSmootherOutput.P_smooth       =P_smooth;
    KalmanSmootherOutput.PP_smooth      =PP_smooth;
end