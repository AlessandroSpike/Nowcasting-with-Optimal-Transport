function [KalmanFilterOutput,SystemMatrices] = kalman_filter(x,SystemMatrices)
    % model in state-space form:
    % X_{t}=C S_{t} + e_{t}
    % S_{t}=A S_{t-1} + u_{t}
    % e_{t}~N(0,R); u_{t}~N(0,Q); P_{t}=var(S_{t});
    
    % INPUT: 
    % S_init=vector of states S_{0|0}
    % P_init=variance of the states P_{0|0}
    % C,R,A,Q=system matrices
    
    % OUTPUT:
    % S_filter=S_{t|t} t=0,..,T [contains also initial value]
    % P_filter=P_{t|t} t=0,..,T [contains also initial value]
    % S_forecast=S_{t|t-1} t=1,...,T
    % P_forecast=P_{t|t-1} t=1,...,T
    % KL_struc=[Tx2]structure with intermediate output:
    %          K_{t}=P_{t|t-1}C'[CP_{t|t-1}C'+R]^{-1} t=1,...,T (gain on 1st dim)
    %          L_{t-1}=P_{t-1|t-1}A'[P_{t|t-1}]^{-1} t=1,...,T (on 2nd dim)
    % C,R,A,Q=system matrices
    % loglklhd=log likelihood

    S_init  =SystemMatrices.S; 
    P_init  =SystemMatrices.P;
    
    C       =SystemMatrices.C; 
    R       =SystemMatrices.R;
    A       =SystemMatrices.A; 
    Q       =SystemMatrices.Q;
    
    T=size(x,1); nS=length(S_init); %number of states
    
    S_filter=NaN(nS,T+1); S_filter(:,1)=S_init;
    P_filter=cell(T+1,1); P_filter{1}=P_init;
    S_forecast=NaN(nS,T); P_forecast=cell(T,1); KL_struc=cell(T,2);
    
    loglklhd=0;
    
    for t=1:T
    
        % prediction S_{t|t-1} & P_{t|t-1}
        S_forecast(:,t)=A*S_filter(:,t);
        P_forecast{t}=A*P_filter{t}*A'+Q;
        P_forecast{t}   =.5*(P_forecast{t}+P_forecast{t}');

        %remove rows with missing data
        keeprows        =~isnan(x(t,:)'); 
        
        xt              =x(t,keeprows)'; 
        Ct              =C(keeprows,:); 
        Rt              =R(keeprows,keeprows);
        
        B=Ct*P_forecast{t}; % B_{t|t-1}=E[(X_{t}-X_{t|t-1})(S_{t}-S_{t|t-1})']
        H=Ct*P_forecast{t}*Ct'+Rt; % H_{t|t-1}=E[(X_{t}-X_{t|t-1})(X_{t}-X_{t|t-1})']
        
        KL_struc{t,1}=B'/H; % gain
        KL_struc{t,2}=P_filter{t}*A'*pinv(P_forecast{t}); % needed for smoother
        
        % loglikelihood @ each t
        ll=log(det(inv(H)))-(xt-Ct*S_forecast(:,t))'/H*(xt-Ct*S_forecast(:,t)); 
        loglklhd=loglklhd+(1/2)*ll;
        
        % update S_{t|t} & P_{t|t}
        S_filter(:,t+1)=S_forecast(:,t)+KL_struc{t,1}*(xt-Ct*S_forecast(:,t));
        P_filter{t+1}=P_forecast{t}-KL_struc{t,1}*B;
        P_filter{t+1}   =.5*(P_filter{t+1}+P_filter{t+1}');
    end
    SystemMatrices.CbigT=Ct;

    %load structure
    KalmanFilterOutput.S_filter         =S_filter; 
    KalmanFilterOutput.P_filter         =P_filter; 
    KalmanFilterOutput.S_forecast       =S_forecast; 
    KalmanFilterOutput.P_forecast       =P_forecast;
    
    KalmanFilterOutput.SystemMatrices   =SystemMatrices;
    
    KalmanFilterOutput.KL                =KL_struc; 
    
    KalmanFilterOutput.loglklhd         =loglklhd;


end