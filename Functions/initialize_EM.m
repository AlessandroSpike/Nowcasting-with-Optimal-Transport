function [S_init,P_init,C_init,R_init,A_init,Q_init]=initialize_EM(x,r,p,isSink)
    % initialize algo w\ principal components & LS
    %remove nans for initialization with principal components
    %replace with MA(k) after deleting rows with leading or ending nans
    if isSink==0
        optNaN.method   =5;                 
        optNaN.k        =3;
        [xSpline,~]     =remNaNs_spline(x,optNaN); 
    else
        xSpline=x;
    end
   

    vX    =cov(xSpline); 
    [V,D] =eigs(vX,r); 
    [T,N] =size(xSpline);
    F     =xSpline*V/sqrt(D); % principal components for factors
    
    % observation eqn
    tempC =V*sqrt(D);  % \Lambda (x=L*F' with r=N)
    e     =xSpline-F*tempC'; 
    tempR =cov(e);
    
    C_init        =zeros(N,r*p); 
    C_init(:,1:r) =tempC;
    R_init        =diag(diag(tempR)); %[NxN]
    
    % transition eqn
    % vector of current & lagged factors=[F_{t},F_{t-1},...,F_{t-p}]';
    lag_F =NaN(T-p,r*(p+1));
    
    for j=1:p+1
        lag_F(:,r*(j-1)+1:r*j)=F((p+1)-j+1:end-(j-1),:);
    end
    
    tempA =(lag_F(:,r+1:end)'*lag_F(:,r+1:end))\(lag_F(:,r+1:end)'*lag_F(:,1:r));
    u     =lag_F(:,1:r)-lag_F(:,r+1:end)*tempA; 
    tempQ =cov(u);
    
    A_init                    =zeros(r*p,r*p); 
    A_init(1:r,:)             =tempA'; 
    A_init(r+1:end,1:r*(p-1)) =eye(r*(p-1));
    
    Q_init                    =zeros(r*p,r*p);
    Q_init(1:r,1:r)           =tempQ;
    
    % states & variance
    S_init =zeros(r*p,1);
    vecP   =(eye((r*p)^2)-kron(A_init,A_init))\Q_init(:);
    P_init =reshape(vecP,r*p,r*p); %var(S_init)
end