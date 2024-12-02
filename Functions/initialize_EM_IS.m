function [S_init,P_init,C_init,R_init,A_init,Q_init] = initialize_EM_IS(x,modelSpec,isSink)

r       =modelSpec.r; %number of factors
p       =modelSpec.p; %number of lags in factor VAR
s       =modelSpec.s; %number of lags in idio dynamic

nM      =modelSpec.nM; 


%
blocks  =modelSpec.blocks; %blocks restrictions
nB      =size(blocks,2);




%remove nans for initialization with principal components
%replace with MA(k) after deleting rows with leading or ending nans
if isSink==0
    optNaN.method   =5;                 
    optNaN.k        =3;
    [xSpline,~]     =remNaNs_spline(x,optNaN); 
else
    xSpline=x;
end


[T,N]           =size(xSpline);



nSf             =sum(r.*p); 
nSiM            =nM*s; 
nS              =nSf+nSiM;     %number of states

C_init          =zeros(N,nS); 
A_init          =zeros(nS,nS); 
Q_init          =zeros(nS,nS); 



% initialize state space w\ principal components & least squares
tempX           =xSpline;           %data without nans

tempF           =NaN(T,nSf); 


nSfi            =cumsum([1 r.*p]);

for b=1:nB
    
    % *~~~*~~~*~~~*~~~*~~~* MEASUREMENT EQUATION *~~~*~~~*~~~*~~~*~~~* 
    rb      =r(b); %# of factors per block
    nlF     =p-1; %# of effective lags
    
    btemp   =find(blocks(:,b)); 
    MinB    =btemp(btemp<=nM); 
    

    %factors are initialized on the monthly variables
    vX      =cov(tempX(:,MinB)); 
    [V,~]   =eigs(vX,rb);
    F       =tempX(:,MinB)*V; 
    F       =F(:,1:rb); %principal components for factors
    
    
    lagF_Meqn=nan(T-nlF,rb*(nlF+1)); %[F_{t},F_{t-1},...,F_{t-h}]';
    for j=1:nlF+1
        
        lagF_Meqn(:,rb*(j-1)+1:rb*j)=F((nlF+1)-j+1:end-(j-1),:);
    end
    tempC =lagF_Meqn(:,1:rb)\tempX(nlF+1:end,MinB); %loadingsM

    C_init(MinB,nSfi(b):nSfi(b)+rb-1) =tempC';
    %

    
    %orthogonalize for next block extraction
    tempF(:,nSfi(b):nSfi(b)+rb*(nlF+1)-1)=[zeros(size(tempX,1)-size(lagF_Meqn,1),rb*(nlF+1));lagF_Meqn];
    
    %`data' orthogonal to factors in preceding block
    tempX =tempX-tempF(:,nSfi(b):nSfi(b)+rb*(nlF+1)-1)*C_init(:,nSfi(b):nSfi(b)+rb*(nlF+1)-1)';

    
    
    % *~~~*~~~*~~~*~~~*~~~*~~~* STATE EQUATION *~~~*~~~*~~~*~~~*~~~*~~~* 
    lagF_Seqn =nan(T-(p+1),rb*(p+1)); %[F_{t},F_{t-1},...,F_{t-h}]';
    for j=1:p+1

        lagF_Seqn(:,rb*(j-1)+1:rb*j) =F(((p+1)+1)-j:end-j,:);
    end    
    
    
%     lagF_Seqn   =lagF_Meqn(:,1:rb*(p+1)); %[F_{t},F_{t-1},...,F_{t-p}]';
    tempA       =lagF_Seqn(:,rb+1:end)\lagF_Seqn(:,1:rb);
    u           =lagF_Seqn(:,1:rb)-lagF_Seqn(:,rb+1:end)*tempA; 
    tempQ       =cov(u);
    
    %
    blockA                      =zeros(rb*(nlF+1),rb*(nlF+1));
    blockA(1:rb,1:rb*p)         =tempA'; 
    blockA(rb+1:end,1:end-rb)   =eye(rb*nlF);
    
    A_init(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = blockA;
    
    
    blockQ                      =zeros(rb*(nlF+1),rb*(nlF+1));
    blockQ(1:rb,1:rb)           =tempQ;
    
    Q_init(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = blockQ;
end


% *~~~*~~~*~~~*~~~*~~~* MEASUREMENT EQUATION *~~~*~~~*~~~*~~~*~~~*
C_init(1:nM,nSf+1:nSf+nSiM)=kron(eye(nM),[1 zeros(1,s-1)]);



R_init=diag(1e-04.*ones(N,1)); %can be improved collecting the states



% *~~~*~~~*~~~*~~~*~~~*~~~* STATE EQUATION *~~~*~~~*~~~*~~~*~~~*~~~* 
E=tempX-tempF*C_init(:,1:nSf)'; %orthogonal to all the factors

%idioM
for i=1:nM
    
    iE =E(:,i); 
    
    lagE_Seqn=nan(T-s,s+1); %[e1_{t},...,e1_{t-s},...,enM_{t},...,enM_{t-s}]';
    for j=1:s+1
        
        lagE_Seqn(:,j)=iE((s+1)-j+1:end-(j-1),:);
    end
    
    tempA   =(lagE_Seqn(:,2:end)\lagE_Seqn(:,1));
    uM      =lagE_Seqn(:,1)-lagE_Seqn(:,2:end)*tempA; 
    tempQ   =cov(uM);
    
    %
    blockA                  =zeros(s,s); 
    blockA(1,:)             =tempA'; 
    blockA(2:end,1:end-1)   =eye(s-1);
    
    A_init(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s) = blockA;
    
    
    blockQ                  =zeros(s,s); 
    blockQ(1,1)             =tempQ; 
    
    Q_init(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s) = blockQ;
end



%-------------------------------------------------------------------------%
    
%states & variance
S_init  =zeros(nS,1);
vecP    =(eye(nS^2)-kron(A_init,A_init))\Q_init(:);
P_init  =reshape(vecP,nS,nS); %var(S_init)



% -----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*----- %
