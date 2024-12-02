function [S_init,P_init,C_init,R_init,A_init,Q_init] = initialize_EM_MQ(x,modelSpec,isSink)

%returns initial values for system matrices C R A Q and states' variance P


%NOTE: this is assuming that the variables in x are ordered such that the
%monthly ones come first, then all the quarterly which are not GDP releases,
%then the consecutive GPD releases


n       =size(x,2);      


%unpack model specification
r       =modelSpec.r;               %number of factors
p       =modelSpec.p;               %number of lags in factor VAR


blocks  =modelSpec.blocks;          %blocks restrictions


%restrictions are written as R1*C = R2
R1      =modelSpec.R1;              %matrix for loading restrictions(1) 
R2      =modelSpec.R2;              %matrix for loading restrictions(2)
Rpoly   =[1 R1(:,1)'];


q       =size(R1,1);                %number of lags in M-Q map


nM      =modelSpec.nM;              %number of monthly variables
nQ      =modelSpec.nQ;              %number of quarterly variables
nB      =size(blocks,2);            %number of blocks







%initialize state space
nSf     =sum(r.*(max(p,q)+1));      %number of states (factors & own lags)
nSm     =nM;                        %number of states (idio M)
nSq     =nQ*(q+1);                  %number of states (idio Q & own lags)

nS      =nSf+nSm+nSq;               %total number of states


%coefficients matrices
C_init  =zeros(n,nS);           
A_init  =zeros(nS,nS); 
Q_init  =zeros(nS,nS); 
P_init  =zeros(nS,nS);




%remove nans for initialization with principal components
%replace with MA(k) after deleting rows with leading or ending nans
if isSink==0
    optNaN.method   =1;                 
    optNaN.k        =3;
    [xSpline,iNaN]     =remNaNs_spline(x,optNaN); 
else
    xSpline=x;
    iNaN=[];
end


xNaN            =xSpline;
xNaN(iNaN)      =nan;               %contains original nan pattern 

T               =size(xNaN,1);      %number of available data points



% initialize state space w\ principal components & least squares

tempX           =xSpline;           %data without nans
tempXnan        =xNaN;              %data with original nans pattern

tempFactors     =nan(T,nSf);

%NOTE: the only procedure that uses the spline data is the principal
%components extraction. C R A Q only use actual data points


nSfi     =cumsum([1 r.*(max(p,q)+1)]); %location of factors in C A Q
for b=1:nB
    
  
    
    % * ~ ~ * ~ ~ *  M E A S U R E M E N T   E Q U A T I O N  * ~ ~ * ~ ~ *

    % * ~ ~ * ~ ~ *             F  A  C  T  O  R  S           * ~ ~ * ~ ~ *
    
    
    
    
    rb      =r(b);                  %number of factors per block
    ib      =nSfi(b);               %location of factors in C A Q per block
    k       =max(p,q);              %number of effective factors lags
    

    tempBlk =find(blocks(:,b)); 
    MinB    =tempBlk(tempBlk<=nM);  %monthly variables in block b
    QinB    =tempBlk(tempBlk>nM);   %quarterly variables in block b
    
    

    %factors are initialized on the monthly variables
    if isempty(MinB)
    
        error('There are no monthly variables in the chosen block')
    end
        
    vX      =cov(tempX(:,MinB)); 
    [V,~]   =eigs(vX,rb); 
    F       =tempX(:,MinB)*V; 
    F       =F(:,1:rb);             %principal components for factors

    
    
    %factors in measurement eqn
    lagF_Meqn =nan(T-q,rb*(q+1)); %[F_{t},F_{t-1},...,F_{t-h}]';
    for j=0:q

        lagF_Meqn(:,rb*j+1:rb*(j+1)) =F((q+1)-j:end-j,:);
    end
    
    
    for im =1:length(MinB)
    
        xM      =tempXnan(q+1:end,MinB(im)); 
        fM      =lagF_Meqn(~isnan(xM),1:rb); 
        
        xM      =xM(~isnan(xM),:);
        
        
        %monthly loadings
        tempC   =fM\xM; 

        %initialize C
        C_init( MinB(im), ib:ib+rb-1 ) = tempC';    
    end
    
    
    %quarterly loadings (with restrictions)
    R1B   =kron(R1,eye(rb)); 
    R2B   =kron(R2,ones(rb,1));
    
    for iq =1:length(QinB)
        
        
        xQ      =tempXnan(q+1:end,QinB(iq)); 
        fQ      =lagF_Meqn(~isnan(xQ),:); 
        
        xQ      =xQ(~isnan(xQ),:);
        

        %quarterly loadings (unrestricted)
        tempC   =fQ\xQ;
        
        %apply restrictions (restricted least squares)
        tempC   =tempC - ( fQ(:,1:rb*(q+1))'*fQ(:,1:rb*(q+1)) )\R1B'/...
                         ( R1B / (fQ(:,1:rb*(q+1))' * fQ(:,1:rb*(q+1))) * R1B' )*( R1B*tempC - R2B );


        %initialize C
        C_init( QinB(iq), ib:ib+rb*(q+1)-1 ) = tempC';
        
    end
    
    
    
    %orthogonalize for next block extraction
    tempF          =[zeros( q,rb*(q+1) ); lagF_Meqn];
    
    
    %`data' orthogonal to factors in preceding block
    tempX          =tempX - tempF*C_init(:,ib:ib+rb*(q+1)-1)';
    
    
    %`data' orthogonal to factors in preceding block with original nan pattern
    tempXnan       =tempX; 
    tempXnan(iNaN) =NaN;
    
    
    %store
    tempFactors(:,ib:ib+rb*(q+1)-1) =tempF;
    
    
    
    % * ~ ~ * ~ ~ * ~ ~ *  S T A T E   E Q U A T I O N  * ~ ~ * ~ ~ * ~ ~ *
    
    % * ~ ~ * ~ ~ * ~ ~ *     F  A  C  T  O  R  S       * ~ ~ * ~ ~ * ~ ~ *
    
    
    
    
    %factors VAR
    lagF_Seqn =nan(T-p,rb*p); %[F_{t},F_{t-1},...,F_{t-h}]';
    for j=1:p

        lagF_Seqn(:,rb*(j-1)+1:rb*j) =F((p+1)-j:end-j,:);
    end    
    
    tempA       =[ones(T-p,1) lagF_Seqn]\F(p+1:end,:);
    
    u           =F(p+1:end,:)-[ones(T-p,1) lagF_Seqn]*tempA; 
    tempQ       =cov(u);
    
    
    
    %initialize A for each block
    blockA                      =zeros(rb*(k+1),rb*(k+1));
    blockA(1:rb,1:rb*p)         =tempA(2:end,:)'; 
    blockA(rb+1:end,1:end-rb)   =eye(rb*k);
    
    %initialize A (factors)
    A_init( ib:ib+rb*(k+1)-1 , ib:ib+rb*(k+1)-1 ) = blockA;
    
    
    
    
    %initialize Q for each block
    blockQ                      =zeros(rb*(k+1),rb*(k+1));
    blockQ(1:rb,1:rb)           =tempQ;
    
    %initialize Q (factors)
    Q_init( ib:ib+rb*(k+1)-1 , ib:ib+rb*(k+1)-1 ) = blockQ;
    

    
    %initialize P (factors)
    blockP                      =( eye((rb*(k+1))^2) - kron(blockA,blockA) )\blockQ(:);
    
    P_init( ib:ib+rb*(k+1)-1 , ib:ib+rb*(k+1)-1 ) = reshape(blockP,rb*(k+1),rb*(k+1));

end






% * ~ ~ * ~ ~ *  M E A S U R E M E N T   E Q U A T I O N  * ~ ~ * ~ ~ *

% * ~ ~ * ~ ~ *         I D I O S Y N C R A T I C         * ~ ~ * ~ ~ *


%initialize C (idiosyncratic)
C_init(1:nM,nSf+1:nSf+nSm)      =eye(nM);
C_init(nM+1:nM+nQ,nSf+nSm+1:nS) =kron(eye(nQ),Rpoly);


%initialize R (small)
R_init                          =diag(1e-04.*ones(n,1)); 







% * ~ ~ * ~ ~ * ~ ~ *  S T A T E   E Q U A T I O N  * ~ ~ * ~ ~ * ~ ~ *

% * ~ ~ * ~ ~ * ~ ~ *   I D I O S Y N C R A T I C   * ~ ~ * ~ ~ * ~ ~ *


E =tempX-tempFactors*C_init(:,1:nSf)';  %orthogonal to all the factors
             
%NOTE: this only works if variables enter the X matrix in the following
%order: first all the monthly, then all the quarterly which are not GDP,
%then all the GDP releases

%idioM
for i=1:nM
    
    idioM       =E(:,i);
    idioM       =idioM(~isnan(idioM));
    
    tempA       =[ones(numel(idioM)-1,1) idioM(1:end-1)]\idioM(2:end);   %AR(1)
    
    uM          =idioM(2:end)-[ones(numel(idioM)-1,1) idioM(1:end-1)]*tempA; 
    tempQ       =cov(uM);
    
    
    %initialize A (idio M)
    A_init(nSf+i,nSf+i) =tempA(2);

    
    %initialize Q (idio M)
    Q_init(nSf+i,nSf+i) =tempQ;
        

    %initialize P (idio M)
    P_init(nSf+i,nSf+i) =(1-tempA(2)^2)\tempQ;
end




%idioQ
for i=nM+1:n
    
    %quarterly other than GDP have idio ~ AR(1)
    if ~ismember(i,find(modelSpec.isGDPrelease)) 
        
        %find relevant idio Q
        idioQ       =E(:,i);
        idioQ       =idioQ(~isnan(idioQ));

        
        %position of idio Q in states' vector
        iQ_starts   =nSf+nSm+1+(i-nM-1)*(q+1);
        iQ_ends     =nSf+nSm+(i-nM)*(q+1);

        
        
        %relevant coefficients
        tempA       =[ones(numel(idioQ)-1,1) idioQ(1:end-1)]\idioQ(2:end); %AR(1)

        uQ          =idioQ(2:end)-[ones(numel(idioQ)-1,1) idioQ(1:end-1)]*tempA; 
        tempQ       =cov(uQ);

        

        %initialize A (idio Q)
        blockA                  =zeros((q+1),(q+1)); 
        blockA(1,1)             =tempA(2)'; 
        blockA(2:end,1:end-1)   =eye(q);

        A_init(iQ_starts:iQ_ends,iQ_starts:iQ_ends) =blockA;



        %initialize Q (idio Q)
        blockQ                  =zeros((q+1),(q+1)); 
        blockQ(1,1)             =tempQ; 

        Q_init(iQ_starts:iQ_ends,iQ_starts:iQ_ends) =blockQ;
        

        %initialize P (idio Q)
        tempP                   =(eye((q+1)^2) - kron(blockA,blockA))\blockQ(:);
        
        P_init(iQ_starts:iQ_ends,iQ_starts:iQ_ends) =reshape(tempP,q+1,q+1);

    end
end


%idio for GDP releases ~ VAR(1)
idioG       =E(:,modelSpec.isGDPrelease); 
idioG       =idioG(all(~isnan(idioG')),:);


nG          =size(idioG,2);                     %number of GDP releases in VAR


%position of idio G in states' vector
iG_starts   =nSf+nSm+(nQ-nG)*(q+1)+1;
iG_ends     =nSf+nSm+nSq;


tempA       =[ones(size(idioG,1)-1,1) idioG(1:end-1,:)]\idioG(2:end,:); %equations in columns

% tempA(2:end,:)= diag(.2*ones(nG,1)); %equations in columns

uG          =idioG(2:end,:)-[ones(size(idioG,1)-1,1) idioG(1:end-1,:)]*tempA; 
tempQ       =cov(uG);

tempA       =tempA(2:end,:)';    %remove intercept -- equations in rows

tempP       =(eye(nG^2)-kron(tempA,tempA))\tempQ(:);
tempP       =reshape(tempP,nG,nG);



%NOTE:  the computation of tempA, tempQ and tempP for idio G assumes that
%       for the GDP releases the vector of states is aligned as 
%       [e_t,e_{t-1},...,e_{t-q}] while in fact it is 
%       [e1_t,e1_{t-1},...,e1_{t-q},e2_t,e2_{t-1},...,e2_{t-q},...]
%       the lines below reshuffle the entries of all three matrices to 
%       match the actual composition of the states vector


%initialize A (idio G)
blockA      =zeros(nG*(q+1),nG*(q+1)); 
for j=1:nG
    
    blockA( (q+1)*(j-1)+1, 1:q+1:(iG_ends-iG_starts+1) )       =tempA(j,:); %equation in rows
    blockA( (q+1)*(j-1)+2:(q+1)*j, (q+1)*(j-1)+1:(q+1)*j-1 )   =eye((q+1)-1);
    
end

A_init(iG_starts:iG_ends,iG_starts:iG_ends) =blockA;




%initialize Q (idio G)
blockQ                                      =zeros(nG*(q+1),nG*(q+1)); 
blockQ(1:(q+1):nG*(q+1),1:(q+1):nG*(q+1))   =tempQ;


%initialize P (idio G)
blockP                                      =zeros(nG*(q+1),nG*(q+1)); 
blockP(1:(q+1):nG*(q+1),1:(q+1):nG*(q+1))   =tempP;


Q_init(iG_starts:iG_ends,iG_starts:iG_ends) =blockQ;
P_init(iG_starts:iG_ends,iG_starts:iG_ends) =blockP;
        





%initialize states (all)
S_init =zeros(nS,1);



%load structure
SystemMatrices_init.S =S_init; 
SystemMatrices_init.P =P_init;

SystemMatrices_init.C =C_init; 
SystemMatrices_init.R =R_init;
SystemMatrices_init.A =A_init; 
SystemMatrices_init.Q =Q_init;