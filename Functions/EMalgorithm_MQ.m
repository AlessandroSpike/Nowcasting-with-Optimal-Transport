function [SystemMatrices_end,loglklhd,x_init]=EMalgorithm_MQ(x,modelSpec,SystemMatrices_init,dove,isSink)

%returns smoothed states and their variance + estimates of the system's matrices



[T,n]       =size(x);      


%unpack model specification
r       =modelSpec.r;               %number of factors
p       =modelSpec.p;               %number of lags in factor VAR


blocks  =modelSpec.blocks;          %block restrictions



%restrictions are written as R1*C = R2
R1      =modelSpec.R1;              %matrix for loading restrictions(1) 
R2      =modelSpec.R2;              %matrix for loading restrictions(2)
Rpoly   =[1 R1(:,1)'];


q       =size(R1,1);                %number of lags in M-Q map


nM      =modelSpec.nM;              %number of monthly variables
nQ      =modelSpec.nQ;              %number of quarterly variables
nB      =size(blocks,2);            %number of blocks






% *     *     *     *   E X P E C T A T I O N   *     *     *     *

KalmanFilterOutput          =kalman_filter(x,SystemMatrices_init);
KalmanSmootherOutput        =kalman_smoother(KalmanFilterOutput);


%smoothed states and their variance 
S_smooth                    =KalmanSmootherOutput.S_smooth;     %E[S(t)]
P_smooth                    =KalmanSmootherOutput.P_smooth;     %Var[S(t)]
PP_smooth                   =KalmanSmootherOutput.PP_smooth;    %Cov[S(t),S(t-1)]


%log likelihood
loglklhd                    =KalmanFilterOutput.loglklhd;




% *     *     *     *   M A X I M I Z A T I O N    *     *     *     *
%[Shumway&Stoffer(2000)]


nSf     =sum(r.*(max(p,q)+1));      %number of states (factors & own lags)
nSm     =nM;                        %number of states (idio M)
nSq     =nQ*(q+1);                  %number of states (idio Q & own lags)

nS      =nSf+nSm+nSq;               %total number of states



%coefficients matrices
C_end   =zeros(n,nS); 
A_end   =zeros(nS,nS); 
Q_end   =zeros(nS,nS);
P_end   =zeros(nS,nS);


P       =zeros(nS,nS); 
Pl      =zeros(nS,nS); 
PPl     =zeros(nS,nS);

for t=1:T

    P   =P+P_smooth{t+1};           %sum(Var[S(t)])
    Pl  =Pl+P_smooth{t};            %sum(Var[S(t-1)])
    PPl =PPl+PP_smooth{t+1};        %sum(Cov[S(t),S(t-1)])
end




% * ~ ~ * ~ ~ * ~ ~ *  S T A T E   E Q U A T I O N  * ~ ~ * ~ ~ * ~ ~ *
    
% * ~ ~ * ~ ~ * ~ ~ *     F  A  C  T  O  R  S       * ~ ~ * ~ ~ * ~ ~ *



nSfi     =cumsum([1 r.*(max(p,q)+1)]); %location of factors in C A Q

for b=1:nB
    
    
    rb          =r(b);              %number of factors per block
    k           =max(p,q);          %number of effective lags
    
       
    %position of factors in states' vector
    F_starts    =nSfi(b);
    F_ends      =nSfi(b)+rb*p-1;
    
 
    
    %SUFFICIENT STATISTICS
       
    
    %E[F(t)F(t)']
    E_FF    =S_smooth( F_starts:F_ends,2:end )*S_smooth( F_starts:F_ends,2:end )'+...
             P( F_starts:F_ends, F_starts:F_ends );               
    
    %E[F(t-1)F(t-1)']
    E_FlFl  =S_smooth( F_starts:F_ends,1:end-1 )*S_smooth( F_starts:F_ends,1:end-1 )'+...
             Pl( F_starts:F_ends,F_starts:F_ends );              
    
    %E[F(t)F(t-1)']
    E_FFl   =S_smooth( F_starts:F_ends,2:end )*S_smooth( F_starts:F_ends,1:end-1 )'+...
             PPl( F_starts:F_ends,F_starts:F_ends );             
        
    
         
    %relevant coefficients
    tempA   =E_FFl/E_FlFl; 
    tempQ   =(1/T)*(E_FF-tempA*E_FFl');

    

    %update A matrix for relevant block
    blockA                      =zeros(rb*(k+1),rb*(k+1));
    blockA(1:rb,1:rb*p)         =tempA(1:rb,1:rb*p); 
    blockA(rb+1:end,1:end-rb)   =eye(rb*k);
    
    
    %update A (factors)
    A_end(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = blockA;
    
    
    
    
    %update Q matrix for relevant block
    blockQ                      =zeros(rb*(k+1),rb*(k+1));
    blockQ(1:rb,1:rb)           =tempQ(1:rb,1:rb);
    
    
    %update Q (factors)
    Q_end(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = blockQ;
    
    
    
    
    %update P matrix for relevant block
    blockP                      =( eye((rb*(k+1))^2) - kron(blockA,blockA) )\blockQ(:);
    
    P_end(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = reshape(blockP,rb*(k+1),rb*(k+1));

end







% * ~ ~ * ~ ~ * ~ ~ *  S T A T E   E Q U A T I O N  * ~ ~ * ~ ~ * ~ ~ *

% * ~ ~ * ~ ~ * ~ ~ *   I D I O S Y N C R A T I C   * ~ ~ * ~ ~ * ~ ~ *

%idioM
for i=1:nM
    
    
    %position of idioM in states' vector
    iM_starts   =nSf+i;
    iM_ends     =nSf+i;
    

    
    %SUFFICIENT STATISTICS

    
    %E[eM(t)eM(t)']
    E_eMeM   =S_smooth( iM_starts:iM_ends,2:end )*S_smooth( iM_starts:iM_ends,2:end )'+...
                P( iM_starts:iM_ends,iM_starts:iM_ends ); 
          
    %E[eM(t-1)eM(t-1)']
    E_eMleMl =S_smooth( iM_starts:iM_ends,1:end-1 )*S_smooth( iM_starts:iM_ends,1:end-1 )'+...
                Pl( iM_starts:iM_ends,iM_starts:iM_ends ); 
          
    %E[eM(t)eM(t-1)']
    E_eMeMl  =S_smooth( iM_starts:iM_ends,2:end )*S_smooth( iM_starts:iM_ends,1:end-1 )'+...
                PPl( iM_starts:iM_ends,iM_starts:iM_ends ); 
          
          
          
          
    %relevant coefficients
    tempA   =E_eMeMl/E_eMleMl;  %AR(1)
    tempQ   =(1/T)*(E_eMeM-tempA*E_eMeMl');
    
    
    %update A (idio M)
    A_end( iM_starts:iM_ends,iM_starts:iM_ends ) =tempA;

    
    %update Q (idio M)
    Q_end( iM_starts:iM_ends,iM_starts:iM_ends ) =tempQ;


    %update P (idio M)
    P_end( iM_starts:iM_ends,iM_starts:iM_ends ) =(1-tempA^2)\tempQ;
end



%idioQ
for i=nM+1:n
    
    %quarterly other than GDP have idio ~ AR(1)
    if ~ismember(i,find(modelSpec.isGDPrelease))
           
        
        
        %position of idio Q in states' vector
        iQ_starts   =nSf+nSm+1+(i-nM-1)*(q+1);
        iQ_ends     =nSf+nSm+1+(i-nM-1)*(q+1);
        


        % SUFFICIENT STATISTICS

        
        %E[eQ(t)eQ(t)']
        E_eQeQ   =S_smooth( iQ_starts:iQ_ends,2:end)*S_smooth( iQ_starts:iQ_ends,2:end)'+...
                    P( iQ_starts:iQ_ends,iQ_starts:iQ_ends ); 

        %E[eQ(t-1)eQ(t-1)']
        E_eQleQl =S_smooth( iQ_starts:iQ_ends,1:end-1)*S_smooth( iQ_starts:iQ_ends,1:end-1)'+...
                    Pl( iQ_starts:iQ_ends,iQ_starts:iQ_ends ); 

        %E[eQ(t)eQ(t-1)']
        E_eQeQl  =S_smooth( iQ_starts:iQ_ends,2:end)*S_smooth( iQ_starts:iQ_ends,1:end-1)'+...
                    PPl( iQ_starts:iQ_ends,iQ_starts:iQ_ends ); 

              
              
              
        %relevant coefficients
        tempA   =E_eQeQl/E_eQleQl;  %AR(1)
        tempQ   =(1/T)*(E_eQeQ-tempA*E_eQeQl');
    

        %update position of idio Q in states' vector    
        iQ_ends     =nSf+nSm+(i-nM)*(q+1);


        %update A matrix for relevant block
        blockA                  =zeros(q+1,q+1); 
        blockA(1,1)             =tempA; 
        blockA(2:end,1:end-1)   =eye(q);

        %update A (idio Q)
        A_end( iQ_starts:iQ_ends,iQ_starts:iQ_ends ) =blockA;

          
        %update Q matrix for relevant block
        blockQ                  =zeros(q+1,q+1); 
        blockQ(1,1)             =tempQ; 

        %update Q (idio Q)
        Q_end( iQ_starts:iQ_ends,iQ_starts:iQ_ends ) =blockQ;  

                

        %initialize P (idio Q)
        blockP                   =(eye((q+1)^2) - kron(blockA,blockA))\blockQ(:);
        
        P_end(iQ_starts:iQ_ends,iQ_starts:iQ_ends) =reshape(blockP,q+1,q+1);

    end
end



%idio for GDP releases ~ VAR(1)
nG          =sum(modelSpec.isGDPrelease);       %number of GDP releases in VAR
k           =q+1;                               %number of parameters in each equation


%position of idio G in states' vector
iG_starts   =nSf+nSm+(nQ-nG)*k+1;
iG_ends     =nSf+nSm+nSq;



% SUFFICIENT STATISTICS

%E[eQ(t)eQ(t)']
E_eQeQ   =S_smooth( iG_starts:k:iG_ends,2:end)*S_smooth( iG_starts:k:iG_ends,2:end)'+...
            P( iG_starts:k:iG_ends,iG_starts:k:iG_ends ); 

%E[eQ(t-1)eQ(t-1)']
E_eQleQl =S_smooth( iG_starts:k:iG_ends,1:end-1)*S_smooth( iG_starts:k:iG_ends,1:end-1)'+...
            Pl( iG_starts:k:iG_ends,iG_starts:k:iG_ends ); 

%E[eQ(t)eQ(t-1)']
E_eQeQl  =S_smooth( iG_starts:k:iG_ends,2:end)*S_smooth( iG_starts:k:iG_ends,1:end-1)'+...
            PPl( iG_starts:k:iG_ends,iG_starts:k:iG_ends ); 


        
%relevant coefficients        
tempA   =E_eQeQl/E_eQleQl;  %VAR(1)
tempQ   =(1/T)*(E_eQeQ-tempA*E_eQeQl');


    
%update A matrix for relevant block (rearrange positions)
blockA  =zeros(nG*k,nG*k); 

for j=1:nG

    blockA( k*(j-1)+1, 1:k:(iG_ends-iG_starts+1) )  =tempA(j,:);
    blockA( k*(j-1)+2:k*j, k*(j-1)+1:k*j-1 )        =eye(q);

end



%update Q matrix for relevant block
blockQ                              =zeros(nG*k,nG*k); 
blockQ(1:k:nG*k,1:k:nG*k)           =tempQ;



%update P matrix for relevant block
blockP = (eye((nG*k)^2) - kron(blockA,blockA))\blockQ(:);
blockP = reshape(blockP,nG*k,nG*k);



%update A
A_end( iG_starts:iG_ends,iG_starts:iG_ends ) =blockA;

  

%update Q
Q_end( iG_starts:iG_ends,iG_starts:iG_ends ) =blockQ;
          

%update P
P_end( iG_starts:iG_ends,iG_starts:iG_ends ) =blockP;




  
  
      
  
% * ~ ~ * ~ ~ *  M E A S U R E M E N T   E Q U A T I O N  * ~ ~ * ~ ~ *

% * ~ ~ * ~ ~ *             F  A  C  T  O  R  S           * ~ ~ * ~ ~ *

% * ~ ~ * ~ ~ *                      &                    * ~ ~ * ~ ~ *

% * ~ ~ * ~ ~ *         I D I O S Y N C R A T I C         * ~ ~ * ~ ~ *

 
 
% define superblocks as combination of factors, and group variables
% according to which one they load on distinguishing bwn M and Q


k                   =max(p,q);          %number of effective lags


superB              =unique(blocks,'rows'); 
nSB                 =size(superB,1);


%factors and their lags
MloadPerSB          =zeros(nSB,nSf);    
QloadPerSB          =zeros(nSB,nSf); 
QloadPerSBr         =zeros(nSB,nSf);


%position of factors in states vector
nSfi                =cumsum([1 r.*(max(p,q)+1)]); 
nSfiB               =repmat(r,nSB,1).*superB;

for b=1:nB

    %position of monthly loadings in super block
    MloadPerSB(:,nSfi(b):nSfi(b)+r(b)-1)           =repmat(superB(:,b),1,r(b));    
    
    %position of quarterly loadings in super block
    QloadPerSB(:,nSfi(b):nSfi(b)+(k+1)*r(b)-1)     =repmat(superB(:,b),1,(k+1)*r(b));

    %position of restricted quarterly loadings in super block
    QloadPerSBr(:,nSfi(b):nSfi(b)+(q+1)*r(b)-1)    =repmat(superB(:,b),1,(q+1)*r(b));
end



%idiosyncratic
iMloadPerSB         =zeros(n,nS); 
iQloadPerSB         =zeros(n,nS); 

iMloadPerSB(1:nM,nSf+1:nSf+nSm)   =eye(nM); 
iQloadPerSB(nM+1:n,nSf+nSm+1:end) =kron(eye(nQ),ones(1,q+1));


%build identifiers -- these will pick the right columns of the states
%vector depending on which variables load on which states in each block
MloadPerSB          =logical(MloadPerSB); 
QloadPerSB          =logical(QloadPerSB);       %quarterly loading for each super block
QloadPerSBr         =logical(QloadPerSBr);      %quarterly restricted loading for each super block
iMloadPerSB         =logical(iMloadPerSB);    
iQloadPerSB         =logical(iQloadPerSB);      %idiosyncratic quarterly loading for each super block

for sb=1:nSB
    
    
    %find all variables that load on all factors in super block
    selectV     =find(ismember(blocks,superB(sb,:),'rows'));
    
    MinSB       =selectV(selectV<=nM);          %monthly variables in block
    nMSB        =numel(MinSB); 
    
    QinSB       =selectV(selectV>nM);           %quarterly variables in block
    nQSB        =numel(QinSB);
    

    %vec(C)=[sum(kron(E(FF'),W))]^{-1}[vec(sum(Wy*E(F')))-vec(sum(W*E(F*e')))]
    %vec(C)=[MC1]^{-1}[MC2]
    
    
    
    % M O N T H L Y   L O A D I N G S
    
    if~isempty(MinSB)
        
        MC1     =zeros(nMSB*sum(MloadPerSB(sb,:)),nMSB*sum(MloadPerSB(sb,:))); 
        MC2     =zeros(nMSB,sum(MloadPerSB(sb,:)));
        
        for t=1:T
            
            %track missing values
            Wx =x(t,MinSB)';    W =diag(~isnan(Wx));    Wx(isnan(Wx)) =0;
            
            
            %
            MC1 =MC1 + kron(S_smooth(MloadPerSB(sb,:),t+1)*S_smooth(MloadPerSB(sb,:),t+1)'+...
                             P_smooth{t+1}(MloadPerSB(sb,:),MloadPerSB(sb,:)),W);
            %
            MC2 =MC2 + Wx*S_smooth(MloadPerSB(sb,:),t+1)'-...
                       W*(S_smooth(any(iMloadPerSB(MinSB,:),1),t+1)*S_smooth(MloadPerSB(sb,:),t+1)'+...
                           P_smooth{t+1}(any(iMloadPerSB(MinSB,:),1),MloadPerSB(sb,:)));
            
        end
        
        %update C matrix for monthly variables
        vecC    =MC1\MC2(:);         
        tempC   =reshape(vecC,nMSB,sum(MloadPerSB(sb,:)));
        
        
        %update C
        C_end(MinSB,MloadPerSB(sb,:))=tempC;
        
    end
    
    
    
    % Q U A R T E R L Y   L O A D I N G S
    
    if~isempty(QinSB)
        
        
        %same formulas as above
        
        MC1     =zeros(nQSB*sum(QloadPerSB(sb,:)),nQSB*sum(QloadPerSB(sb,:))); 
        MC1r    =zeros(nQSB*sum(QloadPerSBr(sb,:)),nQSB*sum(QloadPerSBr(sb,:))); 
        MC2     =zeros(nQSB,sum(QloadPerSB(sb,:)));
        
        for t=1:T
            
            %track missing values
            Wx =x(t,QinSB)';     W =diag(~isnan(Wx));     Wx(isnan(Wx)) =0;
            
            %
            MC1  =MC1+kron(S_smooth(QloadPerSB(sb,:),t+1)*S_smooth(QloadPerSB(sb,:),t+1)'+...
                  P_smooth{t+1}(QloadPerSB(sb,:),QloadPerSB(sb,:)),W);
            %
            MC1r =MC1r+kron(S_smooth(QloadPerSBr(sb,:),t+1)*S_smooth(QloadPerSBr(sb,:),t+1)'+...
                  P_smooth{t+1}(QloadPerSBr(sb,:),QloadPerSBr(sb,:)),W);
            %
            MC2  =MC2+Wx*S_smooth(QloadPerSB(sb,:),t+1)'-...
                  W*(kron(eye(nQSB),Rpoly)*S_smooth(any(iQloadPerSB(QinSB,:),1),t+1)*S_smooth(QloadPerSB(sb,:),t+1)'+...
                  kron(eye(nQSB),Rpoly)*P_smooth{t+1}(any(iQloadPerSB(QinSB,:),1),QloadPerSB(sb,:)));
        end
        
        
        %update C matrix for quarterly variables
        vecC    =MC1\MC2(:); 
        tempC   =reshape(vecC,nQSB,sum(QloadPerSB(sb,:)));
        
        
        %impose restrictions (loading identifiers)
        nSfiBi  =nSfiB(sb,:); 
        nSfiBi  =nSfiBi(nSfiBi~=0); 
        
        %number of restricted coefficients
        nCr     =(q+1)*sum(nSfiBi); 
        
        
        %position of restricted coefficients
        irBr    =[1 cumsum(nSfiBi)*q+1]; 
        irBc    =[1 cumsum(nSfiBi)*(q+1)+1];
        
        
        R1B     =zeros(nCr-sum(nSfiBi),nCr); 
        R2B     =zeros(nCr-sum(nSfiBi),1);
        
        for ir=1:length(nSfiBi)
            
            R1B(irBr(ir):irBr(ir+1)-1,irBc(ir):irBc(ir+1)-1) =kron(R1,eye(nSfiBi(ir))); 
            R2B(irBr(ir):irBr(ir+1)-1)                       =kron(R2,ones(nSfiBi(ir),1));
        end
        
        R1B     =kron(eye(nQSB),R1B); 
        R2B     =kron(ones(nQSB,1),R2B); 
        
        
        %all C coefficients
        tempCall                        =zeros(nSf,nQSB); 
        tempCall(QloadPerSB(sb,:),:)    =tempC'; 
        
        
        %C coefficients subject to restriction
        tempCr                          =zeros(nSf,nQSB); 
        tempCr(QloadPerSBr(sb,:),:)     =tempCall(QloadPerSBr(sb,:),:); 

        tempCr                          =tempCr(:); 
        tempCr(tempCr==0)               =[];
        
        
        %impose loadings restrictions C 
        tempCr                          =tempCr - MC1r\R1B'/(R1B/MC1r*R1B')*(R1B*tempCr-R2B);
        tempCr                          =reshape(tempCr,nCr,nQSB); 
        
        
        %replace
        tempCall(QloadPerSBr(sb,:),:)   =tempCr;
        
        
        %update C matrix for quarterly variables
        C_end(QinSB,QloadPerSB(sb,:))   =tempCall(QloadPerSB(sb,:),:)';    

    end
end


%update C idiosyncratic
C_end(1:nM,nSf+1:nSf+nSm)           =eye(nM);
C_end(nM+1:nM+nQ,nSf+nSm+1:nS)      =kron(eye(nQ),Rpoly);


%update R (small number)
R_end                               =diag(1e-04*ones(n,1));




%load structure with results
SystemMatrices_end.S =S_smooth(:,1);
SystemMatrices_end.P =P_end;
SystemMatrices_end.C =C_end; 
SystemMatrices_end.R =R_end;
SystemMatrices_end.A =A_end; 
SystemMatrices_end.Q =Q_end;
 %x_state per sinkhorn
    if isSink==1
        x_aus= ((S_smooth(1:5,2:end)'*C_end(:,1:5)').*kron(std(x),ones(T,1)))+...
        kron(mean(x),ones(T,1));         %estimated X
        x_init=x;
        x_init(dove)=x_aus(dove);
    else
        x_init=x;
    end

% -----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*----- %







