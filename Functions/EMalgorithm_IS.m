function [SystemMatrices_end,loglklhd,x_init]=EMalgorithm_IS(x,modelSpec,SystemMatrices_init,dove,isSink)


[T,N]   =size(x);


%unpack model specification
r       =modelSpec.r;               %number of factors
p       =modelSpec.p;               %number of lags in factor VAR
s       =modelSpec.s;               %number of lags in idio dynamic

%
blocks  =modelSpec.blocks;          %block restrictions



nM      =modelSpec.nM;              %number of monthly variables
nB      =size(blocks,2);            %number of blocks


% *     *     *     *   E X P E C T A T I O N   *     *     *     *

KalmanFilterOutput          =kalman_filter(x,SystemMatrices_init);
KalmanSmootherOutput        =kalman_smoother(KalmanFilterOutput);


%smoothed states and their variance 
S_smooth                    =KalmanSmootherOutput.S_smooth;
P_smooth                    =KalmanSmootherOutput.P_smooth;
PP_smooth                   =KalmanSmootherOutput.PP_smooth;


%log likelihood
loglklhd                    =KalmanFilterOutput.loglklhd;



% *     *     *     *   M A X I M I Z A T I O N    *     *     *     *
%[Shumway&Stoffer(2000)]

nSf     =sum(r.*p); 
nSiM    =nM*s; 
nS      =nSf+nSiM; %number of states


%coefficients matrices
C_end   =zeros(N,nS); 
A_end   =zeros(nS,nS); 
Q_end   =zeros(nS,nS);


P       =zeros(nS,nS); 
Pl      =zeros(nS,nS); 
PPl     =zeros(nS,nS);

for t=1:T
    
    P   =P+P_smooth{t+1};       %sum(var(S_{t})
    Pl  =Pl+P_smooth{t};        %sum(var(S_{t-1})
    PPl =PPl+PP_smooth{t+1};    %sum(cov(S_{t}S_{t-1})
end


% *~~~*~~~*~~~*~~~*~~~*~~~* STATE EQUATION *~~~*~~~*~~~*~~~*~~~*~~~*
% *~~~* FACTORS *~~~*

nSfi=cumsum([1 r.*p]);

for b=1:nB

    rb      =r(b); %# of factors per block
    
    
    % sufficient statistics
    E_FF    =S_smooth(nSfi(b):nSfi(b)+rb*p-1,2:end)*S_smooth(nSfi(b):nSfi(b)+rb*p-1,2:end)'+...
             P(nSfi(b):nSfi(b)+rb*p-1,nSfi(b):nSfi(b)+rb*p-1); %E(F_{t}F_{t}')

    E_FlFl  =S_smooth(nSfi(b):nSfi(b)+rb*p-1,1:end-1)*S_smooth(nSfi(b):nSfi(b)+rb*p-1,1:end-1)'+...
             Pl(nSfi(b):nSfi(b)+rb*p-1,nSfi(b):nSfi(b)+rb*p-1); %E(F_{t-1}F_{t-1}')

    E_FFl   =S_smooth(nSfi(b):nSfi(b)+rb*p-1,2:end)*S_smooth(nSfi(b):nSfi(b)+rb*p-1,1:end-1)'+...
             PPl(nSfi(b):nSfi(b)+rb*p-1,nSfi(b):nSfi(b)+rb*p-1); %E(F_{t}F_{t-1}')
    
         
    %
    tempA                       =E_FFl/E_FlFl; 
    blockA                      =zeros(rb*p,rb*p);
    blockA(1:rb,1:rb*p)         =tempA(1:rb,1:rb*p); 
    blockA(rb+1:end,1:end-rb)   =eye(rb*(p-1));
    
    A_end(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = blockA;
    
    
    %
    tempQ                       =(1/T)*(E_FF-tempA*E_FFl'); 
    blockQ                      =zeros(rb*p,rb*p);
    blockQ(1:rb,1:rb)           =tempQ(1:rb,1:rb);
    
    Q_end(nSfi(b):nSfi(b+1)-1,nSfi(b):nSfi(b+1)-1) = blockQ;
    
end


% *~~~* IDIOSYNCRATIC *~~~*
%idioM
for i=1:nM
    
    % sufficient statistics
    E_eMeM   =S_smooth(nSf+1+(i-1)*s:nSf+i*s,2:end)*S_smooth(nSf+1+(i-1)*s:nSf+i*s,2:end)'+...
              P(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s); %E(eM_{t}eM_{t}')

    E_eMleMl =S_smooth(nSf+1+(i-1)*s:nSf+i*s,1:end-1)*S_smooth(nSf+1+(i-1)*s:nSf+i*s,1:end-1)'+...
              Pl(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s); %E(eM_{t-1}eM_{t-1}')
    
    E_eMeMl  =S_smooth(nSf+1+(i-1)*s:nSf+i*s,2:end)*S_smooth(nSf+1+(i-1)*s:nSf+i*s,1:end-1)'+...
              PPl(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s); %E(eM_{t}eM_{t-1}')
    
          
    %
    tempA                   =E_eMeMl/E_eMleMl; 
    blockA                  =zeros(s,s); 
    blockA(1,:)             =tempA(1,:); 
    blockA(2:end,1:end-1)   =eye(s-1);
    
    A_end(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s) = blockA;
   
        
    %
    tempQ                   =(1/T)*(E_eMeM-tempA*E_eMeMl'); 
    blockQ                  =zeros(s,s);
    blockQ(1,1)             =tempQ(1,1); 
    
    Q_end(nSf+1+(i-1)*s:nSf+i*s,nSf+1+(i-1)*s:nSf+i*s) = blockQ;
    
end




% *~~~*~~~*~~~*~~~*~~~* MEASUREMENT EQUATION *~~~*~~~*~~~*~~~*~~~*
% define superblocks as combination of factors and group variables
% according to which one they load on distinguishing bwn M and Q

superB      =unique(blocks,'rows'); 
nSB         =size(superB,1);

%factors and their lags
MloadPerSB  =zeros(nSB,nSf); 


nSfi    =cumsum([1 r.*p]); 

for b=1:nB
    
    %position of monthly loadings in super block
    MloadPerSB(:,nSfi(b):nSfi(b)+r(b)-1)           =repmat(superB(:,b),1,r(b));
    
end

%idiosyncratic
iMloadPerSB     =zeros(N,nS); 

iMloadPerSB(1:nM,nSf+1:nSf+nSiM)    =kron(eye(nM),[1 zeros(1,s-1)]); 


%build identifiers -- these will pick the right columns of the states
%vector depending on which variables load on which states in each block
MloadPerSB      =logical(MloadPerSB); 
iMloadPerSB     =logical(iMloadPerSB); 


for sb=1:nSB
    

    %find all variables that load on all factors in super block
    selectV     =find(ismember(blocks,superB(sb,:),'rows'));
    
    MinSB       =selectV(selectV<=nM); 
    nMSB        =numel(MinSB); 
    
    
    %vec(C)=[sum(kron(E(FF'),W))]^{-1}[vec(sum(WyE(F')))-vec(sum(WE(F*e')))]
    %vec(C)=[MC1]^{-1}[MC2]
    % *~~~* MONTHLY LOADINGS *~~~*
    if~isempty(MinSB)
        
        MC1     =zeros(nMSB*sum(MloadPerSB(sb,:)),nMSB*sum(MloadPerSB(sb,:))); 
        MC2     =zeros(nMSB,sum(MloadPerSB(sb,:)));
        
        for t=1:T
            
            %track missing values
            Wx =x(t,MinSB)';     W =diag(~isnan(Wx));     Wx(isnan(Wx)) =0;
            
            
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
    
end


C_end(1:nM,nSf+1:nSf+nSiM)              =kron(eye(nM),[1 zeros(1,s-1)]);


%update R (small number)
R_end   =diag(1e-04*ones(N,1));


%update P
P_end   =(eye(nS^2) - kron(A_end,A_end))\Q_end(:);
P_end   =reshape(P_end,nS,nS);


%load structure with results
SystemMatrices_end.S =S_smooth(:,1); 
SystemMatrices_end.P =P_end;
SystemMatrices_end.C =C_end; 
SystemMatrices_end.R =R_end;
SystemMatrices_end.A =A_end; 
SystemMatrices_end.Q =Q_end;

 %x_state per sinkhorn
    if isSink==1
        x_aus= S_smooth(:,2:end)'*C_end';         %estimated X
        x_init=x;
        x_init(dove)=x_aus(dove);
    else
        x_init=x;
    end

% -----*-----*-----*-----*-----*-----*-----*-----*-----*-----*-----*----- %


