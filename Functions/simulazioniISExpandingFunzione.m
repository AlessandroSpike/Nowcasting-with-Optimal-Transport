function simulazioniISExpandingFunzione(n,T,prnan,maxiter,r,p,M1,learnRate,sqGradDecay,iterOt,iterRSMpro,init_epsilon,Num_Sim,s)

 %% container
DFM=nan(Num_Sim,1);
OT=nan(Num_Sim,1);
%%  main loop
for i=1:Num_Sim
  
    
    f_0=randn(r,1);
    e_0=randn(n,1);
    A=rand(1)*eye(r,r);
    D=diag(-.9 + 1.8.*rand(n,1));
    Lambda=randn(n,r);
    beta=unifrnd(.1,.9,n,1);
    gamma=(beta./(1-beta))*(1/(1-0.7^2)).*Lambda.^2;
    u_t=zeros(n);
    for n1=1:n
        for n2=1:n
            u_t=(1-D(n1,n2)^2)*sqrt(gamma(n1)*gamma(n2));

        end
    end               
    X=zeros(n,T);
    F=zeros(r,T);
    E=zeros(n,T);
    for t=1:T
        u=randn(n,1)*chol(u_t);
        v=randn(r,r);
        if t==1
            F(:,t)=f_0;
            E(:,t)=e_0;
        else
            F(:,t)=A*F(:,t-1) +v;
            E(:,t)=D*E(:,t-1) +u;
        end
        X(:,t)=Lambda*F(:,t) + E(:,t);  
    end
    X=X';
    X1=X;
    if prnan==1
       missingVal=[1; randi([2 n],round(n*.2),1)];
       X(end,missingVal)=nan;
    elseif prnan==2
       missingVal=[1; randi([2 n],round(n*.2),1)];
       X(end-1,missingVal)=nan;
       missingVal=[1; randi([2 n],round(n*.4),1)];
       X(end,missingVal)=nan;
    elseif prnan==3
       missingVal=[1; randi([2 n],round(n*.2),1)];
       X(end-2,missingVal)=nan;
       missingVal=[1; randi([2 n],round(n*.4),1)];
       X(end-1,missingVal)=nan;
       missingVal=[1; randi([2 n],round(n*.6),1)];
       X(end,missingVal)=nan;
    elseif prnan==4
       missingVal=[1; randi([2 n],round(n*.2),1)];
       X(end-3,missingVal)=nan;
       missingVal=[1; randi([2 n],round(n*.4),1)];
       X(end-2,missingVal)=nan;
       missingVal=[1; randi([2 n],round(n*.6),1)];
       X(end-1,missingVal)=nan;
       missingVal=[1; randi([2 n],round(n*.8),1)];
       X(end,missingVal)=nan;
    end
    disp(i)
    disp('Dati Creati Fatto')
   
    [LL,SystemMatrices_end,StateSpaceP_new,SystemMatrices_end_Collect]=funzioneNowcast_IS(X,r,p,s,maxiter,0,M1,learnRate,...
    sqGradDecay,iterOt,iterRSMpro,init_epsilon);
    disp('Classico Fatto')
    [LL_Sink,SystemMatrices_end_Sink,StateSpaceP_new_Sink,SystemMatrices_end_Collect_Sink]=funzioneNowcast_IS(X,r,p,s,maxiter,1,M1,learnRate,...
    sqGradDecay,iterOt,iterRSMpro,init_epsilon);
    disp('OT Fatto')

    l_DFM = SystemMatrices_end.C(1,1);
    l_OT = SystemMatrices_end_Sink.C(1,1);
    l_VERO=Lambda(1);
    
    Fx_dfm=StateSpaceP_new.States(end,1);
    Fx_ot=StateSpaceP_new_Sink.States(end,1);
    F1=F(end);
    
    f_DFM = Fx_dfm;
    f_OT = Fx_ot;
    f_VERO=F1;

    DFM(i) = 1-((f_VERO*l_VERO-f_DFM*l_DFM)^2)/var(F*l_VERO);
    OT(i) = 1-((f_VERO*l_VERO-f_OT*l_OT)^2)/var(F*l_VERO);
  
   
    disp([' A dfm: ',num2str(mean(DFM(1:i))),' -- A ot: ',num2str(mean(OT(1:i)))])

end
clc;
filename=(['ExpandingAfterJBES_IS_RisSimul_NumVar',num2str(n),'_LungSerie',num2str(T),'_PercNan',num2str(prnan),'_r',num2str(r),'_p',num2str(p),'.mat']);
save(filename,"OT","DFM")
