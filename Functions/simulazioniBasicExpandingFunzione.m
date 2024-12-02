function simulazioniBasicExpandingFunzione(n,T,prnan,maxiter,r,p,M1,learnRate,sqGradDecay,iterOt,iterRSMpro,init_epsilon,Num_Sim)

%% container

DFM=nan(Num_Sim,1);
OT=nan(Num_Sim,1);
%%  main loop
for i=1:Num_Sim
    
    f_0=randn(r,1);
    A=rand(1)*eye(r,r);
    Lambda=randn(n,r);
    
    X=zeros(n,T);
    F=zeros(r,T);
    for t=1:T
        e=randn(n,1);
        u=randn(r,1);
        if t==1
            F(:,t)=f_0;
        else
            F(:,t)=A*F(:,t-1) +u;
        end
        X(:,t)=Lambda*F(:,t) + e;  
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

     [LL,SystemMatrices_end,StateSpaceP_new,SystemMatrices_end_Collect]=funzioneNowcast(X,r,p,maxiter,0,M1,learnRate,...
    sqGradDecay,iterOt,iterRSMpro,init_epsilon);
    disp('Classico Fatto')


    [LL_Sink,SystemMatrices_end_Sink,StateSpaceP_new_Sink,SystemMatrices_end_Collect_Sink]=funzioneNowcast(X,r,p,maxiter,1,M1,learnRate,...
    sqGradDecay,iterOt,iterRSMpro,init_epsilon);
    disp('OT Fatto')

    l_DFM = SystemMatrices_end.C(1);
    l_OT = SystemMatrices_end_Sink.C(1);
    l_VERO=Lambda(1);
    
    Fx_dfm=StateSpaceP_new.States(end);
    Fx_ot=StateSpaceP_new_Sink.States(end);
    F1=F(end);
    
    f_DFM = Fx_dfm;
    f_OT = Fx_ot;
    f_VERO=F1;

    DFM(i) = 1-((f_VERO*l_VERO-f_DFM*l_DFM)^2)/var(F*l_VERO);
    OT(i) = 1-((f_VERO*l_VERO-f_OT*l_OT)^2)/var(F*l_VERO);
  
   
    disp([' A dfm: ',num2str(mean(DFM(1:i))),' -- A ot: ',num2str(mean(OT(1:i)))])
 

end
clc;
filename=(['ExpandingAfterJBES_RisSimul_NumVar',num2str(n),'_LungSerie',num2str(T),'_PercNan',num2str(prnan),'_r',num2str(r),'_p',num2str(p),'.mat']);
save(filename,"DFM","OT")

