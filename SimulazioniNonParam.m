clc;clearvars;close all
rng('default')
%% parameter
nn=[10 50]; %num serie
TT=[50 100]; %lenght simul
Num_Sim=500; % num simul
PrcNan=[.1 .2 .5]; %percentuale nan
Kn=[5];
r=1; % number of factors
Leaf=[50];
% set sinkhorn param
M1=.4;% size batches
learnRate = .001;
sqGradDecay = .95;
iterOt=500; % num iter sink
iterRSMpro=20; % num iter rms  update
init_epsilon=1; 
num_cluster=1;
%% simulare series
for N=1:length(nn)
    n=nn(N);
    for tt=1:length(TT)
        T=TT(tt);
        for prnan=1:length(PrcNan)
            for uu=1:length(Kn)
                for ll=1:length(Leaf)
                    k=1;
                    %% container
                    X_KNN_RMSE=nan(Num_Sim,1);
                    X_OT_RMSE=nan(Num_Sim,1);
                    X_TREE_RMSE=nan(Num_Sim,1);
                    SinkDIV=nan(Num_Sim,iterOt);                  
                    while k<Num_Sim
                       
                        try
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
                            prcNan=PrcNan(prnan);
                            missingVal=randi([100 n*T],n*T*prcNan,1);
                            X(missingVal)=nan;
                          
                            
                            %knn imputation
                            kn=Kn(uu);
                            Xhat_knn = knnimpute(X',kn);
                            %disp('knn Fatto')
                            
                        
                            %tree imputation
                            tmp = templateTree('Surrogate','on');
                            l=1:size(X,2);
                            Xhat_tree=X;
                            leaf=Leaf(ll);
                            for i=1:size(X,2)
                                tengo=setdiff(l,i);
                                missingCustAge = ismissing(X(:,i));
                                % Fit ensemble of regression learners
                                rfCustAge = fitrensemble(X(:,tengo),X(:,i),'Method','LSBoost',...
                                    'NumLearningCycles',leaf,'Learners',tmp);
                                imputedDataTree = predict(rfCustAge,...
                                    X(missingCustAge,tengo));
                                Xhat_tree(ismissing(X(:,i)),i)=imputedDataTree;
                            end
                            %disp('Tree Fatto')
        
                            % sinkhorn imputation
                            x=X;
                            dove=isnan(x);
                            media=2*randn(size(X))+repmat(nanmean(x),size(x,1),1);
                            x(dove)=media(dove);
                            idx = kmeans(x,1,'Replicates',1);
                            Xhat_Sink=nan(size(x));
                            for a=1:num_cluster
                                xaus=x(idx==a,:);
                                M=round(M1*size(xaus,1));
                                rifaccio=1;
                                epsilon=1;
                                while rifaccio==1 
                                    % [Xhat_Sinkaus,S_divergence]=OT_imputer1batch(xaus,iterOt,dove,learnRate,sqGradDecay,iterRSMpro,epsilon);
                                    [Xhat_Sinkaus,S_divergence]=OT_imputerCONDITIONAL(xaus,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon);
                                    Xhat_Sink(idx==a,:)=Xhat_Sinkaus;
                                    if sum(sum(isnan(Xhat_Sinkaus)))>0
                                        epsilon=epsilon+0.1;
                                       disp('occhio')
                                   else
                                       rifaccio=0;
                                   end
                                end
                            end
                            %disp('Sinkhorn Fatto')
                         
        
                             X_KNN_RMSE(k) = sqrt(mean(X1(isnan(X))-Xhat_knn(isnan(X))).^2);
                             X_TREE_RMSE(k) = sqrt(mean(X1(isnan(X))-Xhat_tree(isnan(X))).^2);
                            X_OT_RMSE(k) = sqrt(mean(X1(isnan(X))-Xhat_Sink(isnan(X))).^2);
                            SinkDIV(k,:)=S_divergence;
                            disp(['Iter: ',num2str(k),' RMSE X knn: ',num2str(mean(X_KNN_RMSE(1:k))),' -- X tree: ',num2str(mean(X_TREE_RMSE(1:k))),...
                                ' -- X Ot: ',num2str(mean(X_OT_RMSE(1:k)))])
                            k=k+1;
                            
                         end
                        
    
                    end
    
                    clc;
                    filename=(['AfterJBES_NP_RisSimul_NumVar',num2str(n),'_LungSerie',num2str(T),'_PercNan',num2str(10*prcNan),...
                        'knr',num2str(kn), '.mat']);
                    save(filename,"X_KNN_RMSE","X_TREE_RMSE","X_OT_RMSE","SinkDIV")
                end
            end
        end 
    end
end
