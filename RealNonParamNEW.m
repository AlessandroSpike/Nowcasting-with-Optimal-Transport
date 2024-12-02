clc;clearvars; close all
tic
rng(1,'philox')
%% addpath dati
addpath("Data\")
%% load data
load DatiUSGiannone
%% tolgo o no recessiono
Xtot1=X;
Covid=1;
if Covid==0
    Xtot1(421:432,:)=[]; % periodo covid da togliere
    Dati(421:432,:)=[]; % periodo covid da togliere
    fine=450;
    lung=34*7;
else
    fine=462;
    lung=38*7;
end
%% sinkhorn nowcast
% set sinkhorn param
M1=.4;% size batches
learnRate = .001;
sqGradDecay = .95;
iterOt=200; % num iter sink
iterRSMpro=1; % num iter rms  update
init_epsilon=1;
%% container
NowcastNewOT=nan(lung,1);
NowcastNewKNN=nan(lung,1);
NowcastNewTREE=nan(lung,1);
Contatore=nan(lung,1);
TarNew=nan(lung,1);
%% main
ii=1;
for t=351:3:fine
    ii
    %% transform & standardize
    Target=Xtot1(t,3);
    for i=1:7
        tic
        i
        if i==1
           XNew=Xtot1(73:t+1,:);
           XNew(end-1,3)=nan;
           XNew(end-1,4)=nan;
        elseif i==2
            XNew=Xtot1(73:t,:);
            XNew(end,3)=nan;
            XNew(end,4)=nan;
        else
           XNew=Xtot1(73:t,:);
           XNew(end-i+3:end,:)=nan;
        end

        mX=nanmean(XNew); vX=nanstd(XNew);
        XNew=bsxfun(@minus,XNew,mX); XNew=bsxfun(@rdivide,XNew,vX);
        %% new part
        %sinkhorn imputation
        dove=isnan(XNew);
        optNaN.method   =1;                 
        optNaN.k        =3;
        media=remNaNs_spline(XNew,optNaN);
        %media=2*randn(size(XNew))+repmat(nanmean(XNew),size(XNew,1),1);
        x=XNew;
        x(dove)=media(dove);
        Xhat_Sink=nan(size(x));
        xaus=x;
        M=ceil(M1*size(xaus,1));
        rifaccio=1;
        epsilon=init_epsilon;
        while rifaccio==1   
            [x_endaus,S_divergence]=OT_imputer(xaus,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon);
            if sum(sum(isnan(x_endaus)))>0
               epsilon=epsilon+0.1;
               disp('occhio')
            else
               Xhat_Sink=x_endaus;
               rifaccio=0;
           end
        end

        % knn imputation
        % XKnn=XNew;
        % XKnn=fillmissing(XKnn,'previous');
        % if i==1
        %    XKnn(end-1,3)=nan;
        % elseif x==2
        %     XKnn(end,3)=nan;
        % else
        %    XKnn(end-i+3:end,3)=nan;
        % end
        % Xhat_knn = knnimpute(XKnn,2);
    
        % % tree imputation
        % tmp = templateTree('Surrogate','on');
        % Xhat_tree=XNew;
        % l=1:size(XNew,2);
        % tengo=setdiff(l,3);
        % missingCustAge = ismissing(XNew(:,3));
        % % Fit ensemble of regression learners
        % rfCustAge = fitrensemble(XNew(:,tengo),XNew(:,3),'Method','LSBoost',...
        %     'NumLearningCycles',50,'Learners',tmp,'LearnRate',.5);
        % imputedDataTree = predict(rfCustAge,...
        %     XNew(missingCustAge,tengo));
        % Xhat_tree(ismissing(XNew(:,3)),3)=imputedDataTree;
        %% colleziono
        TarNew(ii)=Target;
        if i>1
            NowcastNewOT(ii)=Xhat_Sink(end,3)*vX(end,3)+mX(end,3);
            % NowcastNewKNN(ii)=Xhat_knn(end,3)*vX(end,3)+mX(end,3);
            % NowcastNewTREE(ii)=Xhat_tree(end,3)*vX(end,3)+mX(end,3);
        else
            NowcastNewOT(ii)=Xhat_Sink(end-1,3)*vX(end,3)+mX(end,3);
            % NowcastNewKNN(ii)=Xhat_knn(end-1,3)*vX(end,3)+mX(end,3);
            % NowcastNewTREE(ii)=Xhat_tree(end-1,3)*vX(end,3)+mX(end,3);
        end

        %  disp([datestr(Dati.Date(t)),' OT: ', num2str(sqrt(mean((TarNew(1:ii)-NowcastNewOT(1:ii)).^2))),' KNN: ',...
        % num2str(sqrt(mean((TarNew(1:ii)-NowcastNewKNN(1:ii)).^2))),' TREE: ',...
        %         num2str(sqrt(mean((TarNew(1:ii)-NowcastNewTREE(1:ii)).^2)))])
        
         Contatore(ii)=i;
         ii=ii+1;
         toc
    end
end

clc
filename=(['NEW_RisRealiUnifrom',num2str(Covid),'.mat']);
% save(filename,"TarNew",'NowcastNewTREE','NowcastNewKNN','NowcastNewOT','Contatore')     
save(filename,"TarNew",'NowcastNewOT','Contatore')     
toc