function [x_end,divergence]=OT_imputerCONDITIONAL(x_init,iterOt,dove,M,learnRate,sqGradDecay,iterRSMpro,epsilon)
   
   divergence=nan(iterOt,1);
   posizioni=1:size(x_init,1);

    for t=1:iterOt
        averageSqGrad = [];
        ausk=randperm(size(posizioni,2));
        ausl=ausk+1;
        ausl(ausl>size(x_init,1))=size(x_init,1);
        posk=ausk(1:M);       
        posl=ausl(1:M);      
        dovek=dove(posk,:);
        dovel=dove(posl,:);
        K=x_init(posk,:);
        L=x_init(posl,:);
        pd=pdist2(K,L);
        c = 1/2*pd.^2;
        
        % mu=ones(size(K,1),1)/size(K,1);
        % nu=ones(1,size(L,1))/size(L,1);
        mu=posk/sum(posk);
        nu=posl/sum(posl);
        mu=mu';
        % 
        f = zeros(M,1);
        g = zeros(1,M);
        for tt=1:100
             g = -epsilon*log( sum(mu .* exp(-(c-f)/epsilon),1));
             f =-epsilon*log( sum(nu .* exp(-(c-g)/epsilon),2) );
        end
        c2 = 1/2*pdist2(K,K).^2;
       
        f2 = zeros(M,1);
        for tt=1:100
             f2=-epsilon*log( sum(nu .* exp(-(c2-f2)/epsilon),2) );
        end
        c3 = 1/2*pdist2(L,L).^2;
       
        f3 = zeros(1,M);
        for tt=1:100
             f3 =-epsilon*log( sum(mu .* exp(-(c3-f3)/epsilon),2) );
        end
        

     
        GradP_k=zeros(size(K));
        for mk = 1:M
            x_k=K(mk,:);
            P_l1num=zeros(size(L));
            P_l1den=zeros(size(L));
            P_l2num=zeros(size(L));
            P_l2den=zeros(size(L));
            for ml = 1:M
                x_l=L(ml,:);
                x_k2=K(ml,:);
                P_l1num(ml,:)=exp((1/epsilon)*(g(ml)-c(mk,ml)))*(x_k-x_l);
                P_l1den(ml,:)=exp((1/epsilon)*(g(ml)-c(mk,ml)));
                P_l2num(ml,:)=exp((1/epsilon)*(f2(ml)-c2(mk,ml)))*(x_k-x_k2);
                P_l2den(ml,:)=exp((1/epsilon)*(f2(ml)-c2(mk,ml)));
            end
            GradP_k(mk,:)=mu(mk)*(1/(1+epsilon))*(sum(P_l1num)./sum(P_l1den)-...
                sum(P_l2num)./sum(P_l2den));
        end
    
        GradP_l=zeros(size(L));
        for ml = 1:M
            x_l=L(ml,:);
            P_k1num=zeros(size(K));
            P_k1den=zeros(size(K));
            P_k2num=zeros(size(K));
            P_k2den=zeros(size(K));
            for mk = 1:M
                x_k=K(mk,:);
                x_l2=L(mk,:);           
                P_k1num(mk,:)=exp((1/epsilon)*(f(mk)-c(mk,ml)))*(x_l-x_k);
                P_k1den(mk,:)=exp((1/epsilon)*(f(mk)-c(mk,ml)));
                P_k2num(mk,:)=exp((1/epsilon)*(f3(mk)-c3(mk,ml)))*(x_l-x_l2);
                P_k2den(mk,:)=exp((1/epsilon)*(f3(mk)-c3(mk,ml)));
            end
            GradP_l(ml,:)=nu(ml)*(1/(1+epsilon))*(sum(P_k1num)./sum(P_k1den)-...
                sum(P_k2num)./sum(P_k2den));
        end
        params=[K;L];
        grad=[GradP_k;GradP_l];
        params2=params([dovek;dovel]==1);
        % paramOld=params2;
        % diffParamOld=100000;
        grad2=grad([dovek;dovel]==1);
        for z= 1:iterRSMpro
            [params2,averageSqGrad] = rmspropupdate(params2,grad2,averageSqGrad,learnRate,sqGradDecay);
            % diffParam=mean(abs(paramOld-params2));
            % if (diffParamOld-diffParam)<10^-5
            %     break
            % end
            % paramOld=params2;
            % diffParamOld=diffParam;
        end
        params([dovek;dovel]==1)=params2;
        x_init(posk,:)=(params(1:M,:));
        x_init(posl,:)=(params(M+1:end,:)); 
        divergence(t)=sum(mu.*(f-f2))+sum(nu'.*(g'-f3));
       
        % if t>2 && abs(divergence(t)-divergence(t-1))/divergence(t-1)<10^-5
        % 
        %     break
        % end
    end
    
    x_end=x_init;
end
