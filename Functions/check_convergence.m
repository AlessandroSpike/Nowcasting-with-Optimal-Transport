function llconverged=check_convergence(new_value,old_value,threshold)
     delta_loglik =abs(new_value-old_value);
    avg_loglik   =(abs(new_value)+abs(old_value)+eps)/2;
    
    if (delta_loglik/avg_loglik)<threshold; llconverged=1; else
        llconverged=0;
    end
   
end

