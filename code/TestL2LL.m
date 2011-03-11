%% Get (average) log-likelihoods of validation set:
load ../data/L2_dec_all.mat
fly = L2_dec_strct;
clear L2_dec_strct;

rand('seed', 1);
valIdx = randsample(length(fly.indices), floor(0.2 * length(fly.indices)));

disp('Calculating average log-likelihoods of poopy set');
mutantAvgLL = zeros(length(valIdx),1);
infcount = 0;
for ii=1:length(valIdx)
    i = valIdx(ii);
    ll = 0;
    if mod(ii,1000)==0
        disp(['On iteration ' num2str(ii) ' of ' num2str(length(valIdx))]);
    end
    j = fly.indices{i}(7:end-1);
    k = fly.indices{i}(8:end);
    mu_VT = [fly.VT(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
        ones(length(j),1)] * params.VT.theta;
%     mu_VT = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), fly.VT(j), fly.VS(j), ...
%         fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
%         params.VT.theta;
    mu_VS = [fly.VS(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
        ones(length(j),1)] * params.VS.theta;
%     mu_VS = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), fly.VT(j), fly.VS(j), ...
%         fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
%         params.VS.theta;
    mu_VR = [fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
        ones(length(j),1)] * params.VR.theta;
%     mu_VR = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), fly.VT(j), fly.VS(j), ...
%         fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
%         params.VR.theta;
    mu_pos_o = [fly.pos_o(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
        ones(length(j),1)] * params.VT.theta;
%     mu_pos_o = [fly.pos_o(j), fly.VT(j), fly.VS(j), fly.VR(j), ...
%         fly.stim_RT(j,1), fly.stim_RT(j,2),ones(length(j),1)] * params.pos_o.theta;
    llVT = log(normpdf(fly.VT(k), mu_VT, params.VT.sigma));
    llVS = log(normpdf(fly.VS(k), mu_VS, params.VS.sigma));
    llVR = log(normpdf(fly.VR(k), mu_VR, params.VR.sigma));
    llpos_o = log(normpdf(fly.pos_o(k), mu_pos_o, params.pos_o.sigma));
    
    mutantAvgLL(ii) = mean(llVT + llVS + llVR + llpos_o);
    
    if mutantAvgLL(ii) == -Inf
        infcount = infcount + 1;
        if infcount == 1
            break;
        end
    end
    
end

