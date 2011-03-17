%% Returns average log-likelihoods for given fly trajectories
function avgLL = GetAvgLLs(fly, params, idx, trajStart, model)

avgLL = zeros(length(idx),1);
% infcount = 0;
for ii=1:length(idx)
    i = idx(ii);
    if mod(ii,1000)==0
        disp(['On iteration ' num2str(ii) ' of ' num2str(length(idx))]);
    end
    j = fly.indices{i}(trajStart-1:end-1);
    k = fly.indices{i}(trajStart:end);
    if model==0
        mu_VT = [fly.VT(j), ones(length(j),1)] * params.VT.theta;
        mu_VS = [fly.VS(j), ones(length(j),1)] * params.VS.theta;
        mu_VR = [fly.VR(j), ones(length(j),1)] * params.VR.theta;
        mu_cosPO = [fly.cosPO(j), fly.sinPO(j), ones(length(j),1)] ...
            * params.cosPO.theta;
        mu_sinPO = [fly.cosPO(j), fly.sinPO(j), ones(length(j),1)] ...
            * params.sinPO.theta;
    elseif model==1
        mu_VT = [fly.VT(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VT.theta;
        mu_VS = [fly.VS(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VS.theta;
        mu_VR = [fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VR.theta;
        mu_pos_o = [fly.pos_o(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.pos_o.theta;
    elseif model==2
        mu_VT = [fly.VT(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VT.theta;
        mu_VS = [fly.VS(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VS.theta;
        mu_VR = [fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VR.theta;
        mu_cosPO = [fly.cosPO(j), fly.sinPO(j), fly.stim_RT(j,1), ...
            fly.stim_RT(j,2), ones(length(j),1)] * params.cosPO.theta;
        mu_sinPO = [fly.cosPO(j), fly.sinPO(j), fly.stim_RT(j,1), ...
            fly.stim_RT(j,2), ones(length(j),1)] * params.sinPO.theta;
    elseif model==3
        mu_VT = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
            params.VT.theta;
        mu_VS = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
            params.VS.theta;
        mu_VR = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
            params.VR.theta;
        mu_pos_o = [fly.pos_o(j), fly.VT(j), fly.VS(j), fly.VR(j), ...
            fly.stim_RT(j,1), fly.stim_RT(j,2),ones(length(j),1)] * params.pos_o.theta;
    elseif model==4
        mu_VT = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
            params.VT.theta;
        mu_VS = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
            params.VS.theta;
        mu_VR = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ones(length(j),1)] * ...
            params.VR.theta;
        mu_cosPO = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), fly.VR(j), ...
            fly.stim_RT(j,1), fly.stim_RT(j,2),ones(length(j),1)] * params.cosPO.theta;
        mu_sinPO = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), fly.VS(j), fly.VR(j), ...
            fly.stim_RT(j,1), fly.stim_RT(j,2),ones(length(j),1)] * params.sinPO.theta;
    elseif model==5
        mu_VT = [fly.VT(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VT.theta;
        mu_VS = [fly.VS(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VS.theta;
        mu_VR = [fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2), ...
            ones(length(j),1)] * params.VR.theta;
        mu_cosPO = [fly.cosPO(j), fly.sinPO(j), fly.VR(j), fly.stim_RT(j,1), ...
            fly.stim_RT(j,2), ones(length(j),1)] * params.cosPO.theta;
        mu_sinPO = [fly.cosPO(j), fly.sinPO(j), fly.VR(j), fly.stim_RT(j,1), ...
            fly.stim_RT(j,2), ones(length(j),1)] * params.sinPO.theta;
    end
    llVT = log(normpdf(fly.VT(k), mu_VT, params.VT.sigma));
    llVS = log(normpdf(fly.VS(k), mu_VS, params.VS.sigma));
    llVR = log(normpdf(fly.VR(k), mu_VR, params.VR.sigma));
    if model==1 || model==3
        multmax = 1-2*(fly.pos_o(k) - fly.pos_o(j) > pi);
        llpos_o = ...
            max(log(normpdf(fly.pos_o(k), mu_pos_o, params.pos_o.sigma)),...
            log(normpdf(fly.pos_o(k) + 2*pi*multmax, mu_pos_o, params.pos_o.sigma)));
        avgLL(ii) = mean(llVT + llVS + llVR + llpos_o);
    elseif model==0 || model==2 || model==4 || model==5
        llcosPO = log(normpdf(fly.cosPO(k), mu_cosPO, params.cosPO.sigma));
        llsinPO = log(normpdf(fly.sinPO(k), mu_sinPO, params.sinPO.sigma));
        avgLL(ii) = mean(llVT + llVS + llVR + llcosPO + llsinPO);
    end
    
%     if avgLL(ii) == -Inf
%         infcount = infcount + 1;
%         if infcount == 1
%             break;
%         end
%     end
    
end