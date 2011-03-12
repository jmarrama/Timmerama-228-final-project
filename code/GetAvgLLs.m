%% Returns average log-likelihoods for given fly trajectories
function avgLL = GetAvgLLs(fly, params, idx)

avgLL = zeros(length(idx),1);
% infcount = 0;
for ii=1:length(idx)
    i = idx(ii);
    if mod(ii,1000)==0
        disp(['On iteration ' num2str(ii) ' of ' num2str(length(idx))]);
    end
    j = fly.indices{i}(6:end-1);
    k = fly.indices{i}(7:end);
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
    multmax = 2*(fly.pos_o(k) > 1)-1;
    llpos_o = ...
        max(log(normpdf(fly.pos_o(k), mu_pos_o, params.pos_o.sigma)),...
        log(normpdf(fly.pos_o(k) - 2*pi*multmax, mu_pos_o, params.pos_o.sigma)));
    
    avgLL(ii) = mean(llVT + llVS + llVR + llpos_o);
    
%     if avgLL(ii) == -Inf
%         infcount = infcount + 1;
%         if infcount == 1
%             break;
%         end
%     end
    
end