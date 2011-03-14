%% Function to get expected sufficient statistics for simple hidden model
function [gamma xi_summed loglik] = ...
    GetSimpleESS(fly, params, idx, K, start, last, maxStim)
% idx = index of current trajectory
% K = cardinality of hidden states
% params:
%   params.pi = 1xK vector (prior on states)
%   params.VT/VS/VR/PO.mu = Kx1 vector
%   params.VT/VS/VR/PO.sigma = Kx1 vector
%   params.S
%   params.stimRT{i,j} = KxK vector,
%       stim_RT(m,:) = [i j]
%       (k,l) = p(s(t)=l|s(t-1)=k)
% OUTPUT:
%   gamma = TxK vector, T time steps, K values of hidden state
%       gamma(t,k) = prob of kth value at time step t
           

len = last-start+1;
% ess.VT = zeros(len, K);
% ess.VS = zeros(len, K);
% ess.VR = zeros(len, K);
% ess.PO = zeros(len, K);

alpha = zeros(len, K);
beta = zeros(len, K);
gamma = zeros(len, K);
scale = zeros(len,1);
xi_summed = cell(maxStim,maxStim);
for i=1:maxStim
    for j=1:maxStim
        xi_summed{i,j} = zeros(K, K);
    end
end


VT = fly.VT(fly.indices{idx}(start-1:last));
VS = fly.VS(fly.indices{idx}(start-1:last));
VR = fly.VR(fly.indices{idx}(start-1:last));
PO = fly.pos_o(fly.indices{idx}(start-1:last));
stim_RT = fly.stim_RT(fly.indices{idx}(start-1:last),:);
% TxK matrix, (t,k) is likelihood of t-th observation set given hidden
% variable value is k
obslik = GetObsLik(params, VT, VS, VR, PO);
% Forward Equations
alpha(1,:) = params.pi .* obslik(1,:);
scale(1) = sum(alpha(1,:));
alpha(1,:) = alpha(1,:) / scale(1);
for t=2:len
    alpha(t,:) = alpha(t-1,:)*params.stimRT{stim_RT(t,1),stim_RT(t,2)} ...
        .* obslik(t,:);
    [alpha(t,:) scale(t)] = normaliseC(alpha(t,:));
%     scale(t) = sum(alpha(t,:));
%     alpha(t,:) = alpha(t,:) / scale(t);
end
loglik = sum(log(scale));
% Backward Equations
beta(len,:) = 1;
gamma(len,:) = alpha(len,:) .* beta(len,:);
gamma(len,:) = gamma(len,:) ./ sum(gamma(len,:));
for t=len-1:-1:1
    beta(t,:) = normaliseC(beta(t+1,:) .* obslik(t+1,:) ...
        * params.stimRT{stim_RT(t+1,1), stim_RT(t+1,2)}');
%     beta(t,:) = beta(t+1,:) .* obslik(t+1,:) ...
%         * params.stimRT{stim_RT(t+1,1), stim_RT(t+1,2)}';
%     beta(t,:) = beta(t,:) / sum(beta(t,:));
    gamma(t,:) = normaliseC(alpha(t,:) .* beta(t,:));
%     gamma(t,:) = alpha(t,:) .* beta(t,:);
%     gamma(t,:) = gamma(t,:) ./ sum(gamma(t,:));
    i = stim_RT(t+1,1);
    j = stim_RT(t+1,2);
    xi_temp = mk_stochastic(params.stimRT{i,j} .* ...
        (alpha(t,:)' * (obslik(t+1,:) .* beta(t,:))));
    xi_summed{i,j} = xi_summed{i,j} + xi_temp;
%     for k=1:K
%         xi_temp = alpha(t,k) * params.stimRT{i,j}(k,:) .* obslik(t+1,:) ...
%             .* beta(t,:);
%         if any(abs(xi_temp - xi_temp_mat(k,:)) > 1e-3)
%             disp('xi_temp effed');
%         end
%         for l=1:K
%             xi_temp(l) = alpha(t,k) * params.stimRT{i,j}(k,l) ...
%                 * normpdf(VT(t+1), params.VT.mu(l), params.VT.sigma(l)) ...
%                 * normpdf(VS(t+1), params.VS.mu(l), params.VS.sigma(l)) ...
%                 * normpdf(VR(t+1), params.VR.mu(l), params.VR.sigma(l)) ...
%                 * normpdf(PO(t+1), params.PO.mu(l), params.PO.sigma(l)) ...
%                 * beta(t,l);
%         end
%         xi_summed{i,j}(k,:) = xi_summed{i,j}(k,:) + normaliseC(xi_temp);
%         xi_summed{i,j}(k,:) = xi_summed{i,j}(k,:) + xi_temp ./ sum(xi_temp);
%     end
end
