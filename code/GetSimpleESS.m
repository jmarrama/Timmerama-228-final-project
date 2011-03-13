%% Function to get expected sufficient statistics for simple hidden model
function gamma = GetSimpleESS(fly, params, idx, K, start, last)
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
ess.VT = zeros(len, K);
ess.VS = zeros(len, K);
ess.VR = zeros(len, K);
ess.PO = zeros(len, K);

alpha = zeros(len, K);
beta = zeros(len, K);
gamma = zeros(len, K);
scale = zeros(len,1);

VT = fly.VT(fly.indices{idx}(start-1:last));
VS = fly.VS(fly.indices{idx}(start-1:last));
VR = fly.VR(fly.indices{idx}(start-1:last));
PO = fly.pos_o(fly.indices{idx}(start-1:last));
stim_RT = fly.stim_RT(fly.indices{idx}(start-1:last),:);
% Forward Equations
alpha(1,:) = params.pi ...
    .* normpdf(VT(1), params.VT.mu, params.VT.sigma)' ...
    .* normpdf(VS(1), params.VS.mu, params.VS.sigma)' ...
    .* normpdf(VR(1), params.VR.mu, params.VR.sigma)' ...
    .* normpdf(PO(1), params.PO.mu, params.PO.sigma)';
scale(1) = sum(alpha(1,:));
alpha(1,:) = alpha(1,:) / alphascale(1);
for t=2:len
    alpha(t,:) = alpha(t-1,:)*params.stimRT{stim_RT(t,1),stim_RT(t,2)} ...
        .* normpdf(VT(t), params.VT.mu, params.VT.sigma)' ...
        .* normpdf(VS(t), params.VS.mu, params.VS.sigma)' ...
        .* normpdf(VS(t), params.VR.mu, params.VR.sigma)' ...
        .* normpdf(PO(t), params.PO.mu, params.PO.sigma)';
    scale(t) = sum(alpha(t,:));
    alpha(t,:) = alpha(t,:) / scale(t);
end
loglik = sum(log(scale));
% Backward Equations
beta(len,:) = 1;
gamma(len,:) = alpha(len,:) .* beta(len,:);
gamma(len,:) = gamma(len,:) ./ sum(gamma(len,:));
for t=len:-1:1
    beta(t,:) = beta(t+1,:) ...
        .* normpdf(VT(t+1), params.VT.mu, params.VT.sigma)' ...
        .* normpdf(VS(t+1), params.VS.mu, params.VS.sigma)' ...
        .* normpdf(VR(t+1), params.VR.mu, params.VR.sigma)' ...
        .* normpdf(PO(t+1), params.PO.mu, params.PO.sigma)' ...
        * params.stimRT{stim_RT(t+1,1), stim_RT(t+1,2)}';
    beta(t,:) = beta(t,:) / sum(beta(t,:));
    gamma(t,:) = alpha(t,:) .* beta(t,:);
    gamma(t,:) = gamma(t,:) ./ sum(gamma(t,:));
end
