%% Function to compute observation probabilities for each state and time step
function obslik = GetObsLik(params, VT, VS, VR, PO);

K = length(params.VT.mu);
T = length(VT);

obslik = zeros(T,K);
for t=1:T
    obslik(t,:) = ...
        normpdf(VT(t), params.VT.mu, params.VT.sigma)' ...
        .* normpdf(VS(t), params.VS.mu, params.VS.sigma)' ...
        .* normpdf(VR(t), params.VR.mu, params.VR.sigma)' ...
        .* normpdf(PO(t), params.PO.mu, params.PO.sigma)';
end