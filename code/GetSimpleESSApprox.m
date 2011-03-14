%% Function to get ESS for simple model w/ particle filtering
function [PX ess_trans] = ...
    GetSimpleESSApprox(fly, params, idx, K, start, last, maxStim, M)
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
           

T = last-start+1;

ess_trans = cell(maxStim,maxStim);
for i=1:maxStim
    for j=1:maxStim
        ess_trans{i,j} = zeros(K, K);
    end
end

VT = fly.VT(fly.indices{idx}(start:last));
VS = fly.VS(fly.indices{idx}(start:last));
VR = fly.VR(fly.indices{idx}(start:last));
PO = fly.pos_o(fly.indices{idx}(start:last));
stim_RT = fly.stim_RT(fly.indices{idx}(start:last),:);

% TxK matrix, (t,k) is likelihood of t-th observation set given hidden
% variable value is k
obslik = GetObsLik(params, VT, VS, VR, PO);

% hidden sequence X (will grow to MxT)
Xprev = randsample(K, M, true, params.pi);
% W represents weights - Mx1
Wprev = normaliseC(params.pi(Xprev))';
for t=2:T
    Xidx = randsample(M, M, true, Wprev);
    X = zeros(M,t);
    X(:,1:t-1) = Xprev(Xidx,:);
    i = stim_RT(t,1);
    j = stim_RT(t,2);
    W = Wprev(Xidx);
    for k=1:K
        kIdx = X(:,t-1) == k;
        numK = sum(kIdx);
        if numK > 0
            X(kIdx,t) = randsample(K, numK, true, params.stimRT{i,j}(k,:));
        end
    end
%     for m=1:M
%         X(m,t) = randsample(K, 1, true, params.stimRT{i,j}(X(m,t-1),:));
%     end
    stimRTidx = K*(X(:,t-1)-1) + X(:,t);
    W = normaliseC(W .* params.stimRT{i,j}(stimRTidx) .* obslik(t,X(:,t))');
    Xprev = X;
    Wprev = W;
end

PX = zeros(T,K);
for t=1:T
    for k=1:K
        PX(t,k) = sum(W(X(:,t)==k));
    end
end

for t=1:T-1
    i = stim_RT(t+1,1);
    j = stim_RT(t+1,2);
    for m=1:M
        ess_trans{i,j}(X(m,t),X(m,t+1)) = ...
            ess_trans{i,j}(X(m,t),X(m,t+1)) + W(m);
    end
end
