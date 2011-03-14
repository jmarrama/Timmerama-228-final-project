%% Returns average log-likelihoods for given fly trajectories (Viterbi)
function avgLL = GetHiddenAvgLLs(fly, params, idx, start)

% cardinality of hidden variable
K = length(params.pi);

avgLL = zeros(length(idx),1);
% infcount = 0;
for ii=1:length(idx)
    
    if mod(ii,1000)==0
        disp(['On iteration ' num2str(ii) ' of ' num2str(length(idx))]);
    end
    
    i = idx(ii);
    
    VT = fly.VT(fly.indices{i}(start-1:end));
    VS = fly.VS(fly.indices{i}(start-1:end));
    VR = fly.VR(fly.indices{i}(start-1:end));
    PO = fly.pos_o(fly.indices{i}(start-1:end));
    stim_RT = fly.stim_RT(fly.indices{i}(start-1:end),:);
    % TxK matrix, (t,k) is likelihood of t-th observation set given hidden
    % variable value is k
    obslik = GetObsLik(params, VT, VS, VR, PO);
    
    T = length(VT);
    scale = zeros(1,T);
    delta = zeros(T,K);
    psi = zeros(T,K);
%     path = zeros(1,T);
    
    t=1;
    [delta(t,:) n] = normaliseC(params.pi .* obslik(t,:));
    scale(t) = 1/n;
    
    for t=2:T
        for k=1:K
            j = stim_RT(t,1);
            l = stim_RT(t,2);
            [delta(t,k) psi(t,k)] = ...
                max(delta(t-1,:) .* params.stimRT{j,l}(:,k)');
            delta(t,k) = delta(t,k) * obslik(t,k);
        end
        [delta(t,:) n] = normaliseC(delta(t,:));
        scale(t) = 1/n;
    end
    
%     [p path(T)] = max(delta(T,:));
%     for t=T-1:-1:1
%         path(t) = psi(t+1,path(t+1));
%     end
    
    avgLL(ii) = -mean(log(scale));
    
end