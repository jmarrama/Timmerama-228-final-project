function x = lograndsample(vec, N, logPs)

maxLogP = max(logPs);
logPsCumSum = maxLogP + log(cumsum(exp(logPs - maxLogP)));

x = zeros(N,1);
for n=1:N
    logP = log(rand);
    for i=1:length(logPsCumSum)
        if logP < logPsCumSum(i)
            break;
        end
    end
    x(n) = vec(i);
end

