%% runHiddenFly.m Runs the Whole Thang

% 0 = dark
% 1 = dec
% 2 = inc
expType = 2;

% number of hidden states (i.e. cardinality of hidden variable)
numStates = 20;
maxStim = 16;

% first prototype with control, decrement
if expType==0
    disp('Loading dark data...');
    load ../data/control_dark_all.mat
    fly = control_dark_strct;
    clear control_dark_strct;
elseif expType==1
    disp('Loading decrement data...');
    load ../data/control_dec_all.mat
    fly = control_dec_strct;
    clear control_dec_strct;
elseif expType==2
    disp('Loading increment data...');
    load ../data/control_inc_all.mat
    fly = control_inc_strct;
    clear control_inc_strct;
else
    disp('Invalid experiment type specified');
end


% remove data that we are not going to use
% NOTE: may want to include pos_x and pos_y later!
disp('Removing unnecesssary data...');
fly = rmfield(fly, {'tubes', 'day_times', 'pos_x', 'pos_y'});

% split into training, validation, and test data
disp('Splitting into training, validation, and test data...');
[trainIdx, valIdx, testIdx] = splitData(fly);

% change data so that it is in accordance with our model assumptions:
disp('Changing data according to our model assumptions...');
numFlies = length(fly.indices);
fly.stim_RT = fly.stim_RT + 1;
for i=1:2
    tmpVec = fly.stim_RT(:,i);
    tmpVec(tmpVec > maxStim) = maxStim;
    fly.stim_RT(:,i) = tmpVec;
end

% Fit MLE Linear Gaussian Parameters
disp('Performing EM to Learn Parameters');
numExamples = 0;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
   numExamples = numExamples + length(fly.indices{i}) - 6; 
end
params.VT.mu = rand(numStates,1)-0.5;
params.VS.mu = rand(numStates,1)-0.5;
params.VR.mu = rand(numStates,1)-0.5;
params.PO.mu = rand(numStates,1)-0.5;
params.VT.sigma = 0.5*(1+rand(numStates,1));
params.VS.sigma = 0.5*(1+rand(numStates,1));
params.VR.sigma = 0.5*rand(numStates,1);
params.PO.sigma = 0.5*rand(numStates,1);
for i=1:maxStim
    for j=1:maxStim
        params.stimRT{i,j} = rand(numStates);
        params.stimRT{i,j} = params.stimRT{i,j} ./ ...
            repmat(sum(params.stimRT{i,j},2),1,numStates);
    end 
end
params.pi = rand(1,numStates);
params.pi = params.pi ./ sum(params.pi);
for iter=1:100
    disp(['Iteration ' num2str(iter) ' of 100...']);
    W = zeros(numExamples, numStates);
    idx = 1;
    
    disp('E-Step...');
    for ii=1:length(trainIdx)
        i = trainIdx(ii);
        trajLen = length(fly.indices{i});
        numSamp = length(fly.indices{i}) - 6;
        Xvt(idx:idx+numSamp-1) = fly.VT(fly.indices{i}(7:end));
        Xvs(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(7:end));
        Xvr(idx:idx+numSamp-1) = fly.VR(fly.indices{i}(7:end));
        Xpo(idx:idx+numSamp-1) = fly.pos_o(fly.indices{i}(7:end));
        W(idx:idx+numSamp-1,:) = GetSimpleESS(fly, params, i, 7, trajLen);
        idx = idx + numSamp;
    end
    
    disp('M-Step...');
    for k=1:numStates
        [params.VT.mu(k) params.VT.sigma(k)] = ...
            FitGaussianParameters(Xvt, W(:,k));
        [params.VS.mu(k) params.VS.sigma(k)] = ...
            FitGaussianParameters(Xvs, W(:,k));
        [params.VR.mu(k) params.VR.sigma(k)] = ...
            FitGaussianParameters(Xvr, W(:,k));
        [params.PO.mu(k) params.PO.sigma(k)] = ...
            FitGaussianParameters(Xpo, W(:,k));
    end
end


%% Get (average) log-likelihoods of validation set:
disp('Calculating average log-likelihoods of validation set');
valAvgLL = GetAvgLLs(fly, params, valIdx);

%% LL cut-off with L2...
if expType==0
    disp('Loading L2 dark data...');
    load ../data/L2_dark_all.mat
    flyMutant = L2_dark_strct;
    clear L2_dark_strct;
elseif expType==1
    disp('Loading L2 decrement data...');
    load ../data/L2_dec_all.mat
    flyMutant = L2_dec_strct;
    clear L2_dec_strct;
elseif expType==2
    disp('Loading L2 increment data...');
    load ../data/L2_inc_all.mat
    flyMutant = L2_inc_strct;
    clear L2_inc_strct;
else
    disp('Invalid experiment type specified');
end
disp('Splitting L2 into training, validation, and test data...');
[mutantTrainIdx, mutantValIdx, mutantTestIdx] = splitData(flyMutant);

mutantAvgLL = GetAvgLLs(flyMutant, params, mutantValIdx);

llcuts = -10:0.1:10;
f1s = zeros(1,length(llcuts));
for i=1:length(llcuts)
    f1s(i) = EvaluateCutoff(valAvgLL, mutantAvgLL, llcuts(i));
end
plot(llcuts,f1s);
title('F1 as a function of Average Log Likelihood Cut-offs');
xlabel('Avg LL Cut-off');
ylabel('F1 score');

llcut = llcuts(f1s == max(f1s));

%% Finally, testing!!
disp('Testing the Model');
testWildAvgLL = GetAvgLLs(fly, params, testIdx);
testMutantAvgLL = GetAvgLLs(flyMutant, params, mutantTestIdx);
[f1 precision recall] = EvaluateCutoff(testWildAvgLL, testMutantAvgLL, llcut);





