%% runFly.m Runs the Whole Thang

% 0 = dark
% 1 = dec
% 2 = inc
expType = 2;
maxStim = 1;
trajStart = 7;

% first prototype with control, decrement
if expType==0
    disp('Loading dark data...');
    load ../data/control_dark_all.mat
    fly = control_dark_strct;
    clear control_dark_strct;
    disp('Loading L2 dark data...');
    load ../data/L2_dark_all.mat
    flyMutant = L2_dark_strct;
    clear L2_dark_strct;
elseif expType==1
    disp('Loading decrement data...');
    load ../data/control_dec_all.mat
    fly = control_dec_strct;
    clear control_dec_strct;
    disp('Loading L2 decrement data...');
    load ../data/L2_dec_all.mat
    flyMutant = L2_dec_strct;
    clear L2_dec_strct;
elseif expType==2
    disp('Loading increment data...');
    load ../data/control_inc_all.mat
    fly = control_inc_strct;
    clear control_inc_strct;
    disp('Loading L2 increment data...');
    load ../data/L2_inc_all.mat
    flyMutant = L2_inc_strct;
    clear L2_inc_strct;
else
    disp('Invalid experiment type specified');
end


% remove data that we are not going to use
% NOTE: may want to include pos_x and pos_y later!
disp('Removing unnecesssary data...');
fly = rmfield(fly, {'tubes', 'day_times', 'pos_x', 'pos_y'});
flyMutant = rmfield(flyMutant, {'tubes', 'day_times', 'pos_x', 'pos_y'});

% split into training, validation, and test data
disp('Splitting into training, validation, and test data...');
[trainIdx, valIdx, testIdx] = splitData(fly);

% change data so that it is in accordance with our model assumptions:
disp('Changing data according to our model assumptions...');
numFlies = length(fly.indices);
fly.stim_RT = fly.stim_RT + 1;
for i=1:2
    tmpVec = fly.stim_RT(:,i);
    tmpVec(tmpVec > maxStim) = 0;
    fly.stim_RT(:,i) = tmpVec;
end

evaluation = zeros(20,3);
for maxStim=1:20
% change data so that it is in accordance with our model assumptions:
disp('Changing L2 data according to our model assumptions...');
numMutantFlies = length(flyMutant.indices);
flyMutant.stim_RT = flyMutant.stim_RT + 1;
for i=1:2
    tmpVec = flyMutant.stim_RT(:,i);
    tmpVec(tmpVec > maxStim) = 0;
    flyMutant.stim_RT(:,i) = tmpVec;
end

% Fit MLE Linear Gaussian Parameters
disp('Learning the Parameters...');
numExamples = 0;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
   numExamples = numExamples + length(fly.indices{i}) - trajStart + 1; 
end
W = ones(numExamples, 1);
disp('Learning VT Parameters');
X = zeros(numExamples, 1);
%U = zeros(numExamples, 7);
U = zeros(numExamples, 3);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - trajStart + 1;
    X(idx:idx+numSamp-1) = fly.VT(fly.indices{i}(trajStart:end));
    j = fly.indices{i}(trajStart -1:end-1);
    U(idx:idx+numSamp-1,:) = [fly.VT(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
%     U(idx:idx+numSamp-1,:) = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), ...
%         fly.VT(j), fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.VT.theta, params.VT.sigma] = FitLinearGaussianParameters(X, U, W);
disp('Learning VS Parameters');
X = zeros(numExamples, 1);
U = zeros(numExamples, 3);
% U = zeros(numExamples, 7);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - trajStart + 1;
    X(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(trajStart:end));
    j = fly.indices{i}(trajStart - 1:end-1);
    U(idx:idx+numSamp-1,:) = [fly.VS(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
%     U(idx:idx+numSamp-1,:) = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), ...
%         fly.VT(j), fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.VS.theta, params.VS.sigma] = FitLinearGaussianParameters(X, U, W);
disp('Learning VR Parameters');
X = zeros(numExamples, 1);
U = zeros(numExamples, 3);
% U = zeros(numExamples, 7);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - trajStart + 1;
    X(idx:idx+numSamp-1) = fly.VR(fly.indices{i}(trajStart:end));
    j = fly.indices{i}(trajStart - 1:end-1);
    U(idx:idx+numSamp-1,:) = [fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
%     U(idx:idx+numSamp-1,:) = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), ...
%         fly.VT(j), fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.VR.theta, params.VR.sigma] = FitLinearGaussianParameters(X, U, W);
disp('Learning pos_o Parameters');
X = zeros(numExamples, 1);
% U = zeros(numExamples, 6);
U = zeros(numExamples, 3);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - trajStart + 1;
    X(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(trajStart:end));
    j = fly.indices{i}(trajStart - 1:end-1);
    U(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
%     U(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.VT(j), fly.VS(j), ...
%         fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.pos_o.theta, params.pos_o.sigma] = FitLinearGaussianParameters(X, U, W);


%% Get (average) log-likelihoods of validation set:
disp('Calculating average log-likelihoods of validation set');
valAvgLL = GetAvgLLs(fly, params, valIdx, trajStart);



disp('Splitting L2 into training, validation, and test data...');
[mutantTrainIdx, mutantValIdx, mutantTestIdx] = splitData(flyMutant);

mutantAvgLL = GetAvgLLs(flyMutant, params, mutantValIdx, trajStart);

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
testWildAvgLL = GetAvgLLs(fly, params, testIdx, trajStart);
testMutantAvgLL = GetAvgLLs(flyMutant, params, mutantTestIdx, trajStart);
[f1 precision recall] = EvaluateCutoff(testWildAvgLL, testMutantAvgLL, llcut);


evaluation(maxStim,:) = [f1 precision recal];
end


