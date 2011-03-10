%% runFly.m Runs the Whole Thang

% first prototype with control, decrement
disp('Loading data...');
load ../data/control_dec_all.mat
fly = control_dec_strct;
clear control_dec_strct;

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
    tmpVec(tmpVec > 15) = 0;
    fly.stim_RT(:,i) = tmpVec;
end

% Fit MLE Linear Gaussian Parameters
disp('Learning the Parameters...');
numExamples = 0;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
   numExamples = numExamples + length(fly.indices{i}) - 1; 
end
W = ones(numExamples, 1);
disp('Learning VT Parameters');
X = zeros(numExamples, 1);
U = zeros(numExamples, 7);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - 1;
    X(idx:idx+numSamp-1) = fly.VT(fly.indices{i}(2:end));
    j = fly.indices{i}(1:end-1);
    U(idx:idx+numSamp-1,:) = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), ...
        fly.VT(j), fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.VT.theta, params.VT.sigma] = FitLinearGaussianParameters(X, U, W);
disp('Learning VS Parameters');
X = zeros(numExamples, 1);
U = zeros(numExamples, 7);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - 1;
    X(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(2:end));
    j = fly.indices{i}(1:end-1);
    U(idx:idx+numSamp-1,:) = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), ...
        fly.VT(j), fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.VS.theta, params.VS.sigma] = FitLinearGaussianParameters(X, U, W);
disp('Learning VR Parameters');
X = zeros(numExamples, 1);
U = zeros(numExamples, 7);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - 1;
    X(idx:idx+numSamp-1) = fly.VR(fly.indices{i}(2:end));
    j = fly.indices{i}(1:end-1);
    U(idx:idx+numSamp-1,:) = [cos(fly.pos_o(j)), sin(fly.pos_o(j)), ...
        fly.VT(j), fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.VR.theta, params.VR.sigma] = FitLinearGaussianParameters(X, U, W);
disp('Learning pos_o Parameters');
X = zeros(numExamples, 1);
U = zeros(numExamples, 6);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - 1;
    X(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(2:end));
    j = fly.indices{i}(1:end-1);
    U(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.VT(j), fly.VS(j), ...
        fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.pos_o.theta, params.pos_o.sigma] = FitLinearGaussianParameters(X, U, W);





