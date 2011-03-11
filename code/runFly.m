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
%U = zeros(numExamples, 7);
U = zeros(numExamples, 3);
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    numSamp = length(fly.indices{i}) - 6;
    X(idx:idx+numSamp-1) = fly.VT(fly.indices{i}(7:end));
    j = fly.indices{i}(6:end-1);
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
    numSamp = length(fly.indices{i}) - 6;
    X(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(7:end));
    j = fly.indices{i}(6:end-1);
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
    numSamp = length(fly.indices{i}) - 6;
    X(idx:idx+numSamp-1) = fly.VR(fly.indices{i}(7:end));
    j = fly.indices{i}(6:end-1);
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
    numSamp = length(fly.indices{i}) - 6;
    X(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(7:end));
    j = fly.indices{i}(6:end-1);
    U(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
%     U(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.VT(j), fly.VS(j), ...
%         fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    idx = idx + numSamp;
end
[params.pos_o.theta, params.pos_o.sigma] = FitLinearGaussianParameters(X, U, W);


%% Get (average) log-likelihoods of validation set:
disp('Calculating average log-likelihoods of validation set');
valAvgLL = zeros(length(valIdx),1);
% infcount = 0;
for ii=1:length(valIdx)
    i = valIdx(ii);
    ll = 0;
    if mod(ii,1000)==0
        disp(['On iteration ' num2str(ii) ' of ' num2str(length(valIdx))]);
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
    llpos_o = log(normpdf(fly.pos_o(k), mu_pos_o, params.pos_o.sigma));
    
    valAvgLL(ii) = mean(llVT + llVS + llVR + llpos_o);
    
%     if valAvgLL(ii) == -Inf
%         infcount = infcount + 1;
%         if infcount == 1
%             break;
%         end
%     end
    
end

