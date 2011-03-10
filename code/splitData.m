%% splitData.m Splits Data into Training, Validation, and Test Data!'
% @param {struct} fly: fly trajectories data structure
function [trainIdx, valIdx, testIdx] = splitData(fly)

% set the seed each time for reproducibility 
rand('seed', 1);

% split is 60-20-20
numFlies = length(fly.indices);
numTrain = floor(0.6 * numFlies);
numVal = floor(0.8*numFlies) - numTrain;

trainIdx = randsample(numFlies, numTrain);
notTrainIdx = setdiff(1:numFlies, trainIdx);
valIdx = randsample(notTrainIdx, numVal);
testIdx = setdiff(notTrainIdx, valIdx);