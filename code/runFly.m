%% runFly.m Runs the Whole Thang

% 0 = dark
% 1 = dec
% 2 = inc
expType = 1;
maxStim = 10;
trajStart = 7;
% 0 = single connections
% 1 = fully connected
% 2 = sin and cos of pos_o, fully connected
model = 2;

% first prototype with control, decrement
% if expType==0
%     disp('Loading dark data...');
%     load ../data/control_dark_all.mat
%     fly = control_dark_strct;
%     clear control_dark_strct;
%     disp('Loading L2 dark data...');
%     load ../data/L2_dark_all.mat
%     flyMutant = L2_dark_strct;
%     clear L2_dark_strct;
% elseif expType==1
%     disp('Loading decrement data...');
%     load ../data/control_dec_all.mat
%     fly = control_dec_strct;
%     clear control_dec_strct;
%     disp('Loading L2 decrement data...');
%     load ../data/L2_dec_all.mat
%     flyMutant = L2_dec_strct;
%     clear L2_dec_strct;
% elseif expType==2
%     disp('Loading increment data...');
%     load ../data/control_inc_all.mat
%     fly = control_inc_strct;
%     clear control_inc_strct;
%     disp('Loading L2 increment data...');
%     load ../data/L2_inc_all.mat
%     flyMutant = L2_inc_strct;
%     clear L2_inc_strct;
% else
%     disp('Invalid experiment type specified');
% end


% remove data that we are not going to use
% NOTE: may want to include pos_x and pos_y later!
% disp('Removing unnecesssary data...');
% fly = rmfield(fly, {'tubes', 'day_times', 'pos_x', 'pos_y'});
% flyMutant = rmfield(flyMutant, {'tubes', 'day_times', 'pos_x', 'pos_y'});

% split into training, validation, and test data
disp('Splitting into training, validation, and test data...');
[trainIdx, valIdx, testIdx] = splitData(fly);

% evaluation = zeros(20,3);
% for maxStim=20:-1:1

% change data so that it is in accordance with our model assumptions:
disp('Changing data according to our model assumptions...');
numFlies = length(fly.indices);
fly.stim_RT = fly.stim_RT + 1;
for i=1:2
    tmpVec = fly.stim_RT(:,i);
    tmpVec(tmpVec > maxStim) = 0;
    fly.stim_RT(:,i) = tmpVec;
end


% change data so that it is in accordance with our model assumptions:
disp('Changing L2 data according to our model assumptions...');
numMutantFlies = length(flyMutant.indices);
flyMutant.stim_RT = flyMutant.stim_RT + 1;
for i=1:2
    tmpVec = flyMutant.stim_RT(:,i);
    tmpVec(tmpVec > maxStim) = 0;
    flyMutant.stim_RT(:,i) = tmpVec;
end

fly.sinPO = sin(fly.pos_o);
fly.cosPO = cos(fly.pos_o);
flyMutant.sinPO = sin(flyMutant.pos_o);
flyMutant.cosPO = cos(flyMutant.pos_o);
    
    
% Fit MLE Linear Gaussian Parameters
disp('Learning the Parameters...');
numExamples = 0;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
   numExamples = numExamples + length(fly.indices{i}) - trajStart + 1; 
end
W = ones(numExamples, 1);
disp('Learning all Parameters');
Xvt = zeros(numExamples, 1);
Xvs = zeros(numExamples, 1);
Xvr = zeros(numExamples, 1);
if model==0 || model==1
    Xpo = zeros(numExamples, 1);
elseif model==2
    XpoCos = zeros(numExamples, 1);
    XpoSin = zeros(numExamples, 1);
end
if model==0
    Uvt = zeros(numExamples, 3);
    Uvs = zeros(numExamples, 3);
    Uvr = zeros(numExamples, 3);
    Upo = zeros(numExamples, 3);
elseif model==1
    Uvt = zeros(numExamples, 7);
    Uvs = zeros(numExamples, 7);
    Uvr = zeros(numExamples, 7);
    Upo = zeros(numExamples, 6);
elseif model==2
    Uvt = zeros(numExamples, 7);
    Uvs = zeros(numExamples, 7);
    Uvr = zeros(numExamples, 7);
    Upo = zeros(numExamples, 7);
end
idx = 1;
for ii=1:length(trainIdx)
    i = trainIdx(ii);
    if mod(ii,5000)==0
        disp(['Through trajectory ' num2str(ii) ' of ' ...
            num2str(length(trainIdx)) '...']);
    end
    numSamp = length(fly.indices{i}) - trajStart + 1;
    Xvt(idx:idx+numSamp-1) = fly.VT(fly.indices{i}(trajStart:end));
    Xvs(idx:idx+numSamp-1) = fly.VS(fly.indices{i}(trajStart:end));
    Xvr(idx:idx+numSamp-1) = fly.VR(fly.indices{i}(trajStart:end));
    if model==0 || model==1
        Xpo(idx:idx+numSamp-1) = fly.pos_o(fly.indices{i}(trajStart:end));
    elseif model==2
        XpoCos(idx:idx+numSamp-1) = fly.cosPO(fly.indices{i}(trajStart:end));
        XpoSin(idx:idx+numSamp-1) = fly.sinPO(fly.indices{i}(trajStart:end));
    end
    j = fly.indices{i}(trajStart -1:end-1);
    if model==0
        Uvt(idx:idx+numSamp-1,:) = [fly.VT(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Uvs(idx:idx+numSamp-1,:) = [fly.VS(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Uvr(idx:idx+numSamp-1,:) = [fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Upo(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    elseif model==1
        Uvt(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Uvs(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Uvr(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Upo(idx:idx+numSamp-1,:) = [fly.pos_o(j), fly.VT(j), fly.VS(j), ...
            fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    elseif model==2
        Uvt(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Uvs(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Uvr(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
        Upo(idx:idx+numSamp-1,:) = [fly.cosPO(j), fly.sinPO(j), fly.VT(j), ...
            fly.VS(j), fly.VR(j), fly.stim_RT(j,1), fly.stim_RT(j,2)];
    end
    idx = idx + numSamp;
end
disp('Finally doing the learning...');
[params.VT.theta, params.VT.sigma] = FitLinearGaussianParameters(Xvt, Uvt, W);
[params.VS.theta, params.VS.sigma] = FitLinearGaussianParameters(Xvs, Uvs, W);
[params.VR.theta, params.VR.sigma] = FitLinearGaussianParameters(Xvr, Uvr, W);
if model==0 || model==1
    [params.pos_o.theta, params.pos_o.sigma] = FitLinearGaussianParameters(Xpo, Upo, W);
elseif model==2
    [params.cosPO.theta, params.cosPO.sigma] = ...
        FitLinearGaussianParameters(XpoCos, Upo, W);
    [params.sinPO.theta, params.sinPO.sigma] = ...
        FitLinearGaussianParameters(XpoSin, Upo, W);
end

clear Uvt Uvs Uvr Upo Xvt Xvs Xvr Xpo;

%% Get (average) log-likelihoods of validation set:
disp('Calculating average log-likelihoods of validation set');
valAvgLL = GetAvgLLs(fly, params, valIdx, trajStart, model);



disp('Splitting L2 into training, validation, and test data...');
[mutantTrainIdx, mutantValIdx, mutantTestIdx] = splitData(flyMutant);

mutantAvgLL = GetAvgLLs(flyMutant, params, mutantValIdx, trajStart, model);

llcuts = -20:0.1:15;
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
testWildAvgLL = GetAvgLLs(fly, params, testIdx, trajStart, model);
testMutantAvgLL = GetAvgLLs(flyMutant, params, mutantTestIdx, trajStart, model);
[f1 precision recall] = EvaluateCutoff(testWildAvgLL, testMutantAvgLL, llcut);


% evaluation(maxStim,:) = [f1 precision recall];
% end

% save(['../data/exp' num2str(expType) 'eval.mat'], 'evaluation');

