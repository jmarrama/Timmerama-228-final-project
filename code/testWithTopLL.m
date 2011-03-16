%% Get (average) log-likelihoods of validation set:
disp('Calculating average log-likelihoods of validation set');
valTopLL = GetHiddenTopLL(fly, params, valIdx, trajStart);

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

% remove data that we are not going to use
% NOTE: may want to include pos_x and pos_y later!
disp('Removing unnecesssary L2 data...');
flyMutant = rmfield(flyMutant, {'tubes', 'day_times', 'pos_x', 'pos_y'});

% change data so that it is in accordance with our model assumptions:
disp('Changing L2 data according to our model assumptions...');
numMutantFlies = length(flyMutant.indices);
flyMutant.stim_RT = flyMutant.stim_RT + 1;
for i=1:2
    tmpVec = flyMutant.stim_RT(:,i);
    tmpVec(tmpVec > maxStim) = maxStim;
    flyMutant.stim_RT(:,i) = tmpVec;
end

disp('Splitting L2 into training, validation, and test data...');
[mutantTrainIdx, mutantValIdx, mutantTestIdx] = splitData(flyMutant);

mutantValTopLL = GetHiddenTopLL(flyMutant, params, mutantValIdx, trajStart);

llcuts = -200:0.1:250;
f1s = zeros(1,length(llcuts));
for i=1:length(llcuts)
    f1s(i) = EvaluateCutoff(valTopLL, mutantValTopLL, llcuts(i));
end
% plot(llcuts,f1s);
% title('F1 as a function of Average Log Likelihood Cut-offs');
% xlabel('Avg LL Cut-off');
% ylabel('F1 score');

llcut = mean(llcuts(f1s == max(f1s)))

%% Finally, testing!!
disp('Testing the Model');
testWildTopLL = GetHiddenTopLL(fly, params, testIdx, trajStart);
testmutantTopLL = GetHiddenTopLL(flyMutant, params, mutantTestIdx, trajStart);
[f1 precision recall] = EvaluateCutoff(testWildTopLL, testmutantTopLL, llcut)



