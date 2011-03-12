function [f1 precision recall] = EvaluateCutoff(avgLL, mutantAvgLL, llcut)

tp = sum(avgLL > llcut)/(sum(avgLL > llcut) + sum(mutantAvgLL > llcut));
fp = sum(mutantAvgLL > llcut)/(sum(avgLL > llcut) + sum(mutantAvgLL > llcut));
fn = sum(avgLL < llcut)/(sum(avgLL < llcut) + sum(mutantAvgLL < llcut));
% of ones that model says are normal, how many are normal?
precision = tp/(tp+fp);
% of ones that are normal, how many does model say are normal?
recall = tp/(tp+fn);
% some "average" of the two
f1 = 2*precision*recall/(precision + recall);