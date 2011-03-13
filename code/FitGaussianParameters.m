% CS228 PA3 Winter 2011
% File: FitGaussianParameters.m
% Copyright (C) 2011, Stanford University
% contact: Huayan Wang, huayanw@cs.stanford.edu

function [mu sigma] = FitGaussianParameters(X, W)

% X: (N x 1): N examples (1 dimensional)
% W: (N x 1): their weights

% Fit N(mu, sigma^2) to the empirical distribution

mu = 0;
sigma = 1;

% YOUR CODE HERE
N = length(X);
% mean = sum_i w_i X_i (w_i = W_i/sum(W))
mu = (W'*X)/sum(W);
% var = sum_i w_i X_i^2 - (sum_i w_i X_i)^2
sigma = sqrt((W'*(X.^2))/sum(W) - mu^2);
