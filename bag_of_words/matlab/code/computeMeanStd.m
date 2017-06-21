function [mu, sigma] = computeMeanStd(vBoW)
% Calculate the mean and standard deviation for all features in the NxD
% matrix (N = number of features, D = dimension of the features).
%
% Input
%   vBoWNeg     NxD matrix (N = number of features, D = dimension of the
%               features)
%
% Output:
%   mu          Mean of the features.
%   sigma       Standard deviation of the features.

%% Calculations
mu = mean(vBoW, 1);
sigma = sqrt(std(vBoW, 1));
end