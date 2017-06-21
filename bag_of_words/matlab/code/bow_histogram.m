function histogram = bow_histogram(vFeatures, vCenters)
% Generate the histogram with for the bag of words which is given by the
% centers and the found features.
%
% Input
%   vFeatures       MxD matrix containing M feature vectors of dim. D
%   vCenters        NxD matrix containing N cluster centers of dim. D
% 
% Output
%   histogram       N-dim. vector containing the resulting BoW activation
%                   histogram.

%% Initialization
nClusters = size(vCenters, 1);
histogram = zeros(nClusters, 1);

%% Histogram generation
% Match all features to the codebook and record the activated
% codebook entries in the activation histogram.
[clusters, ~] = findnn(vFeatures, vCenters);

for c = 1:size(vCenters,1)
     histogram(c) = size(vFeatures(clusters == c,:), 1);
end

end
