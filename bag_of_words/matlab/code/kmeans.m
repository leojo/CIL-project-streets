function vCenters = kmeans(vFeatures, k, numiter)
% Generate k clusters from all the found features by iterating
% 'numiter'-times. It might be that after the given number of iterations
% might not be sufficient to find an optimal solution since it is not
% checked how much the means are chaning between each step.
%
% Input
%   vFeatures       NxD matrix containing N feature vectors of dim. D
%   k               Desired amount of cluster-centers.
%   numiter         Amount of iterations to update the cluster-centers.
%
% Output:
%   vCenters        kxD matrix containing the cluter-centers.

%% Initialization
nPoints  = size(vFeatures, 1);

% Initialize each cluster center to a different random point.
vCenters = vFeatures(randi(nPoints, k, 1),:);

%% Clustering
% Repeat for numiter iterations
for i = 1:numiter,
    % Assign each point to the closest cluster
    [clusters, distances] = findnn(vFeatures, vCenters);
        
    % Shift each cluster center to the mean of its assigned points
    for c = 1:k
       vCenters(c, :) = mean(vFeatures(clusters == c,:)); 
    end
        
    disp(strcat(num2str(i),'/',num2str(numiter),' iterations completed.'));
end;


end
