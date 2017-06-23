function [index, dist] = findnn(descriptors1, descriptors2 )
% Find for each descriptor in descriptors1 the closest descriptor in
% descriptors2.
%
% Input
%   descriptors1    NxD matrix containing N feature vectors of dim. D
%   descriptors2    MxD matrix containing M feature vectors of dim. D
%
% Output:
%   index           N-dim. vector containing for each feature vector in D1
%                   the index of the closest feature vector in D2.
%   dist            N-dim. vector containing for each feature vector in D1
%                   the distance to the closest feature vector in D2.

%% Initialization
n = size(descriptors1,1);
index  = zeros(n,1);
dist = zeros(n,1);

%% Find for each feature vector in D1 the nearest neighbor in D2

% Calculate the distances for each descriptor in D1 to all descriptors in
% D2. Get then the index and the distance of the closest.
for i = 1:n
    % Get next descriptor in image one
    current = descriptors1(i,:);
    
    % Calculate the difference between the current descriptor and all
    % descriptors of set 2. Then square it and sum it correctly.
    % After these steps the SSD between the descriptor of set 1 and
    % all of set 2 is generated.
    ssd = (bsxfun(@minus, descriptors2, current)).^2;
    ssd = sum(ssd,2);
    minDist = min(ssd);
    closest = find(ssd == minDist);
    
    % Store the closest index and its distance (euclidean distance)
    index(i) = closest(1);
    dist(i) = minDist;
end

end
