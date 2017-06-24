function [y] = descriptors_hog(img, vPoints, nBins, cellWidth, cellHeight)
% Generate a descriptor based on the found feature points. The descriptor
% is a histogram of oriented gradients and is generated by a 4x4-cell grid,
% each cell with the specified size.
%
% Input
%   img             The image for which the descriptors and pathces are
%                   generated.
%   vPoints         The feature points for which the descriptors are
%                   generated.
%   cellWidth       Width of the cell used for the 4x4-cells.
%   cellHeight      Height of the cell used for the 4x4-cells.
% 
% Output
%   descriptors     Nx128 matrix which contain the hog-descriptor for each
%                   feature. (N = number of feature points)
%   patches         Nx(16*cellWidth*cellHeight) matrix which represents the
%                   patch which fits to the descriptor.

%% Initialization
w = cellWidth;
h = cellHeight;

% Initialize output
y = zeros(size(vPoints,1), 1);

%% Generate descriptors
for i = 1:size(vPoints,1)
    current = round(vPoints(i,:));
    
    % extract the patch from the image
    localPatch = img((current(2)-2*h):(current(2)+2*h-1), (current(1)-2*w):(current(1)+2*w-1), :);
    
    if mean(mean(localPatch)) >= 255/2
        y(i) = 1;
    else
        y(i) = 0;
    end
end

end