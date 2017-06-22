function vPoints = grid_points(img, nPointsX, nPointsY, border)
% Generate the feature points with the help of a grid. The border is used
% as a margin for th grid (so that there is enough space for the patches
% with the 16 cells)
%
% Input
%   img             The image for which the grid has to be generated.
%   nPointsX        Number of points on the x-axis.
%   nPointsY        Number of points on the y-axis.
%   border          Size of the border for the grid.
% 
% Output
%   vPoints         (nPointsX*nPointsY)x2 matrix which contains all the
%                   corder-points from the grid.

%% Initialization
[h, w] = size(img);

% Make sure there is enough space for the surrounding cells later. Narrow
% down the size.
border = border + 1;
h = h - 2*border;
w = w - 2*border;

%% Grid-generation
vPoints = zeros(2, nPointsX * nPointsY);
i = 0;

for x = border:w/(nPointsX-1):w+border
    vPoints(:, i*nPointsX+1:i*nPointsX+nPointsY) = [
        ones(1,nPointsY) * x;
        border:h/(nPointsY-1):h+border
        ];
    i = i + 1;
end

% Convert the 2xN matrix to a Nx2 matrix.
vPoints = vPoints';
end