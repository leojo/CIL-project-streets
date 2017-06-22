% Reconstruction of image from Laplacian pyramid
%
% Arguments:
%   pyramid 'pyr', as generated by function 'laplacian_pyramid'
%   subwindow indices 'subwindow', given as [r1 r2 c1 c2] (optional) 
%
% tom.mertens@gmail.com, August 2007
% sam.hasinoff@gmail.com, March 2011  [modified to handle subwindows]
%
%
% More information:
%   'The Laplacian Pyramid as a Compact Image Code'
%   Burt, P., and Adelson, E. H., 
%   IEEE Transactions on Communication, COM-31:532-540 (1983). 
%

function R = reconstruct_laplacian_pyramid(pyr,subwindow)

r = size(pyr{1},1);
c = size(pyr{1},2);
nlev = length(pyr);

subwindow_all = zeros(nlev,4);
if ~exist('subwindow','var')
    subwindow_all(1,:) = [1 r 1 c];
else
    subwindow_all(1,:) = subwindow;
end
for lev = 2:nlev
    subwindow_all(lev,:) = child_window(subwindow_all(lev-1,:));
end

% start with low pass residual
R = pyr{nlev};
filter = pyramid_filter;
for lev = nlev-1 : -1 : 1
    % upsample, and add to current level
    R = pyr{lev} + upsample(R,filter,subwindow_all(lev,:));
end
