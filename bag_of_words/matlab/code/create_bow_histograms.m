function [vBoWP, vBoWN] = create_bow_histograms(nameDir, nameGround, vCenters, fs)
% Create all bag-of-word-histograms for each picture in the given
% directory. To generate these, the vCenters are used from the clustering
% process of the features from the positive samples. To extract the
% features, the settings in fs are used.
%
% Input
%   nameDir         Relative path to the directory which contains the
%                   positive training pictures.
%   vCenters        Cluster-centers from the k-means.
%   fs              Contains all the settings (as cellWidth, ...) which are
%                   used for the feature extraction steps.
%
% Output:
%   vBoW            NxD matrix containing the histograms for all the
%                   images. N = number of images in the directory.

%% Initialization
vImgNames = dir(fullfile(nameDir,'*.png'));
nImgs = length(vImgNames);
vBoWP = zeros(nImgs, size(vCenters,1));
vBoWN = zeros(nImgs, size(vCenters,1));

%% Extract features for all images
for i = 1:nImgs
    disp(strcat('  Processing image ', num2str(i),'...'));
    
    % Load the image
    img = double(rgb2gray(imread(fullfile(nameDir,vImgNames(i).name))));
    imgG = double(imread(fullfile(nameGround,vImgNames(i).name)));
    
    % Collect local feature points for each image and compute a hog
    % descriptor and patch for each local feature point.
    vPoints = grid_points(img, fs.nPointsX, fs.nPointsY, fs.border);
    [descriptors, ~] = descriptors_hog(img, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    [y] = descriptors_hog_y(imgG, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    
    descriptorsP = descriptors(find(y >= 0.5), :);
    descriptorsN = descriptors(find(y <  0.5), :);
    
    % Create a BoW activation histogram for this image
    vBoWP(i, :) = bow_histogram(descriptorsP, vCenters)';
    vBoWN(i, :) = bow_histogram(descriptorsN, vCenters)';
end

end