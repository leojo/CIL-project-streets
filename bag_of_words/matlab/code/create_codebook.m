function vCenters = create_codebook(nameDir, nameGround, k, numiter, fs)
% Create a codebook from the images which are located in the specified
% directory. K and numiter are used to find k clusters in
% numiter-iterations with the k-means. Fs contains the settings to extract
% the feature-points and its descriptors.
%
% Input
%   nameDir         Relative path to the directory which contains the
%                   positive training pictures.
%   k               Desired amount of cluster-centers.
%   numiter         Amount of iterations to update the cluster-centers.
%   fs              Contains all the settings (as cellWidth, ...) which are
%                   used for the feature extraction steps.
%
% Output:
%   vCenters        kxD matrix containing the cluter-centers of the
%                   codebook.

%% Initialization
vImgNames = dir(fullfile(nameDir,'*.png'));

nImgs = length(vImgNames);
nFeaturePoints = fs.nPointsX * fs.nPointsY;
expectedFeatures = nImgs * nFeaturePoints;
w = fs.cellWidth;
h = fs.cellHeight;

vFeatures = zeros(expectedFeatures, fs.numBins*4*4); % 16 histograms containing 8 bins
vPatches = zeros(expectedFeatures, 4*w*4*h); % 16*16 image patches
vY = zeros(expectedFeatures, 1);

%% Extract features for all images
for i=1:nImgs
    disp(strcat('  Processing image ', num2str(i),'...'));
    
    % Load the image
    img = double(rgb2gray(imread(fullfile(nameDir,vImgNames(i).name))));
    imgG = double(imread(fullfile(nameGround,vImgNames(i).name)));
    
    % Collect local feature points for each image and compute a hog
    % descriptor and patch for each local feature point.
    vPoints = grid_points(img, fs.nPointsX, fs.nPointsY, fs.border);
    [descriptors, patches] = descriptors_hog(img, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    [y] = descriptors_hog_y(imgG, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    
    currentInterval = (i-1)*nFeaturePoints+1:(i)*nFeaturePoints;
    vFeatures(currentInterval, :) = descriptors;
    vPatches(currentInterval, :) = patches;
    vY(currentInterval, :) = y;
end

vFeatures = vFeatures(find(vY >= 0.5), :);
vPatches = vPatches(find(vY >= 0.5), :);

disp(strcat('    Number of extracted features:',num2str(size(vFeatures,1))));

% Cluster the features using K-Means
disp(strcat('  Clustering...'));
vCenters = kmeans(vFeatures, k, numiter);

% Visualize the code book
disp('Visualizing the codebook...');
visualize_codebook(vCenters, vFeatures, vPatches, fs.cellWidth, fs.cellHeight);

% disp('Press any key to continue...');
% pause;

end