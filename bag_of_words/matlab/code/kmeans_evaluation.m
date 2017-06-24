close all;

cellSize = 4;
fs.cellWidth = cellSize;
fs.cellHeight = cellSize;
fs.nPointsX = 10;
fs.nPointsY = 10;
fs.border = 2*cellSize;
fs.numBins = 8;

fs.maxTraining = 10;
fs.maxTesting = 1;
fs.stride = 8;

sizeCodebook = 300;
numIterations = 10;

[vCentersP, vCentersN] = create_codebook('../data/training/images', '../data/training/groundtruth', sizeCodebook, numIterations, fs);
bow_recognition_multi('../data/test_set_images', fs, vCentersP, vCentersN, @bow_recognition_nearest);

fprintf('\n');

function [vCentersP, vCentersN] = create_codebook(nameDir, nameGround, k, numiter, fs)
vImgNames = dir(fullfile(nameDir,'*.png'));

nImgs = min(length(vImgNames), fs.maxTraining);
w = fs.cellWidth;
h = fs.cellHeight;

vFeaturesP = zeros(0, fs.numBins*4*4); % 16 histograms containing 8 bins
vPatchesP = zeros(0, 4*w*4*h); % 16*16 image patches
vFeaturesN = zeros(0, fs.numBins*4*4); % 16 histograms containing 8 bins
vPatchesN = zeros(0, 4*w*4*h); % 16*16 image patches


%% Extract features for all images
for i=1:nImgs
    disp(strcat('  Processing image ', num2str(i),'...'));
    
    % Load the image
    img = double(rgb2gray(imread(fullfile(nameDir,vImgNames(i).name))));
    imgG = double(imread(fullfile(nameGround,vImgNames(i).name)));
    
    % Collect local feature points for each image and compute a hog
    % descriptor and patch for each local feature point.
    imgH = size(img,1)-(2 * fs.cellWidth * fs.cellWidth);
    imgW = size(img,2)-(2 * fs.cellWidth * fs.cellWidth);
    vPoints = generateCheckerboardPoints([imgH,imgW]./fs.stride, fs.stride) + (fs.cellWidth * fs.cellWidth);
%     vPoints = grid_points(img, fs.nPointsX, fs.nPointsY, fs.border);
    [descriptors, patches] = descriptors_hog(img, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    [y] = descriptors_hog_y(imgG, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    
    descriptorsP = descriptors(find(y >= 0.5), :);
    descriptorsN = descriptors(find(y <  0.5), :);
    patchesP = patches(find(y >= 0.5), :);
    patchesN = patches(find(y <  0.5), :);
    
    % Store positive features => street features
    vFeaturesP = [vFeaturesP; descriptorsP];
    vPatchesP = [vPatchesP; patchesP];
    
    % Store negative features => other features
    vFeaturesN = [vFeaturesN; descriptorsN];
    vPatchesN = [vPatchesN; patchesN];
end

% vY = [ones(size(vFeaturesP,1),1); zeros(size(vFeaturesN,1),1)];
vFeatures = [vFeaturesP;vFeaturesN];
% vPatches = [vPatchesP;vPatchesN];

disp(strcat('    Number of extracted features:', num2str(size(vFeatures,1))));

% Cluster the features using K-Means
disp(strcat('  Clustering...'));
vCentersP = kmeans(vFeaturesP, k, numiter);
vCentersN = kmeans(vFeaturesN, k, numiter);

% vDataPoints = kmeans(vFeatures, k, numiter);
% vFeatures = vFeatures(:, 1:end-1);
% vCenters = vDataPoints(:, 1:end-1);
% vClasses = vDataPoints(:, end);

% Visualize the code book
disp('Visualizing the codebook...');
figure;
visualize_codebook(vCentersP, vFeaturesP, vPatchesP, fs.cellWidth, fs.cellHeight);
figure;
visualize_codebook(vCentersN, vFeaturesN, vPatchesN, fs.cellWidth, fs.cellHeight);

% disp('Press any key to continue...');
% pause;

end


function bow_recognition_multi(nameDir, fs, vBoWPos, vBoWNeg, classifierFunction)
vImgNames = dir(fullfile(nameDir,'test*'));

image_count = min(size(vImgNames,1), fs.maxTesting);
w = fs.cellWidth;
h = fs.cellHeight;

for i = 1:image_count
    % classify each histogram
    img = double(rgb2gray(imread(fullfile(strcat(nameDir, '/', vImgNames(i).name),strcat(vImgNames(i).name, '.png')))));
    imgPred = zeros(size(img));
    
    imgH = size(img,1)-(4 * w);
    imgW = size(img,2)-(4 * h);
    
    % take every x-th pixel
    vPoints = generateCheckerboardPoints([imgH,imgW]./fs.stride, fs.stride) + (fs.cellWidth * fs.cellWidth);
    size(vPoints)
    
    [descriptors, ~] = descriptors_hog(img, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    size(descriptors)
    
    for d = 1:size(descriptors,1)
        l = classifierFunction(descriptors(d,:), vBoWPos, vBoWNeg);
        imgPred(vPoints(d, 2), vPoints(d,1)) = l;
        
        fprintf('.');
        if mod(d, 100)==0
            fprintf('\n')
        end
    end
    
    figure;
    imshow(imgPred);
    
end

end

function sLabel = bow_recognition_nearest(histogram, vBoWPos, vBoWNeg)
% Label the histogram by +1 or 0, based on the closest found histogram in
% the positive sampled or negative sampled respectively.
%
% Input
%   histogram       Histogram to label.
%   vBoWPos         All histograms from the positive samples.
%   vBoWNeg         All histograms from the negative samples.
%
% Output:
%   sLabel          1 if the closest neighbour is positive => contains car
%                   0 if the closest neighbour is negative => no car

%% Find the nearest neighbor in the positive and negative sets
[~, distPos] = findnn(histogram, vBoWPos);
[~, distNeg] = findnn(histogram, vBoWNeg);

%% Label the histogram based on the closer neighbor
if distPos < distNeg
    sLabel = 1;
else
    sLabel = 0;
end

end
