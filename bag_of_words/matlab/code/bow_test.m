function bow_test(nameDir, vCenters, fs, vBoWPos, vBoWNeg, label0, label1)

vImgNames = dir(fullfile(nameDir,'*.png'));
nImgs = length(vImgNames);
vImgNames = vImgNames(randperm(nImgs));
vBoW  = zeros(nImgs, size(vCenters,1));

% Extract features for all images in the given directory
for i = 1:nImgs
    disp(strcat('  Processing image ', num2str(i),'...'));
    
    % load the image
    imgRead = imread(fullfile(nameDir,vImgNames(i).name));
    imgScaled = imresize(imgRead, [300, 400]);
    img = double(rgb2gray(imgRead));
    
    % Collect local feature points for each image
    % and compute a descriptor for each local feature point
    vPoints = grid_points(img, fs.nPointsX, fs.nPointsY, fs.border);
    % create hog descriptors and patches
    [descriptors, ~] = descriptors_hog(img, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);
    
    % Create a BoW activation histogram for this image
    histogram = bow_histogram(descriptors, vCenters)';
    vBoW(i, :) = histogram;
    
    ln = bow_recognition_nearest(histogram, vBoWPos, vBoWNeg);
    lb = bow_recognition_bayes(histogram, vBoWPos, vBoWNeg);
    lnT = label0;
    if ln == 1
        lnT = label1;
    end
    
    lbT = label0;
    if lb == 1
        lbT = label1;
    end
        
    figure(2), imshow(imgScaled);
    hold on;
    text(20, 10, lnT, 'FontSize', 20, 'Color', [1 0 0]);
    text(20, 30, lbT, 'FontSize', 20, 'Color', [0 1 0]);
    
    pause(3.0)
end

end