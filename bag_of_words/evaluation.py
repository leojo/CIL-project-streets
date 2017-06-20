import numpy as np
import sys

cellSize = 4;
cellWidth = cellSize;
cellHeight = cellSize;
nPointsX = 10;
nPointsY = 10;
border = 2*cellSize;
numBins = 8;

def findnn(descriptors1, descriptors2):
    # Find for each descriptor in descriptors1 the closest descriptor in
    # descriptors2.
    #
    # Input
    #   descriptors1    NxD matrix containing N feature vectors of dim. D
    #   descriptors2    MxD matrix containing M feature vectors of dim. D
    #
    # Output:
    #   index           N-dim. vector containing for each feature vector in D1
    #                   the index of the closest feature vector in D2.
    #   dist            N-dim. vector containing for each feature vector in D1
    #                   the distance to the closest feature vector in D2.

    # Initialization
    n = len(descriptors1, 1)
    m = len(descriptors2, 1)
    index = np.zeros(n, 1)
    dist = np.zeros(n, 1)

    # Find for each feature vector in D1 the nearest neighbor in D2

    # Calculate the distances for each descriptor in D1 to all descriptors in
    # D2. Get then the index and the distance of the closest.
    for i in range(1, n):
        # Get next descriptor in image one
        current = descriptors1[i]

        dist[i] = sys.maxsize

        # Calculate the difference between the current descriptor and all
        # descriptors of set 2. Then square it and sum it correctly.
        # After these steps the SSD between the descriptor of set 1 and
        # all of set 2 is generated.
        for v in range(1, m):
            ssd = (descriptors2[v] - current)**2
            ssd = np.sum(ssd, 2)

            if ssd < dist[i]:
                dist[i] = ssd
                index[i] = v


def create_codebook(nameDir, k, numiter, fs):
    # Create a codebook from the images which are located in the specified
    # directory. K and numiter are used to find k clusters in
    # numiter-iterations with the k-means. Fs contains the settings to extract
    # the feature-points and its descriptors.
    #
    # Input
    #   nameDir         Relative path to the directory which contains the
    #                   positive training pictures.
    #   k               Desired amount of cluster-centers.
    #   numiter         Amount of iterations to update the cluster-centers.
    #   fs              Contains all the settings (as cellWidth, ...) which are
    #                   used for the feature extraction steps.
    #
    # Output:
    #   vCenters        kxD matrix containing the cluter-centers of the
    #                   codebook.

    # Initialization
    nImgs = len(vImgNames)
    nFeaturePoints = nPointsX * nPointsY
    expectedFeatures = nImgs * nFeaturePoints
    w = fs.cellWidth
    h = fs.cellHeight

    vFeatures = np.zeros(expectedFeatures, numBins * 4 * 4); # 16 histograms containing 8 bins
    vPatches = np.zeros(expectedFeatures, 4 * w * 4 * h); # 16 * 16 image patches

    # Extract features for all images
    for i in range(1, nImgs):
        print('  Processing image ' + str(i) + '...')

        # Load the image
        img = double(rgb2gray(imread(fullfile(nameDir, vImgNames(i).name))));

        # Collect local feature points for each image and compute a hog
        # descriptor and patch for each local feature point.
        vPoints = grid_points(img, fs.nPointsX, fs.nPointsY, fs.border);
        [descriptors, patches] = descriptors_hog(img, vPoints, fs.numBins, fs.cellWidth, fs.cellHeight);

        currentInterval = (i - 1) * nFeaturePoints + 1:(i) * nFeaturePoints;
        vFeatures(currentInterval,:) = descriptors;
        vPatches(currentInterval,:) = patches;

    disp(strcat('    Number of extracted features:', str(len(vFeatures, 1))));

    # Cluster the features using K-Means
    disp(strcat('  Clustering...'));
    vCenters = kmeans(vFeatures, k, numiter);

    # Visualize the code book
    disp('Visualizing the codebook...');
    visualize_codebook(vCenters, vFeatures, vPatches, fs.cellWidth, fs.cellHeight);

    # disp('Press any key to continue...');
    # pause;