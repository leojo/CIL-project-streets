%
% BAG OF WORDS RECOGNITION EXERCISE
% Alex Mansfield and Bogdan Alexe, HS 2011
%

%% Settings
cellSize = 4;
fs.cellWidth = cellSize;
fs.cellHeight = cellSize;
fs.nPointsX = 10;
fs.nPointsY = 10;
fs.border = 2*cellSize;
fs.numBins = 8;

sizeCodebook = 200;
numIterations = 10;

%% Training
disp('creating codebook');
vCenters = create_codebook('../data/airshow-training-pos', sizeCodebook, numIterations, fs);
%keyboard;
disp('processing positve training images');
vBoWPos = create_bow_histograms('../data/airshow-training-pos', vCenters, fs);
disp('processing negative training images');
vBoWNeg = create_bow_histograms('../data/airshow-training-neg', vCenters, fs);
%vBoWPos_test = vBoWPos;
%vBoWNeg_test = vBoWNeg;

%% Testing
% bow_test('../data/airshow-testing-pos', vCenters, fs, vBoWPos, vBoWNeg, 'Land', 'Sky');

disp('processing positve testing images');
vBoWPos_test = create_bow_histograms('../data/airshow-testing-pos', vCenters, fs);
disp('processing negative testing images');
vBoWNeg_test = create_bow_histograms('../data/airshow-testing-neg', vCenters, fs);

nrPos = size(vBoWPos_test, 1);
nrNeg = size(vBoWNeg_test, 1);

test_histograms = [vBoWPos_test;vBoWNeg_test];
labels = [
    ones(nrPos,1);
    zeros(nrNeg, 1)
    ];

disp('______________________________________')
disp('Nearest Neighbor classifier')
bow_recognition_multi(test_histograms, labels, vBoWPos, vBoWNeg, @bow_recognition_nearest);
disp('______________________________________')
disp('Bayesian classifier')
bow_recognition_multi(test_histograms, labels, vBoWPos, vBoWNeg, @bow_recognition_bayes);
disp('______________________________________')
