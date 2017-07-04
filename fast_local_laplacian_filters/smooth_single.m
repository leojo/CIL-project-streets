
   close all;

%% import image
figure(1);

name='mask_1.png';
I_rgb = imread(sprintf('../data/results/%s', name));


I = (im2double(I_rgb));
I_ratio=double(I_rgb)./repmat(I,[1 1 3])./255;


%% image smoothing
sigma=0.1;
N=100;
fact=-1;
tic
I_smoothed=llf(I,sigma,fact,N);
% I_smoothed=llf_andy(I_ratio,I,sigma,fact,N);
toc
I_smoothed=repmat(I_smoothed,[1 1 3]).*I_ratio;

imshow(I_smoothed);
title('Image Smoothed');

file_split = strsplit(name, '.');
smooth_name = file_split{1};
imwrite(I_smoothed, sprintf('../data/results_smooth/%s.jpg', smooth_name), 'Quality', 100)
