
   close all;

%% import image
path_orig = '../data/training_smooth/images/';
path_new = '../data/training_smooth2/images/';
path_test_orig = '../data/test_set_smooth/';
path_test_new = '../data/test_set_smooth2/';

img_list = dir(sprintf('%s%s', path_orig, '/sat*.jpg'));
figure(1);

for i = 1:size(img_list)
    file=img_list(i);
    I_rgb = imread(sprintf('%s%s', path_orig, file.name));

    I = rgb2gray(im2double(I_rgb));
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
    
    file_split = strsplit(file.name, '.');
    smooth_name = file_split{1};
    imwrite(I_smoothed, sprintf('%s%s.jpg', path_new, smooth_name), 'Quality', 100)
end


%% import image
img_list = dir(sprintf('%s%s', path_test_orig, 'test_*'));

for i = 1:size(img_list)
    file=img_list(i);
    I_rgb = imread(sprintf('%s%s/%s.jpg', path_test_orig, file.name, file.name));

    % name='andy_bernerTriathlon2';
    % I_rgb = imread(sprintf('images/%s.jpeg',name));

    I = rgb2gray(im2double(I_rgb));
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
    
    file_split = strsplit(file.name, '.');
    smooth_name = file_split{1};
    imwrite(I_smoothed, sprintf('%s%s/%s.jpg', path_test_new, file.name, file.name), 'Quality', 100)
end