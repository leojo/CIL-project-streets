
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



%%
function [F]=llf_andy(RGB, I,sigma,fact,N)

    [height width]=size(I);
    n_levels=ceil(log(min(height,width))-log(2))+2;
    discretisation=linspace(0,1,N);
    discretisation_step=discretisation(2);
    
    input_gaussian_pyr=gaussian_pyramid(I,n_levels);
    output_laplace_pyr=laplacian_pyramid(I,n_levels);
    output_laplace_pyr{n_levels}=input_gaussian_pyr{n_levels};
    
    for ref=discretisation
        I_remap=fact*(I-ref).*exp(-(I-ref).*(I-ref)./(2*sigma*sigma));
        
%         figure;
%         subplot(1,2,1); imshow(I.*RGB); title('Input');
%         subplot(1,2,2); imshow((I_remap+I).*RGB); title(strcat('G: ', num2str(ref)));
        
        temp_laplace_pyr=laplacian_pyramid(I_remap,n_levels);
        
%         for level=1:3
%             figure; imshow(1 - temp_laplace_pyr{level} * 10); title(strcat('G: ', num2str(ref), ' Level: ', num2str(level)));
%         end
        
        for level=1:n_levels-1
            output_laplace_pyr{level}=output_laplace_pyr{level}+...
                (abs(input_gaussian_pyr{level}-ref)<discretisation_step).*...
                temp_laplace_pyr{level}.*...
                (1-abs(input_gaussian_pyr{level}-ref)/discretisation_step);            
        end
    end
    
%     for level=1:n_levels-1
%         figure;
%         imshow(1 - (output_laplace_pyr{level}) * 10);
%         title(strcat('level: ', num2str(level-1)));
%     end
    
    F=reconstruct_laplacian_pyramid(output_laplace_pyr);
end