% This code aims to compare the performance of PPDG, ADMM, PDHG 
% for image denoising. 
%
%
%
%  version:  2024/1/26


clear;
clc;

%input original image
ori = imread (fullfile('image', 'shiba_inu_74.jpg'));  
ori   = rgb2gray(ori);
figure (1);
imshow(ori );

%poisson gaussian
im=  imnoise(ori ,'gaussian',0, 0.005); 
figure (2);
imshow(im);
ori = double(ori)/255;

% denoised image by PPDG
outim=TVL0denoisePPDG_box(ori, im, 0.1,100);
figure (3);
imshow(outim, []);

% denoised image by ADMM
outim=TVL0denoiseADMM_box(ori, im, 0.1, 100);
figure (4); 
imshow(outim, []);

% denoised image by ADMM
 outim=TVL0denoisePDHG_box(ori, im, 0.1, 100);
 figure (5); imshow(outim, []);
