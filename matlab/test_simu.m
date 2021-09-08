%%% This Script tests the generation of the events
clc; clear all;clear mex;clear classes;
mex -setup cpp
%% Compile the library. NB: the define files are already provided, this part 
%is useless unless you want to add new features. 
clibgen.generateLibraryDefinition('simu_matlab.hpp')
%% Init the simulator
definesimu_matlab
%% Build the simulator
build(definesimu_matlab)
addpath('simu_matlab')
%% Init simu
s = clib.simu_matlab.SimuICNSMatlab();
s.initSimu(346, 260)
s.initContrast(0.15, 0.15, 0.05)
s.initLat(100, 10, 50, 300)
%% Load Noise
load('../data/noise_pos_3klux.mat')
load('../data/noise_neg_3klux.mat')
s.initNoise(noise_neg, noise_pos)
%% Init Image
path_img = '/home/joubertd/Documents/Data/Simulations/Comparison_ESIM/frames_book/';
img_name = [path_img sprintf('frame_%d.png',1606209988861033)];
I = cast(imread(img_name), 'double');
I = I(:);
s.initImg(I)
s.setDebug()
%% Update image
img_name = [path_img sprintf('frame_%d.png',1606209988909912)];
I = cast(imread(img_name), 'double'); 
s.updateImg(I(:), 48879)
%% Get events
size = s.getBufSize()
ts = zeros([size 1], 'uint64');
x = zeros([size 1], 'uint16');
y = zeros([size 1], 'uint16');
p = zeros([size 1], 'uint8');
[ts x y p] = s.getBuffer(ts, x, y, p);
%% Display events
im = ones([260, 346]) * 125;
idx = (x+1) * 260 + y + 1;
im(idx)= p * 255;
imshow(cast(im, 'uint8'));