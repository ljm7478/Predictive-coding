%% This code is for evaluating the reproducibility of the response to the same movie stimuli
%
% Reference: 
% Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
% and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
% Cortex, In press.

%% history
% v1.0 (original version) --2017/09/13

%% Reproducibility analysis
% Load fMRI responses during watching the movie for the first and second time
% Nv: number of voxels, Nv = 59412.
% Nt: number of volumes for one movie segment, Nt = 240.
% Ns: number of movie segments, Ns = 18.
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/subject1_fmri/';
load([dataroot,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
 
% Calculate the voxelwise correlation between the responses to the same movie for the
% first time and the second time.
Nv=59412
Ns=18
Rmat = zeros(Nv, Ns); 
for seg = 1 : Ns
   dt1 = fmri.data1(:,:,seg);
   dt2 = fmri.data2(:,:,seg);
   Rmat(:,seg) = amri_sig_corr(dt1',dt2','mode','auto'); 
end

% Fisher's r-to-z-transformation
Zmat = amri_sig_r2z(Rmat);

%%
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/hiearchical_mapping/';


%load cii structure 
subject_id='subject1';
fmripath = '/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/subject1/video_fmri_dataset/subject1/fmri/'; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');