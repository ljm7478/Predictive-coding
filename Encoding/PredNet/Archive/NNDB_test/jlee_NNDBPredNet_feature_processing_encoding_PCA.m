%% This code is for processing the CNN features extracted from videos
% 
% Data: The video-fMRI dataset are available online: 
% https://engineering.purdue.edu/libi/lab/Resource.html.

%% History
% v1.0 (original version) --2017/09/17


%% Step1: reframe the features to match with fMRI stimuli (136898 frames to 136750 frames)
% clc;clear
% 
% % dir
% dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
% saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/';
% 
% % calculate the temporal mean 
% secpath = [dataroot, '/', 'total', '/', '', '.h5'];
% lay_feat = h5read(secpath,'/predimg');
% dim = size(lay_feat);
% 
% % Specify the desired size
% newSize = 136750;
% 
% % Cut the last frames
% lay_feat = lay_feat(:, :, :,1:newSize);
% dim = size(lay_feat);
% 
% %save cut frames lay_feat
% h5create([dataroot,'/', 'total', '/', 'pre1_reframes.h5'],'/data',...
%     [size(lay_feat)],'Datatype','single');
% h5write([dataroot,'/', 'total', '/','pre1_reframes.h5'],'/data', lay_feat);

%% Step2: calculate the temporal mean and standard deviation of the feature time series of CNN units
clc;clear
% dir
dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/X';

%open featuremap
secpath = [dataroot, '/', 'total', '/', 'pre1_reframes', '.h5'];
lay_feat = h5read(secpath,'/data');
dim = size(lay_feat);

lay_feat_mean = sum(lay_feat, 4) / dim(4);

h5create([saveroot,'total_500SUM_PredNet_feature_maps_avg.h5'],'/data',...
    [size(lay_feat_mean)],'Datatype','single');
h5write([saveroot,'total_500SUM_PredNet_feature_maps_avg.h5'],'/data', lay_feat_mean);

% calculate the temporal standard deviation
secpath = [dataroot, '/', 'total', '/', 'pre1_reframes', '.h5'];
lay_feat = h5read(secpath,'/data');

lay_feat_mean = h5read([saveroot,'total_500SUM_PredNet_feature_maps_avg.h5'],'/data');

lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
lay_feat = lay_feat.^2;
dim = size(lay_feat);

lay_feat_std = zeros([dim(1:end-1),1]); 
lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));

lay_feat_std = sqrt(lay_feat_std/(dim(end)-1));
lay_feat_std(lay_feat_std==0) = 1;

h5create([saveroot,'total_500SUM_PredNet_feature_maps_std.h5'],'/data',...
    [size(lay_feat_mean)],'Datatype','single');
h5write([saveroot,'total_500SUM_PredNet_feature_maps_std.h5'],'/data',lay_feat_std);

%% Step3: PCA 
clc;clear

%dir
dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/X';
% % % % % % % % % % SVD-updating algorithm % % % % % % % % %
Niter = 2; % number of iteration to compute principle component

lay_feat_mean = h5read([saveroot, '/' , 'total_500SUM_PredNet_feature_maps_avg.h5'], '/data');
lay_feat_std = h5read([saveroot, '/' , 'total_500SUM_PredNet_feature_maps_std.h5'],'/data');

k0 = 0;
percp = 0.90; % explain 99% of the variance of every movie segments
disp(['percp is ', num2str(percp)]);

for iter =  1  : Niter
    
    %load lay_feat (pre1_reframes.h5)
    secpath = [dataroot, '/', 'total', '/', 'pre1_reframes', '.h5'];
    lay_feat = h5read(secpath,'/data');

    %normalization lay_feat
    lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
    lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
    lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
    
    %reshape normalized lay_feat to features-by-time
    dim = size(lay_feat);
    lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
    
    if iter == 1
        [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
    else
        [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
    end  

    disp('second iteration start!'); 
end
s = diag(S);

disp('start saving SVD_layer');
% save principal components
save([saveroot,'/', '500SUM_PredNet_feature_maps_svd_0.90.mat'], 'B', 's', '-v7.3');

%% Step4: Processed the hrf
clc; clear 

% dir
dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/X';

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 12.5; %original code is at TR=2, [if TR=1 && srate is 25] then make it into 12.5 

% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);
tic

disp('load lay_feat mean, std, svd')
% Dimension reduction for features
lay_feat_mean = h5read([saveroot, '/' , 'total_500SUM_PredNet_feature_maps_avg.h5'], '/data');
lay_feat_std = h5read([saveroot, '/' , 'total_500SUM_PredNet_feature_maps_std.h5'],'/data');
load([saveroot,'/', '500SUM_PredNet_feature_maps_svd_0.90.mat'], 'B');

% Dimension reduction for testing data
disp('load lay_feat')
secpath = [dataroot, '/', 'total', '/', 'pre1_reframes', '.h5'];
lay_feat = h5read(secpath,'/data');

disp('compute lay_feat regulaization')
lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

disp('reshape lay_feat')
dim = size(lay_feat);
lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

disp('compute X with svd applied')
X = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components

disp('compute ts')
ts = conv2(hrf,X); % convolude with hrf
ts = ts(4*srate+1:4*srate+size(X,1),:);
ts = ts(srate+1:2*srate:end,:); % downsampling to match fMRI

disp('start saving')
h5create([saveroot,'/','total_500SUM_PredNet_feature_SVD_0.90_hrf.h5'],'/data',...
    [size(ts)],'Datatype','single');
h5write([saveroot,'/','total_500SUM_PredNet_feature_SVD_0.90_hrf.h5'], '/data', ts);
toc   % (123.66 sec)

%% Step5: divided into seg 1-6 (approx. 15min each)
clc; clear
% dir
dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/X';

%open ts
secpath = [saveroot,'/','total_500SUM_PredNet_feature_SVD_0.90_hrf.h5'];
ts = h5read(secpath,'/data');

% Size of the ts array
ts_size = size(ts);

% Calculate the number of rows per segment
rowsPerSegment = ts_size(1) / 6;

% Generate indices to divide the rows into segments
rowIndices = round(linspace(1, ts_size(1) + 1, 7));

% Create a cell array to store the segments
segments = cell(1, 6);

% Split the array into segments
for i = 1:6
    segmentRows = rowIndices(i):rowIndices(i+1)-1;
    segments{i} = ts(segmentRows, :);
    name = strcat('segments',num2str(i));
    eval([name, '= segments{i};']);
end

% %check if array splited correct
% corrcoef(ts(912,:), segments{1,1}(912,:))
% corrcoef(ts(913,:), segments{1,2}(1,:))

% Save the segments in .h5 files
for i = 1:6
    segmentData = segments{i};
    filename = [saveroot, '/', 'total_segment', num2str(i),'_svd0.90', '.h5'];
    h5create(filename, '/data',  size(segmentData), 'Datatype', 'double');
    h5write(filename, '/data',segmentData);
end


% %open .h5
% secpath = [saveroot, '/', 'segment', num2str(i), '.h5'];
% seg1 = h5read(secpath, '/data');



%% Step6: Train Encoding with NNDB fMRI processing w/ fMRIPrep + 6mm smoothing (run1-3 concat then split into test)
clc; clear;
disp('Load fmri responses -NNDB')
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'))

% Load Y
num_subjects = 5;
num_runs = 3;
sub_cii = cell(num_subjects, 1);

for subject = 1:num_subjects
    subject_cii = [];
    
    for run = 1:num_runs
        folder_path = sprintf('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/02_ciftify/sub-%d/sub-%d/MNINonLinear/Results/task-500daysofsummer_run-%d', subject, subject, run);
        
        if run == 3
            if ~isfolder(folder_path)
                warning('Folder not found: %s', folder_path);
                break;  % Skip to the next subject if run-3 folder is missing
            end
            
            files = dir(fullfile(folder_path, '*s6.dtseries.nii'));
            
            if isempty(files)
                warning('No NIfTI files found in folder: %s', folder_path);
                break;  % Skip to the next subject if run-3 folder is empty
            end
            
            cii_run = cell(1, numel(files));
            
            for i = 1:numel(files)
                file_path = fullfile(folder_path, files(i).name);
                cii = ciftiopen(file_path, 'wb_command');
                cii_run{i} = cii.cdata;
            end
            
            subject_cii = cat(2, subject_cii, cat(2, cii_run{:}));
        else
            file_path = sprintf('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/02_ciftify/sub-%d/sub-%d/MNINonLinear/Results/task-500daysofsummer_run-%d/task-500daysofsummer_Atlas_s6.dtseries.nii', subject, subject, run);
            
            if isfile(file_path)
                cii = ciftiopen(file_path, 'wb_command');
                subject_cii = cat(2, subject_cii, cii.cdata);
            end
        end
    end
    
    sub_cii{subject} = subject_cii;
end

% Subcortex only
subcortex_no = 1:59412;
num_subjects = 5;
num_timepoints = 3647;
Y = cell(num_subjects, 1);

for subject = 1:num_subjects
    Y{subject} = sub_cii{subject}(subcortex_no, 1:num_timepoints);
end

% X model featuremaps 
disp('Load model featuremaps')

%dir
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding';

for i = 1:6
    secpath = [saveroot, '/', 'X', '/', 'total_segment', num2str(i),'_svd0.90', '.h5'];
    lay_feat = h5read(secpath,'/data');
    name = strcat('lay_feat',num2str(i));
    eval([name, '= lay_feat;']);
end

disp('Load X1_4')
X1_4 = cat(1, lay_feat1, lay_feat2, lay_feat3, lay_feat4);


%Voxelwise_encoding 
lambda = 0.5;
nfold = 9; %from paper 

for i = 1:5
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
    [W, Rmat, Lambda] = voxelwise_encoding(X1_4, Y{i}', lambda, nfold);

    % save W (feature x voxel)
    name = ['sub', num2str(i), '_W_svd0.90_ciftify_s6.mat'];
    save([saveroot,'/', 'W','/', name], 'W', '-v7.3');   
end

%% Step7: Load test Y and X
clc; clear;

% Load Y
% Load Y
num_subjects = 5;
num_runs = 3;
sub_cii = cell(num_subjects, 1);

for subject = 1:num_subjects
    subject_cii = [];
    
    for run = 1:num_runs
        folder_path = sprintf('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/02_ciftify/sub-%d/sub-%d/MNINonLinear/Results/task-500daysofsummer_run-%d', subject, subject, run);
        
        if run == 3
            if ~isfolder(folder_path)
                warning('Folder not found: %s', folder_path);
                break;  % Skip to the next subject if run-3 folder is missing
            end
            
            files = dir(fullfile(folder_path, '*s6.dtseries.nii'));
            
            if isempty(files)
                warning('No NIfTI files found in folder: %s', folder_path);
                break;  % Skip to the next subject if run-3 folder is empty
            end
            
            cii_run = cell(1, numel(files));
            
            for i = 1:numel(files)
                file_path = fullfile(folder_path, files(i).name);
                cii = ciftiopen(file_path, 'wb_command');
                cii_run{i} = cii.cdata;
            end
            
            subject_cii = cat(2, subject_cii, cat(2, cii_run{:}));
        else
            file_path = sprintf('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/02_ciftify/sub-%d/sub-%d/MNINonLinear/Results/task-500daysofsummer_run-%d/task-500daysofsummer_Atlas_s6.dtseries.nii', subject, subject, run);
            
            if isfile(file_path)
                cii = ciftiopen(file_path, 'wb_command');
                subject_cii = cat(2, subject_cii, cii.cdata);
            end
        end
    end
    
    sub_cii{subject} = subject_cii;
end


% Subcortex only
subcortex_no = 1:59412;
num_subjects = 5;
num_timepoints = 3648;
Y = cell(num_subjects, 1);

for subject = 1:num_subjects
    Y{subject} = sub_cii{subject}(subcortex_no, num_timepoints:5470);
end

%dir
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding';

% get test X (feature)
disp('Load model featuremaps')

for i = 1:6
    secpath = [saveroot, '/', 'X', '/',  'total_segment', num2str(i),'_svd0.99', '.h5'];
    lay_feat = h5read(secpath,'/data');
    name = strcat('lay_feat',num2str(i));
    eval([name, '= lay_feat;']);
end

% get test X for seg5 and 6 (feature)
X_56 = cat(1, lay_feat5, lay_feat6);

%Y_hat build

disp('make Y_hat')
for i = 1:5
    W = load([saveroot,'/', 'W','/', 'sub', num2str(i), '_W_svd0.99_ciftify_s6.mat']);
    name = ['Y_pred56_sub',num2str(i)]; 
    eval([name, '=X_56 * W.W;']);
    
    %save X_hat
    Yhat = eval(name);
    name2 = strcat('sub', num2str(i), '_pred_svd0.99_s6', '.mat');
    save([saveroot,'/', 'Y_hat','/', name2], 'Yhat', '-v7.3');
end

% correaltion between Y_predict and actual Y for test
disp('correaltion and p_values compute start')

for i = 1:5
    disp(['correalting sub', num2str(i)])
    
    Ypred_name = ['Y_pred56_sub',num2str(i)]; 
    Y_pred = eval(Ypred_name);
    
    [correlations, p_values] = auto_corr(Y{i}',Y_pred);
    
    correlations = correlations';
    
    % save correlation (feature x voxel)
    disp('save correlations')

    name = strcat('sub', num2str(i), '_corr_svd0.90_s6', '.mat');
    save([saveroot,'/', 'correlations','/', name], 'correlations', '-v7.3');
    
    % Find p_values < 0.05
    p_values=p_values';
    significant_p = p_values < 0.05;
    
    % Replace numbers above 0.05 with 0
    p_values(~significant_p) = 0;
    
    % Find the count of numbers in p_values
    count = nnz(p_values);
    
    % save significant_p (feature x voxel)
    disp('save p_values')

    name = strcat('sub', num2str(i), '_svd0.90_p_values_0.05_s6', '.mat');
    save([saveroot,'/', 'p_values','/', name], 'p_values', '-v7.3');
end
% [correlations, p_values] = auto_corr(X1_56',X_pred56_sub1);
% correlations = correlations';
% p_values = p_values';

% % Find p_values < 0.05
% significant_p = p_values < 0.05;
% 
% % Replace numbers above 0.05 with 0
% p_values(~significant_p) = 0;
% 
% % Find the count of numbers in p_values
% count = nnz(p_values);
%% Step8: make dtseries 
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'))
fmriroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/dt_result/Cifti_s6/';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding';

%load cifti -sub1
load([saveroot,'sub1_corr_svd0.99_s6', '.mat']); 

sub1_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-1/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub1_cii = rmfield(sub1_cii, 'cdata');
sub1_cii.cdata = corr_sub1';
sub1_cii.diminfo{1,1}.length=size(one,1);
sub1_cii.diminfo{1,2}.length=size(one,2);
sub1_cii.diminfo{1,1}.models(3:end)=[]; %%subcortex removes

%save as dtseries
save_file_name='sub1_500SUM_PredNet_test56_0.99_s6.dtseries.nii';
ciftisavereset(sub1_cii, [fmriroot, save_file_name], 'wb_command');

%load cifti -sub2
sub2_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-2/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub2_cii = rmfield(sub2_cii, 'cdata');

sub2_cii.cdata = corr_sub2';
sub2_cii.diminfo{1,1}.length=size(corr_sub2',1);
sub2_cii.diminfo{1,2}.length=size(corr_sub2',2);
sub2_cii.diminfo{1,1}.models(3:end)=[]; %%subcortex removes

%save as dtseries
save_file_name='sub2_500SUM_PredNet_test56_0.99_s6.dtseries.nii';
ciftisavereset(sub2_cii, [fmriroot, save_file_name], 'wb_command');


%load cifti -sub3
sub3_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-3/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub3_cii = rmfield(sub3_cii, 'cdata');

sub3_cii.cdata = corr_sub3';
sub3_cii.diminfo{1,1}.length=size(corr_sub3',1);
sub3_cii.diminfo{1,2}.length=size(corr_sub3',2);
sub3_cii.diminfo{1,1}.models(3:end)=[]; %%subcortex removes

%save as dtseries
save_file_name='sub3_500SUM_PredNet_test56_0.99_s6.dtseries.nii';
ciftisavereset(sub3_cii, [fmriroot, save_file_name], 'wb_command')


%load cifti -sub4
sub4_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-4/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub4_cii = rmfield(sub4_cii, 'cdata');

sub4_cii.cdata = corr_sub4';
sub4_cii.diminfo{1,1}.length=size(corr_sub4',1);
sub4_cii.diminfo{1,2}.length=size(corr_sub4',2);
sub4_cii.diminfo{1,1}.models(3:end)=[]; %%subcortex removes

%save as dtseries
save_file_name='sub4_500SUM_PredNet_test56_0.99_s6.dtseries.nii';
ciftisavereset(sub4_cii, [fmriroot, save_file_name], 'wb_command')


%load cifti -sub5
sub5_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-5/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub5_cii = rmfield(sub5_cii, 'cdata');

sub5_cii.cdata = corr_sub5';
sub5_cii.diminfo{1,1}.length=size(corr_sub5',1);
sub5_cii.diminfo{1,2}.length=size(corr_sub5',2);
sub5_cii.diminfo{1,1}.models(3:end)=[]; %%subcortex removes

%save as dtseries
save_file_name='sub5_500SUM_PredNet_test56_0.99_s6.dtseries.nii';
ciftisavereset(sub5_cii, [fmriroot, save_file_name], 'wb_command')
