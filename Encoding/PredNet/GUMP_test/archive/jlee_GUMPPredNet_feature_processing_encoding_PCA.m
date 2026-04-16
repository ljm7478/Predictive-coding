
%% Step1: calculate the temporal mean and standard deviation of the feature time series of CNN units
clc;clear
% dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/01_gump/result/h5';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500UMCIT_trained_GUMP_test/encoding';

% calculate the temporal mean 
for i = 0:3
    disp(['sub', num2str(i)])
    secpath = [dataroot, '/', sprintf('E%d.h5', i)];
    lay_feat = h5read(secpath,'/predimg');
    dim = size(lay_feat);

    lay_feat_mean = sum(lay_feat, 4) / dim(4);
    
    disp(['save average for sub', num2str(i)])
    
    h5create([saveroot, '/', sprintf('E%d_feature_maps_avg.h5', i)],'/data',...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([saveroot, '/', sprintf('E%d_feature_maps_avg.h5', i)],'/data', lay_feat_mean);

    % calculate the temporal standard deviation
    disp(['start std for sub', num2str(i)])

    secpath = [dataroot, '/', sprintf('E%d.h5', i)];
    lay_feat = h5read(secpath,'/predimg');

    lay_feat_mean = h5read([saveroot, '/', sprintf('E%d_feature_maps_avg.h5', i)],'/data');

    lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
    lay_feat = lay_feat.^2;
    dim = size(lay_feat);

    lay_feat_std = zeros([dim(1:end-1),1]); 
    lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));

    lay_feat_std = sqrt(lay_feat_std/(dim(end)-1));
    lay_feat_std(lay_feat_std==0) = 1;
    
    disp(['save std for sub', num2str(i)])
    
    h5create([saveroot, '/', sprintf('E%d_feature_maps_std.h5', i)],'/data',...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([saveroot,'/', sprintf('E%d_feature_maps_std.h5', i)],'/data',lay_feat_std);
    
    disp('start next sub!')
end

disp('fin!')

%% Step2: PCA 
clc;clear

%dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/01_gump/result/h5';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500UMCIT_trained_GUMP_test/encoding';
% % % % % % % % % % SVD-updating algorithm % % % % % % % % %
Niter = 2; % number of iteration to compute principle component

lay_feat_mean = h5read([saveroot, '/' , 'Ahat0_feature_maps_avg.h5'], '/data');
lay_feat_std = h5read([saveroot, '/' , 'Ahat0_feature_maps_std.h5'],'/data');

k0 = 0;
percp = 0.90; % explain 99% of the variance of every movie segments
disp(['percp is ', num2str(percp)]);

for iter =  1  : Niter
    
    %load lay_feat (pre1_reframes.h5)
    secpath = [dataroot, '/', 'Ahat0.h5'];
    lay_feat = h5read(secpath,'/predimg');

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
save([saveroot,'/', 'X', '/','Ahat0_feature_maps_svd_0.90.mat'], 'B', 's', '-v7.3');

%% Step3: Processed the hrf
clc; clear 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/01_gump/result/h5'; 
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500UMCIT_trained_GUMP_test/encoding';

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 25; %original code is at TR=2, [if TR=1 && srate is 25] then make it into 12.5 

% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);
tic

disp('load lay_feat mean, std, svd')
% Dimension reduction for features
lay_feat_mean = h5read([saveroot, '/' , 'Ahat0_feature_maps_avg.h5'], '/data');
lay_feat_std = h5read([saveroot, '/' , 'Ahat0_feature_maps_std.h5'],'/data');
load([saveroot,'/', 'X', '/','Ahat0_feature_maps_svd_0.90.mat'], 'B');

% Dimension reduction for testing data
disp('load lay_feat')
secpath = [dataroot, '/', 'Ahat0.h5'];
lay_feat = h5read(secpath,'/predimg');

disp('compute lay_feat regulaization')
lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

disp('reshape lay_feat')
dim = size(lay_feat);
lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

disp('compute X with svd applied')
X = lay_feat'*B/sqrt(size(B,1)); % X: #time-by-#components

disp('compute ts')
ts = conv2(hrf,X); % convolude with hrf
ts = ts(4*srate+1:4*srate+size(X,1),:);
ts = ts(srate+1:2*srate:end,:); % downsampling to match fMRI

disp('start saving')
h5create([saveroot,'/','X', '/', 'Ahat0_feature_SVD_0.90_hrf.h5'],'/data',...
    [size(ts)],'Datatype','single');
h5write([saveroot,'/','X', '/', 'Ahat0_feature_SVD_0.90_hrf.h5'], '/data', ts);
toc   % (123.66 sec)

%% Step4: divided into seg 1-8 (approx. 15min each)
clc; clear
% dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/01_gump/result/h5'; 
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500UMCIT_trained_GUMP_test/encoding';

%open ts
secpath = [saveroot,'/','X', '/', 'Ahat0_feature_SVD_0.90_hrf.h5'];
ts = h5read(secpath,'/data');

% Size of the ts array
ts_size = size(ts);

Nv = [451, 441, 438, 488, 462, 439, 542, 338];

% Create a cell array to store the segments
segments = cell(1, 8);

% Calculate the start and end indices for each segment
start_indices = [1, cumsum(Nv(1:end-1)) + 1];
end_indices = cumsum(Nv);

% Extract each segment from ts and store it in the cell array
for i = 1:8
    segments{i} = ts(start_indices(i):end_indices(i),:);
    
    % Save the segment into a separate file
    segment_filename = ['Ahat_SVD0.90_seg', num2str(i), '.mat'];
    save([saveroot,'/', 'X', '/', segment_filename], 'segments', '-v7.3');
end


%% Step5.1: Train Encoding with GUMP fMRI processing w/ fMRIPrep + 3mm smoothing 
% Load Y
clc;clear 
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
num_runs = 8;
sub_cii = cell(numel(num_subjects), 1);
subcortex_no = 1:59412;
existing_subjects = {};  % To store the subjects that are found

for subject_idx = 1:numel(num_subjects)
    subject = num_subjects{subject_idx};
    subject_cii = cell(1, num_runs);
    subject_exists = true;  % Flag to indicate if subject exists
    
    for run = 1:num_runs
        folder_path = sprintf('/combinelab/03_user/jungmin/01_data/03_Gump/02_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-%d', subject, subject, subject, run);
        
        files = dir(fullfile(folder_path, '*s3.dtseries.nii'));
        cii_run = cell(1, numel(files));

        for i = 1:numel(files)
            file_path = fullfile(folder_path, files(i).name);
            cii = ciftiopen(file_path, 'wb_command');
            cii_run{i} = cii.cdata(subcortex_no, :);
        end

        subject_cii{run} = cat(2, cii_run{:});
        
        % Check if subject data is empty
        if isempty(subject_cii{run})
            subject_exists = false;
            break;  % Exit the loop if subject data is empty for any run
        end
    end
    
    % Store subject data if it exists for all runs
    if subject_exists
        existing_subjects = [existing_subjects, subject];  % Append to existing_subjects
        sub_cii{subject_idx} = subject_cii;
    end
end

% Save existing subjects
save_file = [saveroot, '/', 'Y', '/', 'fMRI_subcortex_existing'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');

%% Step5.2:X model featuremaps 
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';


for i = 0:3
    X = cell(1, 7);
    for seg = 1:7
        filename = sprintf('%s/X/Ahat/Ahat%d_svd0.9hrf_seg%d.mat', saveroot, i, seg);
        loaded_data = load(filename);
        X{seg} = loaded_data.segments;  % Replace 'semgnet' with the desired field name
    end
    name = sprintf('Ahat%d_1_7', i);
    Ahat_concatenated.(name)= cat(1, X{:});
    eval([name, ' = Ahat_concatenated.(name);']);
end


%Voxelwise_encoding 
lambda = [0.1:0.2:0.9];
nfold = 9; %from paper 

%load Y
load([saveroot, '/' , 'Y', '/', 'fMRI_subcortex_existing.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 7:8
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
    sub_cii1_7 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7});
    
    for feature=0:3
        Ahat_field_name = sprintf('Ahat%d_1_7', feature);
        Ahat_argument = Ahat_concatenated.(Ahat_field_name);  % Assuming Ahat_concatenated is the struct from previous code

        [W, Rmat, Lambda] = voxelwise_encoding(Ahat_argument, sub_cii1_7', lambda, nfold);

        % save W (feature x voxel)
        name = [subject, '_', 'Ahat', num2str(feature), '_W_svd0.9_s3.mat'];
        save([saveroot,'/', 'W','/','PCA','/', name], 'W', '-v7.3');  
    end
end

%% Step6: Load test Y and X
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
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500SUMCIT_trained_GUMP_test/feature_processed_encoding';

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
%% Step7: make dtseries 
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'))
fmriroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500SUMCIT_trained_GUMP_test/encoding/dt_result/Cifti_s6/';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/stimuli/PredNet/03_500SUMCIT_trained_GUMP_test/feature_processed_encoding';

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
