%% Step1: calculate the temporal mean and standard deviation of the feature time series of CNN units
% clc;clear
% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

features = {'Ahat'}; %,'E'};

%make folder
folder_name = [saveroot,'/', 'feature_maps','/','6layers', '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end

for i = 0:5

    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        % calculate the temporal mean 
        secpath = [dataroot, '/', 'h5', '/', 'total', '/', sprintf('%s%d.h5', feature, i)];
        lay_feat = h5read(secpath,'/predimg');
        dim = size(lay_feat)

        lay_feat_mean = sum(lay_feat, length(dim)) / dim(end);

        disp(['save average for layer', num2str(i)])

        h5create([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature, i)],'/data',...
            [size(lay_feat_mean)],'Datatype','single');
        h5write([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature, i)],'/data', lay_feat_mean);

    end
end
   
% calculate the temporal standard deviation
    
for i = 0:5
    
    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        disp(['start std for layer', num2str(i)])
    
        secpath = [dataroot, '/', 'h5', '/', 'total', '/', sprintf('%s%d.h5', feature, i)];
        lay_feat = h5read(secpath,'/predimg');
        dim = size(lay_feat)
    
        lay_feat_mean = h5read([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature, i)],'/data');
    
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = lay_feat.^2;
        dim = size(lay_feat);
        
        lay_feat_std = zeros([dim(1:end-1),1]);
        lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));
    
        lay_feat_std = sqrt(lay_feat_std/(dim(end)-1));
        lay_feat_std(lay_feat_std==0) = 1;
        
        disp(['save average for layer', num2str(i)])
        
        h5create([folder_name, sprintf('%s%d_feature_maps_std.h5', feature, i)],'/data',...
            [size(lay_feat_std)],'Datatype','single');
        h5write([folder_name, sprintf('%s%d_feature_maps_std.h5', feature, i)],'/data',lay_feat_std);
        
        disp('start next layer!')
    end
end
disp('fin!')



%% Step2: PCA - Ahat
clc;clear

%dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers/seg3456';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

% % % % % % % % % % SVD-updating algorithm % % % % % % % % %

Niter = 2; % number of iteration to compute principle component
percp = 0.90; % explain 99% of the variance of every movie  segments
disp(['percp is ', num2str(percp)])


for i = 0:5
    disp(['layer is ', num2str(i)])
    
    % load mean and standard deviation
    disp(['load lay_feat for Ahat layer ', num2str(i)])    
    
    lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('Ahat%d_feature_maps_avg.h5', i)], '/data');
    lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('Ahat%d_feature_maps_std.h5', i)], '/data');
    
    % remove decimal point
    lay_feat_mean = single(lay_feat_mean);
    lay_feat_std = single(lay_feat_std);

    k0 = 0;
    for iter =  1  : Niter
        disp(['iteration #: ', num2str(iter)])
        
        %load lay_feat 
        disp(['load lay_feat for iteration ', num2str(iter), ' and for layer ' ,num2str(i)])
        secpath = [dataroot, '/', 'h5', '/', sprintf('Ahat%d.h5', i)];
        lay_feat = h5read(secpath,'/predimg');
    
        %normalization lay_feat
        disp('normalization begins')
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

        %reshape normalized lay_feat to features-by-time
        disp('reshape begins')
        dim = size(lay_feat);
        lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
        
        % clear dim
        % 
        % % remove decimal point
        % lay_feat = single(lay_feat);

        disp('amri_sig_isvd start')

        if iter == 1
            [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
            
            % remove decimal point
            % B = single(B);
            % S = single(S);
        else
            [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
           
            % remove decimal point
            % B = single(B);
            % S = single(S);
        end  
        
        % save up memory
        % disp(['start remove lay_feat for memory for iteration ',num2str(iter)])
        % clear lay_feat 
         
        % start next iteration
        % disp(['Iteration ', num2str(iter), ' completed for Ahat', num2str(i)]);
    end
   
    % save up memory 
    disp('start remove lay_feat, mean, std for memory BEFORE saving B and s ')
    clear lay_feat

    % diagnal S
    disp('diag(s) begins')
    s = diag(S);
    
    disp('start saving SVD_layer');
    % save principal components
    save([saveroot, '/', 'X', '/', sprintf('Ahat%d_feature_maps_svd_0.90.mat', i)], 'B', 's', '-v7.3');
        
    % save up memory
    disp(['start remove B s S for memory for layer ',num2str(i)])
    clear B s S 

    % start next layer
    disp(['SVD_layer for Ahat', num2str(i), ' saved']);
end

%% Step3: Processed the hrf - Ahat
clc; clear 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

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
for i=1:3
    disp(num2str(i))
    % Dimension reduction for features
    disp('load mean and std')
    lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('Ahat%d_feature_maps_avg.h5', i)], '/data');
    lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('Ahat%d_feature_maps_std.h5', i)], '/data');
    load([saveroot,'/', 'X', '/', sprintf('Ahat%d_feature_maps_svd_0.90.mat', i) ], 'B');

    %load lay_feat 
    disp('load lay_feat')
    secpath = [dataroot, '/', sprintf('Ahat%d.h5', i)];
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
    
    filename = [sprintf('Ahat%d_feature_maps_svd_0.90_hrf.mat', i)];
    save([saveroot,'/', 'X', '/', 'Ahat', '/',  filename], 'ts', '-v7.3');
    
    disp(['SVD_layer for Ahat', num2str(i), ' saved']);

    
    clear ts lay_feat
end

%% Step2: PCA - E 
clc;clear

%dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

% % % % % % % % % % SVD-updating algorithm % % % % % % % % %

Niter = 2; % number of iteration to compute principle component
percp = 0.90; % explain 99% of the variance of every movie segments
disp(['percp is ', num2str(percp)])


for i = 0:5
    disp(['layer is ', num2str(i)])
    % Step 1: Calculate mean and standard deviation
    disp('load mean and std')
    
    lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('E%d_feature_maps_avg.h5', i)], '/data');
    lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('E%d_feature_maps_std.h5', i)], '/data');
    disp(['before dim reduced size of lay_feat_mean is ', num2str(size(lay_feat_mean))]);
    disp(['before dim reduced size of lay_feat_std is ', num2str(size(lay_feat_std))]);
    
    half_channel = size(lay_feat_mean,1);
    half_channel = half_channel/2;
    
    lay_feat_mean_1st_half= lay_feat_mean(1:half_channel, :,:,:);
    lay_feat_mean_2nd_half= lay_feat_mean(half_channel+1:end,:,:,:); 
    lay_feat_mean = lay_feat_mean_1st_half + lay_feat_mean_2nd_half; 
    clear lay_feat_mean_1st_half lay_feat_mean_2nd_half
    disp(['after dim reduced size of lay_feat_mean is ', num2str(size(lay_feat_mean))]);

     
    lay_feat_std_1st_half = lay_feat_std(1:half_channel, :,:,:);
    lay_feat_std_2nd_half = lay_feat_std(half_channel+1:end,:,:,:); 
    lay_feat_std = lay_feat_std_1st_half + lay_feat_std_2nd_half; 
    clear lay_feat_std_1st_half lay_feat_std_2nd_half
    disp(['after dim reduced size of lay_feat_std is ', num2str(size(lay_feat_std))]);
    
    k0 = 0;
    for iter =  1  : Niter
        disp(['iteration #: ', num2str(iter)])
        
        %load lay_feat 
        disp('load lay_feat')
        secpath = [dataroot, '/', 'h5','/',sprintf('E%d.h5', i)];
        lay_feat = h5read(secpath,'/predimg');
         
        half_channel_A= size(lay_feat_A,1);
        half_channel_A = half_channel_A/2;
    
        lay_feat_1st_half_A= lay_feat(1:half_channel_A,:,:,:); 
        lay_feat_2nd_half_A = lay_feat(half_channel_A+1:end,:,:,:);
        lay_feat_A1 = lay_feat_1st_half_A+lay_feat_2nd_half_A;
        clear lay_feat_1st_half lay_feat_2nd_half
        disp(['after dim reduced size of lay_feat is ', num2str(size(lay_feat))]);
        
        %normalization lay_feat
        disp('normalization begins')
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

        %reshape normalized lay_feat to features-by-time
        disp('reshape begins')
        dim = size(lay_feat);
        lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
        
        disp('amri_sig_isvd start')

        if iter == 1
            [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
        else
            [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
        end  

        disp(['Iteration ', num2str(iter), ' completed for E', num2str(i)]);
    end
    
    disp('diag(s) begins')
    
    s = diag(S);
    
    disp('start saving SVD_layer');
    % save principal components
    save([saveroot, '/', 'X', '/', 'PCA', '/', 'Ahat', '/', sprintf('E%d_feature_maps_svd_0.90.mat', i)], 'B', 's', '-v7.3');
    disp(['SVD_layer for E', num2str(i), ' saved']);
    
    clear B s S lay_feat
end
%% Step3: Processed the hrf - E
clc; clear 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

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
for i=0
    disp(num2str(i))
    % Dimension reduction for features
    disp('load mean and std')
    lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('E%d_feature_maps_avg.h5', i)], '/data');
    lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('E%d_feature_maps_std.h5', i)], '/data');
    disp(['before dim reduced size of lay_feat_mean is ', num2str(size(lay_feat_mean))]);
    disp(['before dim reduced size of lay_feat_std is ', num2str(size(lay_feat_std))]);
    
    half_channel = size(lay_feat_mean,1);
    half_channel = half_channel/2;
    
    lay_feat_mean_1st_half= lay_feat_mean(1:half_channel, :,:,:);
    lay_feat_mean_2nd_half= lay_feat_mean(half_channel+1:end,:,:,:); 
    lay_feat_mean = lay_feat_mean_1st_half + lay_feat_mean_2nd_half; 
    clear lay_feat_mean_1st_half lay_feat_mean_2nd_half
    disp(['after dim reduced size of lay_feat_mean is ', num2str(size(lay_feat_mean))]);

     
    lay_feat_std_1st_half = lay_feat_std(1:half_channel, :,:,:);
    lay_feat_std_2nd_half = lay_feat_std(half_channel+1:end,:,:,:); 
    lay_feat_std = lay_feat_std_1st_half + lay_feat_std_2nd_half; 
    clear lay_feat_std_1st_half lay_feat_std_2nd_half
    disp(['after dim reduced size of lay_feat_std is ', num2str(size(lay_feat_std))]);

    %load svd 
    load([saveroot,'/', 'X', '/','PCA', '/', 'Ahat','/', sprintf('E%d_feature_maps_svd_0.90.mat', i) ], 'B');

    %load lay_feat 
    disp('load lay_feat')
    secpath = [dataroot, '/', sprintf('E%d.h5', i)];
    lay_feat = h5read(secpath,'/predimg');
     
    half_channel = size(lay_feat,1);
    half_channel = half_channel/2;

    lay_feat_1st_half= lay_feat(1:half_channel,:,:,:); 
    lay_feat_2nd_half = lay_feat(half_channel+1:end,:,:,:);
    lay_feat = lay_feat_1st_half+lay_feat_2nd_half;
    clear lay_feat_1st_half lay_feat_2nd_half
    disp(['after dim reduced size of lay_feat is ', num2str(size(lay_feat))]);
        
    %regularization
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
    
    filename = [sprintf('E%d_feature_maps_svd_0.90_hrf.mat', i)];
    save([saveroot,'/', 'X', '/', 'PCA', '/', 'Ahat', '/',  filename], 'ts', '-v7.3');
    
    disp(['SVD_layer for E', num2str(i), ' saved']);

    
    clear ts lay_feat
end


%% Step4: divided into seg 1-8 (approx. 15min each)
clc; clear
% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

for i=0:3
    %open ts
    load([saveroot,'/', 'X', '/',  'PCA', '/', 'E', '/', sprintf('E%d_feature_maps_svd_0.90_hrf.mat',i)], 'ts');

    % Size of the ts array
    ts_size = size(ts);

    Nv = [451, 441, 438, 488, 462, 439, 542, 338];

    % Create a cell array to store the segments
    segments = cell(1, 8);

    % Calculate the start and end indices for each segment
    start_indices = [1, cumsum(Nv(1:end-1)) + 1];
    end_indices = cumsum(Nv);

    % Extract each segment from ts and store it in the cell array
    for seg = 1:8
        segments = ts(start_indices(seg):end_indices(seg),:);

        % Save the segment into a separate file
        segment_filename = [sprintf('E%d_svd0.9hrf_seg%d.mat', i, seg)];
        save([saveroot,'/', 'X', '/', 'PCA', '/', 'E', '/', segment_filename], 'segments', '-v7.3');
    end

end


%% Step5.1: Train Encoding with GUMP fMRI processing w/ fMRIPrep + 3mm smoothing - 59k
% % Load Y
% clc;clear 
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_runs = 8;
% sub_cii = cell(numel(num_subjects), 1);
% subcortex_no = 1:59412;
% existing_subjects = {};  % To store the subjects that are found
% 
% for subject_idx = 1:numel(num_subjects)
%     subject = num_subjects{subject_idx};
%     subject_cii = cell(1, num_runs);
%     subject_exists = true;  % Flag to indicate if subject exists
% 
%     for run = 1:num_runs
%         folder_path = sprintf('/combinelab/03_user/jungmin/01_data/03_Gump/02_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-%d', subject, subject, subject, run);
% 
%         files = dir(fullfile(folder_path, '*s3.dtseries.nii'));
%         cii_run = cell(1, numel(files));
% 
%         for i = 1:numel(files)
%             file_path = fullfile(folder_path, files(i).name);
%             cii = ciftiopen(file_path, 'wb_command');
%             cii_run{i} = cii.cdata(subcortex_no, :);
%         end
% 
%         subject_cii{run} = cat(2, cii_run{:});
% 
%         % Check if subject data is empty
%         if isempty(subject_cii{run})
%             subject_exists = false;
%             break;  % Exit the loop if subject data is empty for any run
%         end
%     end
% 
%     % Store subject data if it exists for all runs
%     if subject_exists
%         existing_subjects = [existing_subjects, subject];  % Append to existing_subjects
%         sub_cii{subject_idx} = subject_cii;
%     end
% end
% 
% % Save existing subjects
% save_file = [saveroot, '/', 'Y', '/', 'fMRI_59k_cortex'];
% save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');

%% Step5.1: Train Encoding with GUMP fMRI processing w/ fMRIPrep + 3mm smoothing - 10k
% % Load Y
% clc;clear 
% addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_runs = 8;
% sub_cii = cell(numel(num_subjects), 1);
% existing_subjects = {};  % To store the subjects that are found
% 
% for subject_idx = 1:numel(num_subjects)
%     subject = num_subjects{subject_idx};
%     subject_cii = cell(1, num_runs);
%     subject_exists = true;  % Flag to indicate if subject exists
% 
%     for run = 1:num_runs
%         folder_path = sprintf('/zeus-share/local_raid4/user/sunghyoung/01_project/04_encoding_model/02_data/Y/%s/', subject);
% 
%         files = dir(fullfile(folder_path, sprintf('Y_%s_seg-%d_cortex_10k.mat', subject, run)));
%         disp(files.name);
% 
%         cii_run = cell(1, numel(files));
% 
%         for i = 1:numel(files)
%             file_path = fullfile(folder_path, files(i).name);
%             cii = load(file_path);
%             cii_run{i} = cii.data;
%         end
% 
%         subject_cii{run} = cat(2, cii_run{:});
% 
%         % Check if subject data is empty
%         if isempty(subject_cii{run})
%             subject_exists = false;
%             break;  % Exit the loop if subject data is empty for any run
%         end
%     end
% 
%     % Store subject data if it exists for all runs
%     if subject_exists
%         existing_subjects = [existing_subjects, subject];  % Append to existing_subjects
%         sub_cii{subject_idx} = subject_cii;
%     end
% end
% 
% % Save existing subjects
% save_file = [saveroot, '/', 'Y', '/', 'fMRI_10k_cortex'];
% save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');

%% Step5.2:X model featuremaps 
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
features = { 'Ahat', 'E'};


for i = 0:3
    X = cell(1, 7);
    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature);

        for seg = 1:8
            filename = sprintf('%s/X/PCA/%s/%s%d_svd0.9hrf_seg%d.mat', saveroot, feature, feature,i, seg);
            disp([sprintf('%s/%s%d_svd0.9hrf_seg%d.mat', feature, feature,i, seg)])

            loaded_data = load(filename);
            X{seg} = loaded_data.segments;  % Replace 'semgnet' with the desired field name
        end
    name = sprintf('%s%d_23_45678',feature, i);
    concatenated.(name)= cat(1, X{2},X{3},X{4},X{5},X{6},X{7},X{8});
    eval([name, ' = concatenated.(name);']);
    end
end


%Voxelwise_encoding 
lambda = [0.1:0.2:0.9];
nfold = 9; %from paper 

%load Y
load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 1:15
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
   sub_cii1_7 = cat(2, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
   
   for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature);
        
        for layer=0:3
            name = sprintf('%s%d_23_45678',feature, layer);
            argument = concatenated.(name);  % Assuming Ahat_concatenated is the struct from previous code
    
            [W, Rmat, Lambda] = voxelwise_encoding(argument, sub_cii1_7', lambda, nfold);
    
            % save W (feature x voxel)
            name = [subject, '_', feature, num2str(layer), '_W_svd0.9_s3.mat'];
            save([saveroot,'/', 'W','/','PCA','/', '10k', '/','seg1','/', feature, '/', name], 'W', '-v7.3');  
        end
   end
end
%% Step5.2:X model featuremaps  - Error
% clc; clear
% disp('Load model featuremaps')
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% 
% for i = 0:3
%     X = cell(1, 7);
%     for seg = 1:7
%         filename = sprintf('%s/X/PCA/E/E%d_svd0.9hrf_seg%d.mat', saveroot, i, seg);
%         loaded_data = load(filename);
%         X{seg} = loaded_data.segments;  % Replace 'semgnet' with the desired field name
%     end
%     name = sprintf('E%d_1_7', i);
%     E_concatenated.(name)= cat(1, X{:});
%     eval([name, ' = E_concatenated.(name);']);
% end
% 
% 
% %Voxelwise_encoding 
% lambda = [0.1:0.2:0.9];
% nfold = 9; %from paper 
% 
% %load Y
% load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
%     disp(['Voxelwise_encoding start sub', num2str(i)])
% 
%     sub_cii1_7 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7});
% 
%     for feature=0:3
%         E_field_name = sprintf('E%d_1_7', feature);
%         E_argument = E_concatenated.(E_field_name);  % Assuming Ahat_concatenated is the struct from previous code
% 
%         [W, Rmat, Lambda] = voxelwise_encoding(E_argument, sub_cii1_7', lambda, nfold);
% 
%         % save W (feature x voxel)
%         name = [subject, '_', 'E', num2str(feature), '_W_svd0.9_s3.mat'];
%         save([saveroot,'/', 'W','/','PCA','/', '10k', '/', 'E', '/',name], 'W', '-v7.3');  
%     end
% end


%% Step6.1: Load test Y and X

% X model featuremaps 
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
features = { 'Ahat', 'E'};


for i = 0:3
    disp(['layer is ', num2str(i)])
    X = cell(1, 7);
    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature);
         for seg = 1:8
            filename = sprintf('%s/X/PCA/%s/%s%d_svd0.9hrf_seg%d.mat', saveroot, feature, feature,i, seg);
            loaded_data = load(filename);
            X{seg} = loaded_data.segments;  % Replace 'semgnet' with the desired field name
        
            name = sprintf('%s%d_seg%d', feature, i, seg);
            concatenated.(name)= cat(1, X{seg});
            eval([name, ' = concatenated.(name);']);
        end
    end
end

disp('make Y_hat')

load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

for i = 1:15
    subject = num_subjects{i}
    disp(['Voxelwise_encoding start ', subject])    
    
    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature); 
        for seg=1:8
            folder_name = [saveroot,'/', 'Y_hat','/','PCA', '/', '10k', '/', sprintf('seg%d',seg),'/', feature]
            if not(exist(folder_name,'dir'))
                mkdir(folder_name)
            end  
             for layer=0:3
                name = sprintf('%s%d_seg%d', feature, layer, seg)
                X = concatenated.(name); 
    
                W = load([saveroot,'/', 'W', '/', 'PCA',  '/', '10k', '/',  sprintf('seg%d',seg), '/', feature, '/', subject, '_', feature, num2str(layer), '_W_svd0.9_s3.mat']);
                name = 'Y_prediction';
                eval([name, '= X* W.W;']);
    
                %save Y_hat
                Yhat = eval(name);
                name2 = strcat(subject,'_', feature , num2str(layer), '_pred_svd0.90.mat');
               
                save([saveroot,'/', 'Y_hat','/','PCA', '/', '10k','/', sprintf('seg%d',seg),  '/',  feature, '/', name2], 'Yhat', '-v7.3');
             end
        end
    end
end

%% Step 6.1: concat Yhat for all segments + correalting with total movie fMRI
clc; clear

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

for i = 4:15
    subject = num_subjects{i}
    disp(i)
    disp(['Voxelwise_encoding start ', subject])    
    
    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer=0:3 
            for seg = 1:8
                Y_hat = load([saveroot,'/', 'Y_hat','/','PCA', '/', '10k', '/', sprintf('seg%d',seg), '/', feature, '/', strcat(subject,'_', feature , num2str(layer), '_pred_svd0.90.mat')]);
                disp([saveroot,'/', 'Y_hat','/','PCA', '/', '10k', '/',sprintf('seg%d',seg), '/', feature, '/', strcat(subject,'_', feature , num2str(layer), '_pred_svd0.90.mat')])
                X{seg} = Y_hat.Yhat;
            end
            Y_pred = sprintf('%s%d_seg1_8', feature, layer)
            concatenated.(Y_pred)= cat(1, X{:});
    
            %correaltions
            %load Y 
            sub_cii1_8 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
            [correlations, p_values] = auto_corr(sub_cii1_8',concatenated.(Y_pred));

            correlations = correlations';

            % save correlation (feature x voxel)
            disp('save correlations')
           
            name = strcat(subject,'_', feature , num2str(layer),'_corr_svd0.90_s3', '.mat')
            save([saveroot,'/', 'correlations','/', 'PCA', '/', name], 'correlations', '-v7.3');
        end
    end
end
%% Step7: make dtseries 
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Aaht','E'};

for i = 1:15
    subject =  num_subjects{i}
    X={} ;

    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer=0:3
            % Load cifti -10k
            folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
            files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
    
            file_path = fullfile(folder_path, files.name);
            cii = ciftiopen(file_path, 'wb_command');
            cii = rmfield(cii, 'cdata');
    
            %load correaltion
            load([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/', subject,'_', feature , num2str(layer),'_corr_svd0.90_s3']);
            disp([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/', subject,'_', feature , num2str(layer),'_corr_svd0.90_s3']);
            
            % Update cdata and diminfo
            cii.cdata = correlations;
            cii.diminfo{1, 1}.length = size(correlations, 1);
            cii.diminfo{1, 2}.length = size(correlations, 2);
            cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
            
            % Concatenate cdata for each layer
            if isempty(X)
                X = cii;
            else
                X.cdata = cat(2, X.cdata, cii.cdata);
            end
        end
        
        % Save as dtseries
        save_file_name = ([subject,'_', feature, '.dtseries.nii']);
        ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','/', save_file_name), 'wb_command');
    end
end

%% GUMP dir
clc;clear

saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
num_layer= 3;
features = {'Ahat','E'};

Y_hat = cell(num_features, 1);


for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature);
    for layer=0:3
        sum_values = 0;
        for i = 1:numel(num_subjects)
            subject = num_subjects{i};
            
            % Load correlation
            load(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/',   [subject, '_', feature, num2str(layer), '_corr_svd0.90_s3.mat']));
            disp(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/',   [subject, '_', feature, num2str(layer), '_corr_svd0.90_s3.mat']));
            
            sum_values = sum_values + correlations';
        end
    Y_hat{layer+1} = sum_values / numel(num_subjects);
    end
    save(fullfile(saveroot, 'correlations', 'PCA', '10k', sprintf('%s_avg_corrss_svd0.90_s3.mat', feature)), 'Y_hat', '-v7.3');
end

%% Step7: make dtseries 
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
features = {'Ahat','E'};

X={} ;

for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature);
    for layer=0:3

        % Load cifti -10k
        folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
        files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
    
        file_path = fullfile(folder_path, files.name);
        cii = ciftiopen(file_path, 'wb_command');
        cii = rmfield(cii, 'cdata');
    
        %load correaltion
        load([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/', sprintf('%s_avg_corrss_svd0.90_s3.mat', feature)]);
        disp([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/',sprintf('%s_avg_corrss_svd0.90_s3.mat', feature)])
        Y_hat = Y_hat{layer+1,:};
    
        % Update cdata and diminfo
        cii.cdata = Y_hat';
        cii.diminfo{1, 1}.length = size(Y_hat', 1);
        cii.diminfo{1, 2}.length = size(Y_hat', 2);
        cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
        
        % Concatenate cdata for each feature
        if isempty(X)
            X = cii;
        else
            X.cdata = cat(2, X.cdata, cii.cdata);
        end  
    end
    
    % Save as dtseries
    save_file_name = sprintf('%s_avg.dtseries.nii', feature);
    ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA',  save_file_name), 'wb_command');
    % initialize X cell
    X={} ; 
end

%% step6.2: correaltion between Y_predict and actual Y for test - single segment
% clc; clear
% disp('correaltion and p_values compute start')
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% %load Y
% load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
% 
%     disp(['correalting' , subject])
%     for feature= 0:3
% 
%         Y_pred = load([saveroot,'/', 'Y_hat','/','PCA', '/','10k', '/', 'E', '/',subject,'_', 'E' , num2str(feature), '_pred_svd0.90_s3', '.mat']);
%         Y_pred = Y_pred.Yhat;
% 
%         [correlations, p_values] = auto_corr(sub_cii{i}{8}',Y_pred);
% 
%         correlations = correlations';
% 
%         % save correlation (feature x voxel)
%         disp('save correlations')
% 
%         name = strcat(subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3', '.mat');
%         save([saveroot,'/', 'correlations','/', 'PCA', '/','10k', '/', 'E', '/', name], 'correlations', '-v7.3');
% 
%         % Find p_values < 0.05
%         p_values=p_values';
%         significant_p = p_values <= 0.05;
% 
%         % Replace numbers above 0.05 with 0
%         p_values(~significant_p) = 0;
% 
%         % Find the count of numbers in p_values
%         count = nnz(p_values);
% 
%         % save significant_p (feature x voxel)
%         disp('save p_values')
% 
%         name = strcat(subject, '_', 'E', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat');
%         save([saveroot,'/', 'p_values','/', 'PCA', '/','10k', '/', 'E', '/', name], 'p_values', '-v7.3');
%     end
% end
% % [correlations, p_values] = auto_corr(X1_56',X_pred56_sub1);
% % correlations = correlations';
% % p_values = p_values';
% 
% % % Find p_values < 0.05
% % significant_p = p_values < 0.05;
% % 
% % % Replace numbers above 0.05 with 0
% % p_values(~significant_p) = 0;
% % 
% % % Find the count of numbers in p_values
% % count = nnz(p_values);
% 
%% Step7: make dtseries 
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
%     X={} ;
% 
%     for feature=0:3
% 
% %         % Load cifti -59k
% %         folder_path = sprintf('/combinelab/03_user/jungmin/01_data/03_Gump/02_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-8',subject, subject, subject);
% %         files = dir(fullfile(folder_path, '*s3.dtseries.nii'));
% %         file_path = fullfile(folder_path, files.name);
% %         
% %         cii = ciftiopen(file_path, 'wb_command');
% %         cii = rmfield(cii, 'cdata');
% 
% 
%         % Load cifti -10k
%         folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%         files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%         file_path = fullfile(folder_path, files.name);
%         cii = ciftiopen(file_path, 'wb_command');
%         cii = rmfield(cii, 'cdata');
% 
%         %load correaltion
%         load([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/','E', '/', subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']);
%         disp([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/','E', '/', subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']);
% 
%         % Update cdata and diminfo
%         cii.cdata = correlations;
%         cii.diminfo{1, 1}.length = size(correlations, 1);
%         cii.diminfo{1, 2}.length = size(correlations, 2);
%         cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%         % Concatenate cdata for each feature
%         if isempty(X)
%             X = cii;
%         else
%             X.cdata = cat(2, X.cdata, cii.cdata);
%         end
%     end
% 
%     % Save as dtseries
%     save_file_name = (['E_', subject,'.dtseries.nii']);
%     ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','10k', '/', save_file_name), 'wb_command');
% end
% 
% %% Step 8.5:avg .mat
% clc;clear
% 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_features = 3;
% 
% Y_hat = cell(num_features, 1);
% 
% for feature = 0:num_features
%     sum_values = 0;
%     for i = 1:numel(num_subjects)
%         subject = num_subjects{i};
% 
%         % Load correlation
%         load(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'E', '/',  [subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']));
%         disp(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'E', '/',   [subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']));
% 
%         sum_values = sum_values + correlations';
%     end
%     Y_hat{feature+1} = sum_values / numel(num_subjects);
% end
% save(fullfile(saveroot, 'correlations', 'PCA', '10k','E', 'E_avg_corr_svd0.90_s3.mat'), 'Y_hat', '-v7.3');
% 
% 
% 
%% Step 8.5: make avg dtseries 
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% X={} ;
% 
% for feature=0:3
% 
% %         % Load cifti -59k
% %         folder_path = sprintf('/combinelab/03_user/jungmin/01_data/03_Gump/02_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-8',subject, subject, subject);
% %         files = dir(fullfile(folder_path, '*s3.dtseries.nii'));
% %         file_path = fullfile(folder_path, files.name);
% %         
% %         cii = ciftiopen(file_path, 'wb_command');
% %         cii = rmfield(cii, 'cdata');
% 
% 
%     % Load cifti -10k
%     folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%     files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%     file_path = fullfile(folder_path, files.name);
%     cii = ciftiopen(file_path, 'wb_command');
%     cii = rmfield(cii, 'cdata');
% 
%     %load correaltion
%     load([saveroot,'/', 'correlations','/','PCA', '/','10k','/', 'Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);
%     Y_hat = Y_hat{feature+1,:};
% 
%     % Update cdata and diminfo
%     cii.cdata = Y_hat';
%     cii.diminfo{1, 1}.length = size(Y_hat', 1);
%     cii.diminfo{1, 2}.length = size(Y_hat', 2);
%     cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%     % Concatenate cdata for each feature
%     if isempty(X)
%         X = cii;
%     else
%         X.cdata = cat(2, X.cdata, cii.cdata);
%     end  
% end
% 
% % Save as dtseries
% save_file_name = 'Ahat_avg.dtseries.nii';
% ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','10k',  save_file_name), 'wb_command');
% 
%% Step 9:avg .mat -> p_value
% clc;clear
% 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_features = 3;
% 
% correlation = cell(num_features, 1);
% p_value = cell(num_features, 1);
% 
% for feature = 0:num_features
%     pval_sum_values = 0;
%     corr_sum_values = 0;
%     for i = 1:numel(num_subjects)
%         subject = num_subjects{i};
% 
%         % Load correlation
%         load(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',  [subject, '_', 'Ahat', num2str(feature), '_corr_svd0.90_s3', '.mat']));
%         disp(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',   [subject, '_', 'Ahat', num2str(feature), '_corr_svd0.90_s3', '.mat']));        
% 
%         % Load p-value
%         load(fullfile(saveroot, 'p_values', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',  [subject, '_', 'Ahat', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat']));
%         disp(fullfile(saveroot, 'p_values', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',   [subject, '_', 'Ahat', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat']));
% 
%         zero_indices = p_values == 0;
%         correlations(zero_indices) = 0;
%         corr_sum_values = corr_sum_values + correlations';
% 
%         pval_sum_values = pval_sum_values + p_values';
%     end
%     correlation{feature+1} = corr_sum_values / numel(num_subjects);
%     p_value{feature+1} = pval_sum_values / numel(num_subjects);
% end
% save(fullfile(saveroot, 'p_values', 'PCA', '10k','Ahat', 'Ahat_avg_corr_svd0.90_s3.mat'), 'correlation', '-v7.3');
% save(fullfile(saveroot, 'p_values', 'PCA', '10k','Ahat', 'Ahat_avg_p_values_svd0.90_s3.mat'), 'p_value', '-v7.3');
% 
%% Step9: make individual dtseries - pvalue
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/';
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
% 
%     thresholds = {0.05, 0.01, 0.001};
% 
%     for j = 1:length(thresholds)
%         threshold = thresholds{j};
%         X={} ;
% 
%         for feature=0:3
% 
%             %load p-value
%             load([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);
%             disp([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);   
% 
%             %adjust p_values
%             pValue = p_value{feature+1};
%             pValue(pValue > threshold) = 0;  
% 
%             % Load correlation
%             load([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);
%             disp([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);  
% 
%             zero_indices = pValue == 0;
%             correlations = correlation{feature+1};
%             correlations(zero_indices) = 0;            
% 
%             % Load cifti -10k
%             folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%             files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%             file_path = fullfile(folder_path, files.name);
%             cii = ciftiopen(file_path, 'wb_command');
%             cii = rmfield(cii, 'cdata');
% 
%             % Update cdata and diminfo
%             cii.cdata = correlations';
%             cii.diminfo{1, 1}.length = size(correlations, 2); 
%             cii.diminfo{1, 2}.length = 4; % the number of layers
%             cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%             % Concatenate cdata for each feature
%             if isempty(X)
%                 X = cii;
%             else
%                 X.cdata = cat(2, X.cdata, cii.cdata);
%             end          
% 
% 
%         end
%     cii.cdata = X.cdata;
% 
%     % Save as dtseries
%     save_file_name = sprintf(['%s_Ahat_pvalues%s.dtseries.nii'], subject, num2str(threshold));
%     ciftisavereset(cii, fullfile(saveroot, 'dt_result', 'PCA','10k',  save_file_name), 'wb_command');   
%     end 
% end
% 
%% Step 9: make avg dtseries -> p_value
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% thresholds = {0.05, 0.01, 0.001};
% 
% for j = 1:length(thresholds)
%     threshold = thresholds{j};
%     X={} ;
% 
%     for feature=0:3
% 
%         % Load cifti -10k
%         folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%         files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%         file_path = fullfile(folder_path, files.name);
%         cii = ciftiopen(file_path, 'wb_command');
%         cii = rmfield(cii, 'cdata');
% 
%         %load p-value
%         load([saveroot,'/', 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);
%         disp([saveroot,'/', 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);   
% 
%         %adjust p_values
%         pValue = p_value{feature+1};
%         pValue(pValue > threshold) = 0;  
% 
%         %load correaltion
%         load([saveroot,'/', 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);
% 
%         zero_indices = pValue == 0;
%         correlations = correlation{feature+1};
%         correlations(zero_indices) = 0;          
% 
%         % Update cdata and diminfo
%         cii.cdata = correlations';
%         cii.diminfo{1, 1}.length = size(correlations, 2); 
%         cii.diminfo{1, 2}.length = 4; % the number of layers
%         cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%         % Concatenate cdata for each feature
%         if isempty(X)
%             X = cii;
%         else
%             X.cdata = cat(2, X.cdata, cii.cdata);
%         end    
%     end
% 
%     cii.cdata = X.cdata;
%     % Save as dtseries
%     save_file_name = sprintf(['Ahat_avg_pvalues-%s.dtseries.nii'],num2str(threshold));
%     ciftisavereset(cii, fullfile(saveroot, 'dt_result', 'PCA','10k',  save_file_name), 'wb_command');
% end