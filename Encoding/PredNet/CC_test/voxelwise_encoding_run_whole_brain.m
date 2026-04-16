clc; clear;

%% Add paths for fMRI processing
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

%% Load the processed training fMRI responses
savepath = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/fmri';
load(fullfile(savepath, 'training_fmri.mat'), 'train_data', 'subject_list');

% Check the dimensions of the training data
[n_voxels, n_timepoints, n_train_subjects] = size(train_data);
disp(['Training data dimensions: ', num2str(n_voxels), ' voxels, ', num2str(n_timepoints), ' timepoints, ', num2str(n_train_subjects), ' subjects']);

%% Prepare for voxelwise encoding
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/06_Cam-CAN/result/modified_prednet/h5/';
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 
layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c'; 

lambda = 0.1:0.2:0.9;  % Regularization parameters
nfold = 9;  % Number of folds for cross-validation
clc;
%% Perform whole-brain voxelwise encoding for each layer
clc;

% layername = {'/Ahat0';'/Ahat1'; '/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
% layernumber = 'layers5_32c'; 

for lay = 1:length(layername)
    disp(['Processing Layer: ', num2str(lay)]);

    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];
        save_tmp = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/Ahat/';
    elseif contains(layername{lay}, 'E')
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];
        save_tmp = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/E/';
    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end

    % Load the PredNet feature maps for the current layer
    secpath = fullfile(saveroot, 'PredNet_feature_maps_pcareduced.h5');
    Y_train = h5read(secpath, [layername{lay}, '/data']);  % #timepoints by #components
    [n_timepoints, n_components] = size(Y_train);

    % Ensure Y_train matches the format expected by voxelwise_encoding (timepoints x components)
    if n_timepoints > 193
        Y_train = Y_train(1:193, :);  % Trim to 193 timepoints
        disp('Y_train trimmed to match train_data timepoints');
    elseif n_timepoints < 193
        disp('Warning: train_data has more timepoints than Y_train');
    end
    [n_timepoints, n_components] = size(Y_train);
    disp(['Layer ', num2str(lay), ': Data dimensions - ', num2str(n_timepoints), ' timepoints, ', num2str(n_components), ' components']);
    
    W_all = cell(n_train_subjects, 1);  % Store weights for each subject
    
    for subj = 1:n_train_subjects
       
        % Extract the subject's fMRI data for training 
        fmri_data_subj = train_data(:,:,subj);  % Size: (n_voxels, n_timepoints)
        disp(['Y_train size: ', num2str(size(Y_train)), '  fmri_data size: ', num2str(size(fmri_data_subj))]);

        % Perform voxelwise encoding 
        [W, Rmat, Lambda] = voxelwise_encoding(Y_train, fmri_data_subj', lambda, nfold);

        % Store the weights for later aggregation
        W_all{subj} = W;

        % Optionally, you can store Rmat if needed for further analysis
        % Rmat_all{subj} = Rmat;

        % Save the results for this layer and subject
        % file_name = ['W_lambda5_for_sub_', num2str(subj), '_', layername{lay}(2:end), '.mat'];
        % save_feature_name = extractBetween(saveroot, 'layers4_32c/', '/');
        % disp(["saveroot for feature:  ", save_feature_name{1}]);
        % save(fullfile(saveroot, file_name), 'W', '-v7.3');
    end

    % Now, aggregate the weights across subjects by averaging
    W_aggregated = zeros(size(W_all{1}));  % Initialize aggregation matrix (same size as W)

    % Sum the weights across all subjects
    for subj = 1:n_train_subjects
        W_aggregated = W_aggregated + W_all{subj};  % Sum the weights
    end

    % Average the weights across all subjects
    W_aggregated = W_aggregated / n_train_subjects;  % Size: (n_voxels, n_components)


    % Save the aggregated weights for this layer
    file_name = ['W_aggregated_avg_67sub_', layername{lay}(2:end), '.mat'];
    % save(fullfile(saveroot, 'whole-brain', save_folder, file_name), 'W_aggregated', '-v7.3');
    save(fullfile(save_tmp, 'whole-brain', file_name), 'W_aggregated', '-v7.3');

end

%% Load the processed testing fMRI responses
savepath = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/fmri';
load(fullfile(savepath, 'testing_fmri.mat'), 'test_data', 'subject_list');

% Check the dimensions of the testing data
[n_voxels, n_timepoints, n_test_subjects] = size(test_data);
disp(['Testing data dimensions: ', num2str(n_voxels), ' voxels, ', num2str(n_timepoints), ' timepoints, ', num2str(n_test_subjects), ' subjects']);

%% Y_test and correlation_all_test
clc;
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c'; 

all_corr_Ahat = cell(4, 1); 
all_corr_E = cell(4, 1);     

for lay = 1:length(layername)
    
    % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 4;  % N\ew index for E layers
    else
        e_index = lay;  % Keep the original index for Ahat layers
    end
    
    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];
        save_tmp = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/Ahat/';
    elseif contains(layername{lay}, 'E')
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];
        save_tmp = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/E/';
    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end
    
    secpath = fullfile(saveroot, 'PredNet_feature_maps_pcareduced.h5');
    if exist(secpath, 'file')
        Y_test = h5read(secpath, [layername{lay}, '/data']);  % Load test feature maps
    else
        disp(['File does not exist: ', secpath]);
        continue;  % Skip if file doesn't exist
    end

    for test_subj = 1:n_test_subjects
        % Load the aggregated weights matrix for the current layer
        W_filename = [save_tmp, 'whole-brain/', 'W_aggregated_avg_67sub_', layername{lay}(2:end), '.mat'];
        
        % Error handling: Ensure that the weights matrix exists
        if ~exist(W_filename, 'file')
            disp(['File does not exist: ', W_filename]);
            continue;
        end
        
        load(W_filename, 'W_aggregated');  % W_aggregated is (components, voxels)
        
        % Ensure Y_test and W_aggregated dimensions match
        if size(Y_test, 2) ~= size(W_aggregated, 1)
            error('Dimension mismatch: Y_test and W_aggregated must have matching second and first dimensions');
        end
        
        % Ensure Y_train matches the format expected by voxelwise_encoding (timepoints x components)
        [n_timepoints, n_components] = size(Y_test);
        if n_timepoints > 193
            Y_test = Y_test(1:193, :);  % Trim to 193 timepoints
            disp('Y_test trimmed to match train_data timepoints');
        elseif n_timepoints < 193
            disp('Warning: train_data has more timepoints than Y_test');
        end
        
        % Compute predicted BOLD using the feature maps and the aggregated weights matrix
        pred_fmri = Y_test * W_aggregated;  % Predicted BOLD: (timepoints, voxels)
        
        % Load measured BOLD
        fmri_data_subj = test_data(:,:,test_subj);
        fmri_data_subj = fmri_data_subj';
    
        % Check dimensions match
        if size(pred_fmri, 1) ~= size(fmri_data_subj, 1)
            error('Timepoints mismatch: pred_fmri and fmri_data_subj must have the same number of timepoints');
        end

        % Calculate correlation scores for each voxel
        correlation_scores = zeros(1, size(pred_fmri, 2));  % Preallocate for correlations
        for v = 1:size(pred_fmri, 2)  % Loop over voxels
            correlation_scores(v) = corr(pred_fmri(:, v), fmri_data_subj(:, v));  % Compute correlation for each voxel
        end
        
        if contains(layername{lay}, 'Ahat')
            all_corr_Ahat{e_index}{test_subj} = correlation_scores'; 
        elseif contains(layername{lay}, 'E')
            all_corr_E{e_index}{test_subj} = correlation_scores'; 
        end
    end
end

% Save the all_corr for 'Ahat' and 'E' separately
save(fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/Ahat/whole-brain/all_corr_Ahat.mat'), 'all_corr_Ahat', '-v7.3');
disp('all_corr_Ahat saved successfully.');

save(fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/E/whole-brain/all_corr_E.mat'), 'all_corr_E', '-v7.3');
disp('all_corr_E saved successfully.');

%% Concatenate and average correlation results per layer
clc;
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
% layernumber = 'layers5_32c'; 

all_avg_corr_Ahat = cell(4, 1);  
all_avg_corr_E = cell(4,1);      

for lay = 1:length(layername)
    
    % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 4;  % New index for E layers
    else
        e_index = lay;  % Keep the original index for Ahat layers
    end

    % Determine secpath based on layername and load the corresponding correlation data
    if contains(layername{lay}, 'Ahat')
        all_corr_file = fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/Ahat/whole-brain/', 'all_corr_Ahat.mat');
        load(all_corr_file, 'all_corr_Ahat');  % Load the correlation results for Ahat
        all_corr=all_corr_Ahat;
    elseif contains(layername{lay}, 'E')
        all_corr_file = fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/E/whole-brain/', 'all_corr_E.mat');
        load(all_corr_file, 'all_corr_E');  % Load the correlation results for E
        all_corr=all_corr_E;
    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end

    % Pre-allocate for all correlations of the current layer
    n_voxels = size(all_corr{e_index}{1}, 1);  % Assuming the number of voxels is the same for each layer
    all_layer_corr = zeros(n_voxels, n_test_subjects);  % n_voxels x n_test_subjects

    % Loop over all test subjects and store correlations in the pre-allocated matrix
    for test_subj = 1:n_test_subjects
        all_layer_corr(:, test_subj) = all_corr{e_index}{test_subj};  % Each subject's correlations (for Ahat or E)
    end

    % Compute the average correlation across subjects (per voxel)
    avg_corr = mean(all_layer_corr, 2);  % Average across subjects

    if contains(layername{lay}, 'Ahat')
        all_avg_corr_Ahat{e_index} = avg_corr; 
    elseif contains(layername{lay}, 'E')
        all_avg_corr_E{e_index} = avg_corr; 
    end
end
% layer 4 
concat_corr_Ahat = cat(2, all_avg_corr_Ahat{1,1}, all_avg_corr_Ahat{2,1}, all_avg_corr_Ahat{3,1}, all_avg_corr_Ahat{4,1});
concat_corr_E = cat(2, all_avg_corr_E{1,1}, all_avg_corr_E{2,1}, all_avg_corr_E{3,1}, all_avg_corr_E{4,1});

% layer 5
% concat_corr_Ahat = cat(2, all_avg_corr_Ahat{1,1}, all_avg_corr_Ahat{2,1}, all_avg_corr_Ahat{3,1}, all_avg_corr_Ahat{4,1}, all_avg_corr_Ahat{5,1});
% concat_corr_E = cat(2, all_avg_corr_E{1,1}, all_avg_corr_E{2,1}, all_avg_corr_E{3,1}, all_avg_corr_E{4,1}, all_avg_corr_E{5,1});


% Save the all_avg_corr for 'Ahat' and 'E' separately
save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/','layer5_no_layer0','/Ahat/whole-brain/', 'concat_avg_corr_Ahat.mat']), 'concat_corr_Ahat', '-v7.3');
disp('all_avg_corr_Ahat saved successfully.');

save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/','layer5_no_layer0','/E/whole-brain/', 'concat_avg_corr_E.mat']), 'concat_corr_E', '-v7.3');
disp('all_avg_corr_E saved successfully.');

%% save into dtseries
clc;

layernumber = 'layer5_no_layer0'; 

%load example cii structure 
subject_id='subject1';
fmripath =['/combinelab2/03_user/jungmin/02_data/03_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');

% load concat avg corr
% Ahat
Ahat_save_dir=fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/whole-brain/']);
Ahat = load([Ahat_save_dir, 'concat_avg_corr_Ahat.mat']).concat_corr_Ahat;

cii.cdata = Ahat;
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(Ahat_save_dir, 'whole-brain_lambda5_Ahat.dtseries.nii');
ciftisavereset(cii, (save_file_name), 'wb_command');

% E
E_save_dir=fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/whole-brain/']);
E = load([E_save_dir, 'concat_avg_corr_E.mat']).concat_corr_E;

cii.cdata = E;
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(E_save_dir, 'whole-brain_lambda5_E.dtseries.nii');
ciftisavereset(cii, (save_file_name), 'wb_command');
