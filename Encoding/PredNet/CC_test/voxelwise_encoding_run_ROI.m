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
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c';  

lambda = 0.1:0.2:0.9;  % Regularization parameters
nfold = 9;  % Number of folds for cross-validation
clc;
%% Perform ROI level voxelwise encoding for each layer
clc;

% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
% layernumber = 'layers5_32c';  

% Load the label file to define brain regions
label_32k = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
vertex = label_32k.cdata;  % Extract vertex labels

% Define ROI labels (specific cortical areas)
roi_labels = [
    1, 181, 3, 183, 4, 184, 16, 196, 7, 187, 19, 199, 23, 203, 2, 182, ...
    17, 197, 20, 200, 21, 201, 46, 226, 142, 322, 145, 325, 146, 326, ...
    152, 332, 155, 335, 31, 211, 95, 275, 122, 302, 121, 301, 119, 299, ...
    126, 306, 127, 307, 132, 133, 134, 135, 136, 312, 313, 314, 315, ...
    316, 143, 323, 139, 140, 141, 319, 320, 321, 150, 151, 330, 331, ...
    153, 154, 156, 333, 334, 336, 160, 340, 163, 343, 230, 50, 229, 49, ...
    339, 159, 202, 22, 337, 157, 352, 172, 357, 177, 131, 311, 15, 195, ...
    300, 120, 317, 318, 137, 138, 13, 158, 5, 193, 199, 338, 185, 6, ...
    156, 186, 336, 18, 198, 298, 118
];

% Get the indices of the specific voxels in the selected ROIs
specific_voxels = find(ismember(vertex, roi_labels));  % Efficiently find the voxel indices

% Iterate over each layer to process the data
for lay = 1:length(layername)
    disp(['Processing Layer: ', num2str(lay)]);

    % Determine the path based on layer name
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
        fmri_data_subj = train_data(:, :,subj);
        roi_fmri_data = fmri_data_subj(specific_voxels, :);

        disp(['Y_train size: ', num2str(size(Y_train)), '  roi_fmri_data size: ', num2str(size(roi_fmri_data))]);

        % Perform voxelwise encoding 
        [W, Rmat, Lambda] = voxelwise_encoding(Y_train, roi_fmri_data', lambda, nfold);

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
    W_aggregated = W_aggregated / n_train_subjects;  

    % Save the aggregated weights for this layer
    file_name = ['W_aggregated_avg_67sub_', layername{lay}(2:end), '.mat'];
    save(fullfile(save_tmp, 'ROI', file_name), 'W_aggregated', '-v7.3');
end


%% Load the processed testing fMRI responses
savepath = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/fmri';
load(fullfile(savepath, 'testing_fmri.mat'), 'test_data', 'subject_list');

% Check the dimensions of the testing data
[n_voxels, n_timepoints, n_test_subjects] = size(test_data);
disp(['Testing data dimensions: ', num2str(n_voxels), ' voxels, ', num2str(n_timepoints), ' timepoints, ', num2str(n_test_subjects), ' subjects']);

%% Y_test and correlation_all_test
all_corr = cell(length(layername)/2, 1); 

% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
% layernumber = 'layers5_32c';  

% Load the label file to define brain regions
label_32k = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
vertex = label_32k.cdata;  % Extract vertex labels

% Define ROI labels (specific cortical areas)
roi_labels = [
    1, 181, 3, 183, 4, 184, 16, 196, 7, 187, 19, 199, 23, 203, 2, 182, ...
    17, 197, 20, 200, 21, 201, 46, 226, 142, 322, 145, 325, 146, 326, ...
    152, 332, 155, 335, 31, 211, 95, 275, 122, 302, 121, 301, 119, 299, ...
    126, 306, 127, 307, 132, 133, 134, 135, 136, 312, 313, 314, 315, ...
    316, 143, 323, 139, 140, 141, 319, 320, 321, 150, 151, 330, 331, ...
    153, 154, 156, 333, 334, 336, 160, 340, 163, 343, 230, 50, 229, 49, ...
    339, 159, 202, 22, 337, 157, 352, 172, 357, 177, 131, 311, 15, 195, ...
    300, 120, 317, 318, 137, 138, 13, 158, 5, 193, 199, 338, 185, 6, ...
    156, 186, 336, 18, 198, 298, 118
];

% Get the indices of the specific voxels in the selected ROIs
specific_voxels = find(ismember(vertex, roi_labels));  % Efficiently find the voxel indices

all_corr_Ahat = cell(4, 1); 
all_corr_E = cell(4, 1);     

for lay = 1:4 %length(layername)
    
    % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 4;  % New index for E layers
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
        W_filename = [saveroot, 'ROI/', 'W_aggregated_avg_67sub_', layername{lay}(2:end), '.mat'];
        load(W_filename, 'W_aggregated');  % W_aggregated is (components, voxels)
        
        % Ensure Y_train matches the format expected by voxelwise_encoding (timepoints x components)
        [n_timepoints, n_components] = size(Y_test);
        if n_timepoints > 193
            Y_test = Y_test(1:193, :);  % Trim to 193 timepoints
            disp('Y_test trimmed to match train_data timepoints');
        elseif n_timepoints < 193
            disp('Warning: train_data has more timepoints than Y_test');
        end
        
        % Ensure Y_test and W_aggregated dimensions match
        if size(Y_test, 2) ~= size(W_aggregated, 1)
            error('Dimension mismatch: Y_test and W_aggregated must have matching second and first dimensions');
        end
        
        % Compute predicted BOLD using the feature maps and the aggregated weights matrix
        pred_fmri = Y_test * W_aggregated;  % Predicted BOLD: (timepoints, voxels)
        
        %load measured BOLD
        fmri_data_subj = test_data(:,:,test_subj);
        roi_fmri_data = fmri_data_subj(specific_voxels, :);
        roi_fmri_data = roi_fmri_data';

        % Calculate correlation scores for each voxel
        correlation_scores = zeros(1, size(pred_fmri, 2));  % Preallocate for correlations
        for v = 1:size(pred_fmri, 2)  % Loop over voxels
            correlation_scores(v) = corr(pred_fmri(:, v), roi_fmri_data(:, v));  % Compute correlation for each voxel
        end

         if contains(layername{lay}, 'Ahat')
            updated_test_data = -1 * ones(1, size(test_data, 1));
            updated_test_data(specific_voxels) = correlation_scores';
            all_corr_Ahat{e_index}{test_subj} = updated_test_data; 
        elseif contains(layername{lay}, 'E')
            updated_test_data = -1 * ones(1, size(test_data, 1));
            updated_test_data(specific_voxels) = correlation_scores';
            all_corr_E{e_index}{test_subj} = updated_test_data; 
        end
    end
end

% Save the all_corr for 'Ahat' and 'E' separately
save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/','layer5_no_layer0','/Ahat/ROI/', 'all_corr_Ahat.mat']), 'all_corr_Ahat', '-v7.3');
disp('all_corr_Ahat saved successfully.');

save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/','layer5_no_layer0','/E/ROI/', 'all_corr_E.mat']), 'all_corr_E', '-v7.3');
disp('all_corr_E saved successfully.');

%% Concatenate and average correlation results per layer
clc;
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layer5_no_layer0'; 

all_avg_corr_Ahat = cell(5, 1);  
all_avg_corr_E = cell(5,1);      

for lay = 1:4 %length(layername)
    
    % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 5;  % New index for E layers
    else
        e_index = lay;  % Keep the original index for Ahat layers
    end

    % Determine secpath based on layername and load the corresponding correlation data
    if contains(layername{lay}, 'Ahat')
        all_corr_file = fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/Ahat/ROI/', 'all_corr_Ahat.mat');
        load(all_corr_file, 'all_corr_Ahat');  % Load the correlation results for Ahat
        all_corr=all_corr_Ahat;
    elseif contains(layername{lay}, 'E')
        all_corr_file = fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layer5_no_layer0/E/ROI/', 'all_corr_E.mat');
        load(all_corr_file, 'all_corr_E');  % Load the correlation results for E
        all_corr=all_corr_E;
    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end

    % Pre-allocate for all correlations of the current layer
    n_voxels = size(all_corr{e_index}{1}, 2);  % Assuming the number of voxels is the same for each layer
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
% concat_corr_Ahat = cat(2, all_avg_corr_Ahat{1,1}, all_avg_corr_Ahat{2,1}, all_avg_corr_Ahat{3,1}, all_avg_corr_Ahat{4,1});
% concat_corr_E = cat(2, all_avg_corr_E{1,1}, all_avg_corr_E{2,1}, all_avg_corr_E{3,1}, all_avg_corr_E{4,1});

% layer 5
concat_corr_Ahat = cat(2, all_avg_corr_Ahat{1,1}, all_avg_corr_Ahat{2,1}, all_avg_corr_Ahat{3,1}, all_avg_corr_Ahat{4,1}, all_avg_corr_Ahat{5,1});
concat_corr_E = cat(2, all_avg_corr_E{1,1}, all_avg_corr_E{2,1}, all_avg_corr_E{3,1}, all_avg_corr_E{4,1}, all_avg_corr_E{5,1});

% Save the all_avg_corr for 'Ahat' and 'E' separately
save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/ROI/', 'concat_avg_corr_Ahat.mat']), 'concat_corr_Ahat', '-v7.3');
disp('all_avg_corr_Ahat saved successfully.');

save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/ROI/', 'concat_avg_corr_E.mat']), 'concat_corr_E', '-v7.3');
disp('all_avg_corr_E saved successfully.');


%% save into dtseries
clc;clear

layernumber = 'layer5_no_layer0'; 

%load example cii structure 
subject_id='subject1';
fmripath =['/combinelab2/03_user/jungmin/02_data/03_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');

% load concat avg corr
% Ahat
Ahat_save_dir=fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/ROI/']);
Ahat = load([Ahat_save_dir, 'concat_avg_corr_Ahat.mat']).concat_corr_Ahat;

cii.cdata = Ahat;
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(Ahat_save_dir, 'ROI_lambda5_Ahat.dtseries.nii');
ciftisavereset(cii, (save_file_name), 'wb_command');

% E
E_save_dir=fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/ROI/']);
E = load([E_save_dir, 'concat_avg_corr_E.mat']).concat_corr_E;

cii.cdata = E;
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(E_save_dir, 'ROI_lambda5_E.dtseries.nii');
ciftisavereset(cii, (save_file_name), 'wb_command');




%% save into dtseries difference map
clc; clear;

layernumber = 'layers5_32c'; 

% Load example CIFTI structure
subject_id = 'subject1';
fmripath = ['/combinelab2/03_user/jungmin/02_data/03_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % Path to the CIFTI files
seg = 1;
filename = ['seg', num2str(seg), '/cifti/seg', num2str(seg), '_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename], 'wb_command');

% Load `concat_avg_corr_Ahat` (Predictions)
Ahat_save_dir = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/Ahat/ROI/']);
Ahat = load([Ahat_save_dir, 'concat_avg_corr_Ahat.mat']).concat_corr_Ahat;

% Load `concat_avg_corr_E` (Errors)
E_save_dir = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/E/ROI/']);
E = load([E_save_dir, 'concat_avg_corr_E.mat']).concat_corr_E;

% Check dimensions
disp(['Ahat dimensions: ', num2str(size(Ahat))]);
disp(['E dimensions: ', num2str(size(E))]);

% Ensure dimensions match
if size(Ahat) ~= size(E)
    error('Mismatch in dimensions between Ahat and E.');
end

% Compute Difference Map: Ahat (Prediction) - E (Error)
difference_map = Ahat - E;

% Update the CIFTI structure with the concatenated difference map
cii.cdata = difference_map;

% Update dimensions for the new data
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];  % Adjust the models

% Save the concatenated difference map as a new CIFTI file
difference_save_dir = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/Difference/ROI/']);
if ~exist(difference_save_dir, 'dir')
    mkdir(difference_save_dir);  % Create directory if it doesn't exist
end
save_file_name = fullfile(difference_save_dir, 'ROI_lambda5_Difference_AllLayers.dtseries.nii');
ciftisavereset(cii, save_file_name, 'wb_command');

disp(['Concatenated difference map saved to: ', save_file_name]);


disp('All difference maps saved successfully.');



%% 
clc;clear

layernumber = 'layers5_32c'; 

% Load example CIFTI structure
subject_id = 'subject1';
fmripath = ['/combinelab2/03_user/jungmin/02_data/03_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % Path to the CIFTI files
seg = 1;
filename = ['seg', num2str(seg), '/cifti/seg', num2str(seg), '_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename], 'wb_command');

% Load `concat_avg_corr_Ahat` (Predictions)
Ahat_save_dir = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/Ahat/ROI/']);
Ahat = load([Ahat_save_dir, 'concat_avg_corr_Ahat.mat']).concat_corr_Ahat;

% Load `concat_avg_corr_E` (Errors)
E_save_dir = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/E/ROI/']);
E = load([E_save_dir, 'concat_avg_corr_E.mat']).concat_corr_E;

% Initialize the difference map with the same size as Ahat/E (59412 x 5)
difference_map = ones(size(Ahat));  % Start with an array of ones (prediction > error)

% Loop over layers to compute per-layer difference map
for layer = 1:size(Ahat, 2)
    % For each voxel, compare Ahat and E
    difference_map(:, layer) = 1 * (Ahat(:, layer) > E(:, layer)) - 1 * (Ahat(:, layer) < E(:, layer));
end


% Update the CIFTI structure with the concatenated difference map
cii.cdata = difference_map;

% Update dimensions for the new data
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];  % Adjust the models

% Save the concatenated difference map as a new CIFTI file
difference_save_dir = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/Difference/ROI/']);
if ~exist(difference_save_dir, 'dir')
    mkdir(difference_save_dir);  % Create directory if it doesn't exist
end
save_file_name = fullfile(difference_save_dir, 'ROI_lambda5_Difference_AllLayers_1_or_0.dtseries.nii');
ciftisavereset(cii, save_file_name, 'wb_command');

disp(['Concatenated difference map saved to: ', save_file_name]);


disp('All difference maps saved successfully.');