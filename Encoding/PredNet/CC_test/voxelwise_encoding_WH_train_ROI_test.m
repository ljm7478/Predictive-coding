clc; clear;

%% Add paths for fMRI processing
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

%% Load the processed testing fMRI responses
savepath = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/fmri';
load(fullfile(savepath, 'testing_fmri.mat'), 'test_data', 'subject_list');

% Check the dimensions of the testing data
[n_voxels, n_timepoints, n_test_subjects] = size(test_data);
disp(['Testing data dimensions: ', num2str(n_voxels), ' voxels, ', num2str(n_timepoints), ' timepoints, ', num2str(n_test_subjects), ' subjects']);

%% Y_test and correlation_all_test
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c';  

all_corr = cell(length(layername), 1); 

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

all_corr_Ahat = cell(5, 1); 
all_corr_E = cell(5, 1);     

for lay = 1:length(layername)
    
     % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 5;  % New index for E layers
    else
        e_index = lay;  % Keep the original index for Ahat layers
    end
    
    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];
    elseif contains(layername{lay}, 'E')
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];
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
        W_filename = [saveroot, 'whole-brain/', 'W_aggregated_avg_67sub_', layername{lay}(2:end), '.mat'];
        
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
        
        %load measured BOLD
        fmri_data_subj = test_data(:,:,test_subj);
        fmri_data_subj = fmri_data_subj';

        % Iterate only over specific voxels
        correlation_scores = -1 * ones(1, size(test_data, 1));
        for voxel = 1:numel(specific_voxels)
            voxel_index = specific_voxels(voxel);
            % Calculate correlation only for the specified voxels
            correlation_scores(voxel_index) = corr(pred_fmri(:, voxel_index), fmri_data_subj(:, voxel_index));
        end

        if contains(layername{lay}, 'Ahat')
            all_corr_Ahat{e_index}{test_subj} = correlation_scores;
        elseif contains(layername{lay}, 'E')
            all_corr_E{e_index}{test_subj} = correlation_scores;
        end
    end
end

% Save the all_corr for 'Ahat' and 'E' separately
save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/ROI/', 'WH_train_all_corr_Ahat.mat']), 'all_corr_Ahat', '-v7.3');
disp('all_corr_Ahat saved successfully.');

save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/ROI/', 'WH_train_all_corr_E.mat']), 'all_corr_E', '-v7.3');
disp('all_corr_E saved successfully.');

%% Concatenate and average correlation results per layer
clc;
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c'; 

all_avg_corr_Ahat = cell(5, 1);  
all_avg_corr_E = cell(5,1);      

for lay = 1:length(layername)
    
    % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 5;  % New index for E layers
    else
        e_index = lay;  % Keep the original index for Ahat layers
    end

    % Determine secpath based on layername and load the corresponding correlation data
    if contains(layername{lay}, 'Ahat')
        all_corr_file = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber,'/Ahat/ROI/', 'WH_train_all_corr_Ahat.mat']);
        load(all_corr_file, 'all_corr_Ahat');  % Load the correlation results for Ahat
        all_corr=all_corr_Ahat;
    elseif contains(layername{lay}, 'E')
        all_corr_file = fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/', layernumber, '/E/ROI/', 'WH_train_all_corr_E.mat']);
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
save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/ROI/', 'WH_train_concat_avg_corr_Ahat.mat']), 'concat_corr_Ahat', '-v7.3');
disp('all_avg_corr_Ahat saved successfully.');

save(fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/ROI/', 'WH_train_concat_avg_corr_E.mat']), 'concat_corr_E', '-v7.3');
disp('all_avg_corr_E saved successfully.');

%% save into dtseries
clc;clear

layernumber = 'layers5_32c'; 

%load example cii structure 
subject_id='subject1';
fmripath =['/combinelab2/03_user/jungmin/02_data/03_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/']; % path to the cifti files
seg = 1;
filename = ['seg',num2str(seg),'/cifti/seg', num2str(seg),'_1_Atlas.dtseries.nii'];
cii = ciftiopen([fmripath, filename],'wb_command');

% load concat avg corr
% Ahat
Ahat_save_dir=fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/ROI/']);
Ahat = load([Ahat_save_dir, 'WH_train_concat_avg_corr_Ahat.mat']).concat_corr_Ahat;


cii.cdata = Ahat;
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(Ahat_save_dir, 'WH_train_ROI_lambda5_Ahat.dtseries.nii');
ciftisavereset(cii, (save_file_name), 'wb_command');

% E
E_save_dir=fullfile(['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/ROI/']);
E = load([E_save_dir, 'WH_train_concat_avg_corr_E.mat']).concat_corr_E;

cii.cdata = E;
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(E_save_dir, 'WH_train_ROI_lambda5_E.dtseries.nii');
ciftisavereset(cii, (save_file_name), 'wb_command');
