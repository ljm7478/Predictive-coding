%% Processing CAM-CAN Movie Data (fMRI)
clc; clear;

%% Add paths for fMRI processing 
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

%% Base path and subject ID file
fmripath = '/combinelab/02_data/12_CAM-CAN/yb_jm_preproc/cc700/mri/pipeline/release004/BIDS_20190411/anat/';
subject_list_file = '/combinelab/02_data/12_CAM-CAN/yb_jm_preproc/cc700/mri/pipeline/release004/BIDS_20190411/anat/sublist.txt';
savepath = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/fmri';

% Try to open the subject list file
fileID = fopen(subject_list_file, 'r');
if fileID == -1
    error('Failed to open subject list file: %s. Please check file encoding or permissions.', subject_list_file);
end

% Initialize the subject list array
subject_list = {};  % Initialize as cell array to store subject IDs as strings

% Read each line in the file
tline = fgetl(fileID);
line_num = 1;

while ischar(tline)
    disp(['Line ', num2str(line_num), ': ', tline]);  % Debugging
    tline = strtrim(tline);
    
    if isempty(tline) || ~startsWith(tline, 'sub-')
        tline = fgetl(fileID);
        line_num = line_num + 1;
        continue;
    end

    subject_list{end+1} = tline;
    tline = fgetl(fileID);
    line_num = line_num + 1;
end

fclose(fileID);

% Check if subject_list is empty
if isempty(subject_list)
    error('No valid subject IDs found in the file: %s', subject_list_file);
end

% Debugging: Check the first few subjects in the list
disp('First few subject IDs:');
disp(subject_list(1:min(10, end)));

% Shuffle subject list to ensure random train/test split
subject_list = subject_list(randperm(length(subject_list)));  % Shuffle the list

% Select 100 subjects for processing
subject_list = subject_list(1:100); % Adjust this as needed

train_pct = 0.8; % Percentage for training data
test_pct = 1 - train_pct; % Percentage for testing data

% Initialize variables for training and testing data
train_data = []; % Empty array for training data
test_data = [];  % Empty array for testing data

train_idx = 1;  % Initialize index for storing training data
test_idx = 1;   % Initialize index for storing testing data

%% Missing subjects log file
missing_subjects_file = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/missing_fMRI_subjects.txt';

% Open the missing_subjects.txt file for appending
missing_subjects_fid = fopen(missing_subjects_file, 'a'); 
if missing_subjects_fid == -1
    error('Failed to open missing subjects file: %s', missing_subjects_file);
end

% Process each subject (split into train/test)
for sub_idx = 1:length(subject_list)
    subject_id = subject_list{sub_idx};
    disp(['Processing subject: ', subject_id]);
    
    % Load the CIFTI data for this subject
    filename = sprintf('%s/MNINonLinear/Results/epi_movie/epi_movie_Atlas.dtseries.nii', subject_id);
    full_filename = fullfile(fmripath, filename);
    
     % Check if the file exists
    if ~isfile(full_filename)
        warning('File does not exist for subject: %s\nFile: %s', subject_id, full_filename);
        
        % Log the missing subject ID to the missing_subjects.txt file
        fprintf(missing_subjects_fid, '%s\n', subject_id);
        continue;
    end

    try
        % Try to open the CIFTI file
        cii = ciftiopen(full_filename, 'wb_command');
    catch ME
        warning('Failed to open CIFTI file for subject: %s\nError: %s', subject_id, ME.message);
        
        % Log the missing subject ID to the missing_subjects.txt file
        fprintf(missing_subjects_fid, '%s\n', subject_id);
        continue;
    end

    disp(['Successfully loaded data for subject: ', subject_id]);

    Nv = 59412;  % Number of cortical vertices (you can adjust this if needed)
    Nt = size(cii.cdata, 2); % Number of timepoints (volumes)
    disp(['Subject ', num2str(subject_id), ': Nv = ', num2str(Nv), ', Nt = ', num2str(Nt)]);

    % Mask out the vertices in subcortical areas
    lbl = zeros(size(cii.cdata, 1), 1);  % Initialize mask
    lbl(1:Nv) = 1;  % Mark the first 59412 vertices (cortex)
    lbl = (lbl == 1);  % Binary mask for cortical regions
    data = double(cii.cdata(lbl,:));  % Extract cortical data

    % Detrend the data
    for i = 1:size(data, 1)
        data(i,:) = amri_sig_detrend(data(i,:), 4);  % Detrend the signal (make sure this function is available)
    end

    % Standardize the data
    data = bsxfun(@minus, data, mean(data, 2));
    data = bsxfun(@rdivide, data, std(data, [], 2));

    % Store the data in train/test matrices
    if sub_idx <= floor(length(subject_list) * train_pct)
        train_data(:,:,train_idx) = data;
        train_idx = train_idx + 1;
    else
        test_data(:,:,test_idx) = data;
        test_idx = test_idx + 1;
    end
end

fclose(missing_subjects_fid);

% Save the processed training and testing fMRI data
% save(fullfile(savepath, 'training_fmri.mat'), 'train_data', 'subject_list', '-v7.3');
% save(fullfile(savepath, 'testing_fmri.mat'), 'test_data', 'subject_list', '-v7.3');
