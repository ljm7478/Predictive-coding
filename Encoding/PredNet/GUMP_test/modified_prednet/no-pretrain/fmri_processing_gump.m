%% This code is for processing the BOLD fMRI response to natural movies
clc;clear

%% add paths for fMRI processing 
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01 _software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;
%% Put all training segments together
clc;
Nv = 59412; % number of vertices on the cortical surface
num_volumes = [451, 441, 438, 488, 462, 439, 542, 338]; % number of volumes for each run

fmripath_template = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-%d';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri';

%% Processing Data for Each Subject
clc;
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', ...
            'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', ...
            'sub-18', 'sub-19', 'sub-20'};

% Initialize storage for concatenated fMRI data
fmri_data_concat = []; 

for subj = 2:length(subjects)
    subject = subjects{subj};
    % Initialize storage for fMRI data (use single precision to save memory)
    fmri_data = struct();
    
    % Loop through the 8 runs
    for run = 1:8
        disp(['Processing subject: ', subject, ', run: ', num2str(run)]);
        
        % Generate the file path
        fmripath = sprintf(fmripath_template, subject, subject, subject, run);
        filename = sprintf('/ses-movie_task-movie_run-%d_Atlas_s3.dtseries.nii', run);
        fmri_file = [fmripath, filename];
        
        % Open CIFTI file
        cii = ciftiopen(fmri_file, 'wb_command');
        
        % Check the number of volumes for the current run
        Nt = num_volumes(run); % Get number of volumes for this run
        
        % Mask out the vertices in subcortical areas
        lbl = zeros(size(cii.cdata, 1), 1);
        lbl(1:Nv) = 1;
        lbl = (lbl == 1);
        data = double(cii.cdata(lbl, 1:Nt));  % Use all volumes
        
        % Remove the 4th order polynomial trend for each voxel
        for i = 1:size(data, 1)
            data(i,:) = amri_sig_detrend(data(i,:), 4);
        end
        
        % Standardization: remove the mean and divide by standard deviation
        data = bsxfun(@minus, data, mean(data, 2));
        data = bsxfun(@rdivide, data, std(data, [], 2));

        % Store the data for this run
        fmri_data.(['run', num2str(run)]) = data;

        % Concatenate along the temporal dimension
        fmri_data_concat = [fmri_data_concat, data];  % Concatenate temporally
    end

    % Save the processed data for this subject
    sub_fmri_dir = [saveroot, '/', subject,'/'];
    if ~exist(sub_fmri_dir, 'dir') 
        mkdir(sub_fmri_dir)
    end
    save([sub_fmri_dir, subject  '_per_run_fmri.mat'], 'fmri_data', '-v7.3');
    save([sub_fmri_dir, subject  '_concat.mat'], 'fmri_data_concat', '-v7.3');


    % Training & testing sets
    for test_run = 1:8
        % Define training data excluding the test run
        train_runs = setdiff(1:8, test_run);
        fmri_train = [];  % Initialize the training data

        % Concatenate training runs (excluding the test run)
        for train_run = train_runs
            fmri_train = [fmri_train, fmri_data_concat(:, sum(num_volumes(1:train_run-1)) + 1 : sum(num_volumes(1:train_run)))];
        end

        % Save training data
        save([sub_fmri_dir, subject '_fmri_train_excluding_run', num2str(test_run), '.mat'], 'fmri_train', '-v7.3');
    end
end



