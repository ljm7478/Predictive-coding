clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;
%%
% Set the base directory
base_dir = '/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/02_fmri/Preprocessed/241121_noT1/output/ciftify/sub-01/MNINonLinear/Results/';

% List all tasks
task_folders = {
    'task-CirclePred_run-01',
    'task-CircleUnpred_run-02',
    'task-GarborFreqPred_run-05',
    'task-GarborFreqUnpred_run-06',
    'task-GarborOrientationPred_run-03',
    'task-GarborOrientationUnpred_run-04',
    'task-GarborSCPred_run-07',
    'task-GarborSCUnpred_run-08',
    'task-MoviePred_run-09',
    'task-MovieUnpred_run-10'
};

% Loop through each task folder
for i = 1:length(task_folders)
    task_folder = task_folders{i};
    dtseries_file = fullfile(base_dir, task_folder, [task_folder '_Atlas_s3.dtseries.nii']);
    
    if exist(dtseries_file, 'file')
        try
            % Open CIFTI file using ciftiopen
            cifti_data = ciftiopen(dtseries_file,  'wb_command');
            
            % Get data size
            data_size = size(cifti_data.cdata);
            fprintf('Task: %s\n', task_folder);
            fprintf('Data dimensions: %d timepoints x %d brain regions\n', data_size(1), data_size(2));
        catch ME
            fprintf('Error opening file: %s\n', dtseries_file);
            fprintf('Error message: %s\n', ME.message);
        end
    else
        fprintf('File not found: %s\n', dtseries_file);
    end
end