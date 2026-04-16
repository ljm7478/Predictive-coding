
%% GUMP
%% 59k - GUMP 
clc;clear 
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
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
save_file = [saveroot, '/', 'Y', '/', 'fMRI_59k_cortex'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');

%% 10k -GUMP

clc;clear 
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
num_runs = 8;
sub_cii = cell(numel(num_subjects), 1);
existing_subjects = {};  % To store the subjects that are found

for subject_idx = 1:numel(num_subjects)
    subject = num_subjects{subject_idx};
    subject_cii = cell(1, num_runs);
    subject_exists = true;  % Flag to indicate if subject exists
    
    for run = 1:num_runs
        folder_path = sprintf('/zeus-share/local_raid4/user/sunghyoung/01_project/04_encoding_model/02_data/Y/%s/', subject);
        
        files = dir(fullfile(folder_path, sprintf('Y_%s_seg-%d_cortex_10k.mat', subject, run)));
        disp(files.name);

        cii_run = cell(1, numel(files));

        for i = 1:numel(files)
            file_path = fullfile(folder_path, files(i).name);
            cii = load(file_path);
            cii_run{i} = cii.data;
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
save_file = [saveroot, '/', 'Y', '/', 'fMRI_10k_cortex'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');

%% 10k -GUMP -ROI

clc;clear 

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';


num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
num_runs = 8;
sub_cii = cell(numel(num_subjects), 1);
existing_subjects = {};  % To store the subjects that are found

% 10k cii
cii_10k_LR = ciftiopen('/combinelab/03_user/jungmin/04_resampled_cifti/label_file/ResultsRegions_ROI.10k_fs_LR.dlabel.nii', 'wb_command');

% find ROI index 
vertex=cii_10k_LR.cdata;

ROI_condition = ismember(vertex, [1, 2, 3, 4, 5, 10, 11, 13, 15, 16, 17, 22]);

for subject_idx = 1:numel(num_subjects)
    subject = num_subjects{subject_idx};
    subject_cii = cell(1, num_runs);
    subject_exists = true;  % Flag to indicate if subject exists
    
    for run = 1:num_runs
        folder_path = sprintf('/zeus-share/local_raid4/user/sunghyoung/01_project/04_encoding_model/02_data/Y/%s/', subject);
        
        files = dir(fullfile(folder_path, sprintf('Y_%s_seg-%d_cortex_10k.mat', subject, run)));
        disp(files.name);

        cii_run = cell(1, numel(files));

        for i = 1:numel(files)
            file_path = fullfile(folder_path, files(i).name);
            cii = load(file_path);
            cii_run{i} = ROI_condition .* cii.data;
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
save_file = [saveroot, '/', 'Y', '/', 'fMRI_10k_ROI'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');



%% HCP
%% 59k - HCP 
%!!sub725751 is missing for clean ver HCP!!

clc;clear 
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test';

subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
taskIDs = {'tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP'};
sub_cii = cell(numel(subjectIDs), 1);
existing_subjects = {};  % To store the subjects that are found
subcortex_no = 1:59412;

for subject_idx = 1:15   %15 subjects for test
    subject = subjectIDs(subject_idx);
    subject_cii = cell(1, numel(taskIDs));
    subject_exists = true;  % Flag to indicate if subject exists

    disp(['start next sub ', num2str(subject)])

    for task = 1:numel(taskIDs)
        task_idx = taskIDs{task};
        disp(task_idx)
        folder_path = sprintf('/combinelab/02_data/01_HCP/7T_MOVIE_2mm_FIX/%d/MNINonLinear/Results/%s', subject, task_idx);

        files = fullfile(folder_path, sprintf('%s_Atlas_MSMAll_hp2000_clean.dtseries.nii', task_idx));

        cii = ciftiopen(files, 'wb_command');
        cii_59k = cii.cdata(subcortex_no, :);

        subject_cii{task} = cii_59k; 

        % Check if subject data is empty
        if isempty(subject_cii{task})
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
save_file = [saveroot, '/', 'Y', '/', 'fMRI_59k_cortex_hcp_clean_ver'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');


%% 10k -HCP
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test';

subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
taskIDs = {'tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP'};
sub_cii = cell(numel(subjectIDs), 1);
existing_subjects = {};  % To store the subjects that are found
subcortex_no = 1:18722;

for subject_idx = 1:numel(subjectIDs)
    subject = subjectIDs(subject_idx);
    subject_cii = cell(1, numel(taskIDs));
    subject_exists = true;  % Flag to indicate if subject exists

    disp(['start next sub ', num2str(subject)])

    for task = 1:numel(taskIDs)
        task_idx = taskIDs{task};
        disp(task_idx)
        folder_path = sprintf('/combinelab/02_data/01_HCP/7T_MOVIE_2mm/%d/MNINonLinear/Results/%s', subject, task_idx);

        files = fullfile(folder_path, sprintf('%s_Atlas_MSMAll.10k.dtseries.nii', task_idx));

        cii = ciftiopen(files, 'wb_command');
        cii_10k = cii.cdata(subcortex_no, :);

        subject_cii{task} = cii_10k; 

        % Check if subject data is empty
        if isempty(subject_cii{task})
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
save_file = [saveroot, '/', 'Y', '/', 'fMRI_10k_cortex'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');