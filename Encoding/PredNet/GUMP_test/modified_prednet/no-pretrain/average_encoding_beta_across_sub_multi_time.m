%%
clc; clear;

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Subject list
num_subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06', ...
                'sub-09','sub-10','sub-14','sub-15','sub-16','sub-17', ...
                'sub-18','sub-19','sub-20'};

% Feature type ('E' or 'Ahat')
type = 'E'; %or 'E';  

% Layer names
% layername = {'/E0'; '/E1'; '/E2'; '/E3'};  % or Ahat0~Ahat3
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};

layername = {'/E0'; '/E1'; '/E2'; '/E3'; '/E4'; '/E5'; '/E6'};  % or Ahat0~Ahat3
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6'};

% layername = {'/C0'; '/C1'; '/C2'; '/C3'}; 


% Determine folder name
if length(layername) == 4
    layer_name = 'layer4';
else
    layer_name = 'layer7';
end

% Data root
dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/multi_time_result/1second_25frame/', ...
             layer_name, '/', type, '/'];

% Template CIFTI file
template_path = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
template_file = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
cii = ciftiopen([template_path, template_file],'wb_command');

% Initialize storage later (after loading first W)
layer_wise_group_beta = [];

% Loop over each layer
for lay = 1:length(layername)
    current_layer = layername{lay}(2:end);  % Remove leading '/'
    disp(['Processing layer: ', current_layer]);

    for sub = 1:length(num_subjects)
        sid = num_subjects{sub};
        fname = fullfile(dataroot, sid, ['W_lambda5_' current_layer '.mat']);
        load(fname, 'W');  % W: [feature x voxel]

        % Take absolute value of beta weights and average across feature axis
        % So we get: [voxel x 1]
        beta_mean = mean(W, 1); % beta_mean = mean(abs(W), 2);

        % Initialize group matrix
        if sub == 1
            all_beta = zeros(length(num_subjects), length(beta_mean));
        end
        all_beta(sub, :) = beta_mean;
    end

    % Group average
    group_beta_map = mean(all_beta, 1);  % [1 x voxel]
    
    % Stack into layer-wise matrix: [layer x voxel]
    layer_wise_group_beta(lay, :) = group_beta_map;
end

%layer-wise group map
% Save as CIFTI (should be voxel x length(layername)
new_cii = cii;
new_cii.cdata = layer_wise_group_beta';
new_cii.diminfo{1,1}.length = size(new_cii.cdata, 1);
new_cii.diminfo{1,2}.length = size(new_cii.cdata, 2);
new_cii.diminfo{1,1}.models(3:end) = [];

outfile = fullfile(dataroot,['group_beta_' type '_all_layers.dtseries.nii']);
ciftisavereset(new_cii, outfile, 'wb_command');
fprintf('Saved: %s\n', outfile);