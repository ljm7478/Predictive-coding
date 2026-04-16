%% Predictive Coding Analysis Pipeline

clc; clear;
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));

%% Paths
basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result_orig_mk_stimulus/';
layer_type = 'layer4';
pred_dir = fullfile(basedir, layer_type, 'E');
pe_dir   = fullfile(basedir, layer_type, 'Ahat');

subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
n_subjects = length(subjects);
n_layers = 4;

% Glasser label
label_32k = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
region_labels = label_32k.cdata;
region_names = arrayfun(@(x) sprintf('R%d',x), unique(region_labels(region_labels>0)), 'UniformOutput', false);

% Scenes CSV
scenes_csv_file = '/mnt/data/scenes.csv'; % or your local path

%% Aggregate across subjects
pred_corr_all = zeros(n_subjects, n_layers, length(region_labels));
pe_corr_all   = zeros(n_subjects, n_layers, length(region_labels));

for s = 1:n_subjects
    subject_id = subjects{s};
    fprintf('Subject %s...\n', subject_id);
    
    % Load Prediction
    file_pred = fullfile(pred_dir, subject_id, [subject_id, '_avgcorr_lambda5.mat']);
    load(file_pred, 'concat_corr');
    pred_corr_all(s,:,:) = concat_corr;
    
    % Load PE
    file_pe = fullfile(pe_dir, subject_id, [subject_id, '_avgcorr_lambda5.mat']);
    load(file_pe, 'concat_corr');
    pe_corr_all(s,:,:) = concat_corr;
end

%% 1 Layer Depth Profile
mean_pred_layer = squeeze(mean(mean(pred_corr_all,3),1)); % [n_layers x 1]
mean_pe_layer   = squeeze(mean(mean(pe_corr_all,3),1));   % [n_layers x 1]

figure;
plot(1:n_layers, mean_pred_layer, '-o', 'LineWidth',2);
hold on;
plot(1:n_layers, mean_pe_layer, '-o', 'LineWidth',2);
xlabel('Layer');
ylabel('Mean Correlation');
legend({'Prediction','Prediction Error'});
title('Layer Depth Profile: P vs PE');
grid on;

%% 2 Layer x Region Crossmatrix
avg_pred_corr = squeeze(mean(pred_corr_all,1)); % [n_layers x n_voxels]
avg_pe_corr   = squeeze(mean(pe_corr_all,1));   % [n_layers x n_voxels]

figure;
subplot(1,2,1);
layer_region_crossmatrix(avg_pred_corr, region_labels, region_names);
title('Layer x Region: Prediction');

subplot(1,2,2);
layer_region_crossmatrix(avg_pe_corr, region_labels, region_names);
title('Layer x Region: PE');

%% 3 PE Event Triggered Avg
% Average PE signal over cortex
pe_mean_voxelwise = squeeze(mean(pe_corr_all,1)); % [n_layers x n_voxels]

% For now use Layer 4 PE signal mean over voxels:
pe_signal = mean(pe_mean_voxelwise(end,:),2)'; % [T] assuming PE signal over time

% But your pe_corr_all is voxelwise avg_corr ?? you probably need to load the *per frame PE signal* ?? adjust accordingly.

event_types = {'scene1', 'scene2', 'scene3', 'scene4'}; % fill in with your actual scene names!
TR_sec = 2.47;
[avg_traces, time_vector] = pe_event_triggered_avg(pe_signal, scenes_csv_file, event_types, 10, 20, TR_sec);

%% 4 PE/P Ratio Map
ratio_map = pe_p_ratio_map(avg_pred_corr, avg_pe_corr);

% Save ratio_map to CIFTI
cii_template = ciftiopen('/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii', 'wb_command');
cii_template.cdata = ratio_map';
cii_template.diminfo{1,1}.length = size(cii_template.cdata,1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata,2);
cii_template.diminfo{1,1}.models(3:end) = [];
save_file_name = fullfile(basedir, layer_type, 'Prediction_vs_PE_perlayer', 'PE_P_ratio_map.dtseries.nii');
ciftisavereset(cii_template, save_file_name, 'wb_command');

fprintf('DONE.\n');

