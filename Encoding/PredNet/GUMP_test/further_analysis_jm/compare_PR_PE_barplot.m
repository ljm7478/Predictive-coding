clc; clear;

% Load diff_all from earlier computation if saved
% e.g., load('/path/to/diff_all.mat');  % if saved before
% Otherwise, re-run or re-import from existing .mat files

% Assuming diff_all is still in workspace:
% diff_all = [n_subjects x n_layers x n_voxels];
basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result_orig_mk_stimulus/';
layer_type = 'layer4';
n_layers = 4;
subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
n_subjects = length(subjects);

% Load visual ROI mask
label_32k = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
vertex = label_32k.cdata;
n_voxels = length(vertex);

visual_voxels = find(ismember(vertex, [1, 181, 3, 183, 4, 184, 16, 196, 7, 187, 19, 199, ...
    23, 203, 2, 182, 17, 197, 20, 200, 21, 201, 46, 226, 142, 322, 145, 325, ...
    146, 326, 152, 332, 155, 335, 31, 211, 95, 275, 122, 302, 121, 301, ...
    119, 299, 126, 306, 127, 307, 132, 133, 134, 135, 136, ...
    312, 313, 314, 315, 316, 143, 323, 139, 140, 141, ...
    319, 320, 321, 150, 151, 330, 331, 153, 154, 156, ...
    333, 334, 336, 160, 340, 163, 343, 230, 50, 229, ...
    49, 339, 159, 202, 22, 337, 157, 352, 172, 357, 177, ...
    131, 311, 15, 195, 300, 120, 317, 318, 137, 138, ...
    13, 19, 158, 5, 193, 338, 185, 6, 186, 18, 198, 298, 118]));

roi_mask = zeros(1, n_voxels);
roi_mask(visual_voxels) = 1;

% Load layerwise diff_all matrix if not already in workspace
fprintf('Loading Prediction - PE differences per subject/layer...\n');
diff_all = zeros(n_subjects, n_layers, n_voxels);
for s = 1:n_subjects
    subject_id = subjects{s};
    pred_corr = load(fullfile(basedir, layer_type, 'E', subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    pe_corr   = load(fullfile(basedir, layer_type, 'Ahat', subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    diff_all(s, :, :) = pred_corr - pe_corr;
end

%% Compute ROI-mean and global-mean difference per subject/layer
mean_roi = zeros(n_subjects, n_layers);
mean_whole = zeros(n_subjects, n_layers);

for l = 1:n_layers
    layer_diff = squeeze(diff_all(:, l, :));  % [subjects x voxels]
    mean_roi(:, l) = mean(layer_diff(:, visual_voxels), 2);  % ROI mean
    mean_whole(:, l) = mean(layer_diff, 2);                  % Whole brain mean
end

%% Plot
figure('Position', [200, 300, 1000, 400]);

subplot(1,2,1);
bar_data = mean(mean_roi, 1);
bar_err = std(mean_roi, 0, 1) / sqrt(n_subjects);
bar(1:n_layers, bar_data); hold on;
errorbar(1:n_layers, bar_data, bar_err, 'k.', 'LineWidth', 1.2);
title('Prediction - PE (Visual ROI)');
xlabel('Layer'); ylabel('Corr diff (r)');
xticks(1:n_layers); xticklabels({'L1','L2','L3','L4'});
ylim padded; grid on;

subplot(1,2,2);
bar_data = mean(mean_whole, 1);
bar_err = std(mean_whole, 0, 1) / sqrt(n_subjects);
bar(1:n_layers, bar_data); hold on;
errorbar(1:n_layers, bar_data, bar_err, 'k.', 'LineWidth', 1.2);
title('Prediction - PE (Whole Brain)');
xlabel('Layer'); ylabel('Corr diff (r)');
xticks(1:n_layers); xticklabels({'L1','L2','L3','L4'});
ylim padded; grid on;

sgtitle('Encoding Difference (Prediction - PE) Across Layers');

