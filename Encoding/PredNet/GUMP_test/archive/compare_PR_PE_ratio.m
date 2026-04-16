clc; clear;

% Add paths
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Set paths
basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result_orig_mk_stimulus/';
% basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/';

layer_type = 'layer4';  % or 'layer7'
pred_dir = fullfile(basedir, layer_type, 'E');
pe_dir   = fullfile(basedir, layer_type, 'Ahat');

subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
n_subjects = length(subjects);
n_layers = 7;

% Load one subject to get voxel count
sample = load(fullfile(pred_dir, subjects{1}, [subjects{1} '_avgcorr_lambda5.mat']));
n_voxels = size(sample.concat_corr, 2);

% Initialize: [subjects x layers x voxels]
diff_net_all = zeros(n_subjects, n_layers, n_voxels);
diff_ratio_all = zeros(n_subjects, n_layers, n_voxels);

% Load difference (Prediction - PE) for all subjects/layers
for s = 1:n_subjects
    subject_id = subjects{s};
    fprintf('Loading %s...\n', subject_id);
    pred_corr = load(fullfile(pred_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    pe_corr   = load(fullfile(pe_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    
    diff_net_all(s, :, :) = pred_corr - pe_corr;
    diff_ratio_all(s, :, :) = log((pred_corr + eps) ./ (pe_corr + eps));
end

% Load Glasser visual ROI mask
% label_32k = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
% vertex = label_32k.cdata;
% visual_voxels = find(ismember(vertex, [1, 181, 3, 183, 4, 184, 16, 196, 7, 187, 19, 199, ...
%     23, 203, 2, 182, 17, 197, 20, 200, 21, 201, 46, 226, 142, 322, 145, 325, ...
%     146, 326, 152, 332, 155, 335, 31, 211, 95, 275, 122, 302, 121, 301, ...
%     119, 299, 126, 306, 127, 307, 132, 133, 134, 135, 136, ...
%     312, 313, 314, 315, 316, 143, 323, 139, 140, 141, ...
%     319, 320, 321, 150, 151, 330, 331, 153, 154, 156, ...
%     333, 334, 336, 160, 340, 163, 343, 230, 50, 229, ...
%     49, 339, 159, 202, 22, 337, 157, 352, 172, 357, 177, ...
%     131, 311, 15, 195, 300, 120, 317, 318, 137, 138, ...
%     13, 19, 158, 5, 193, 338, 185, 6, 186, 18, 198, 298, 118]));
% 
% roi_mask = zeros(1, n_voxels);
% roi_mask(visual_voxels) = 1;

% Load template for saving
cii_template = ciftiopen('/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii', 'wb_command');

% Output directory
outdir = fullfile(basedir, layer_type, 'Prediction_vs_PE_perlayer');
if ~exist(outdir, 'dir'); mkdir(outdir); end

avg_diff_concat = zeros(n_layers, n_voxels);
t_map_concat = zeros(n_layers, n_voxels);
sig_map_concat = zeros(n_layers, n_voxels);

% Process each layer
for l = 1:n_layers
    layer_str = sprintf('Layer%d', l);
    fprintf('\n--- Processing %s ---\n', layer_str);

    net_diff = squeeze(diff_all(:, l, :));  % [subjects x voxels]
    ratio_diff = squeeze(diff_ratio_all(:, l, :));

    % %% (1) Visual ROI analysis
    % roi_diff = this_diff(:, visual_voxels);           % [subjects x ROI voxels]
    % roi_mean_diff = mean(roi_diff, 2);                % [subjects x 1]
    % [~, p_roi] = ttest(roi_mean_diff);
    % fprintf('Visual ROI paired t-test p = %.4f\n', p_roi);
    % 
    % % Permutation
    % n_perm = 5000;
    % obs_mean = mean(roi_mean_diff);
    % perm_vals = zeros(n_perm,1);
    % for i = 1:n_perm
    %     flip = randi([0 1], size(roi_mean_diff));
    %     flipped = roi_mean_diff .* (2*flip - 1);
    %     perm_vals(i) = mean(flipped);
    % end
    % p_perm = mean(abs(perm_vals) >= abs(obs_mean));
    % fprintf('Visual ROI permutation p = %.4f\n', p_perm);

    %% (2) Net Whole-brain stats
    [~, p_map, ~, stats] = ttest(net_diff);
    t_map = stats.tstat;
    avg_net_diff = mean(net_diff, 1);  % across subjects

    % FDR correction
    p_fdr = mafdr(p_map', 'BHFDR', true);  % Use one output only
    sig_mask_fdr = p_fdr < 0.05;
    sig_map = avg_diff(:) .* sig_mask_fdr(:);  

    avg_diff_concat(l, :) = avg_diff;
    t_map_concat(l, :) = t_map;
    sig_map_concat(l, :) = sig_map;


    %% (2) RAtio Whole-brain stats
    [~, p_map, ~, stats] = ttest(ratio_diff);
    t_map = stats.tstat;
    avg_ratio_diff = mean(ratio_diff, 1);  % across subjects

    % FDR correction
    p_fdr = mafdr(p_map', 'BHFDR', true);  % Use one output only
    sig_mask_fdr = p_fdr < 0.05;
    sig_map_ratio = avg_diff(:) .* sig_mask_fdr(:);  

    avg_ratio_diff(l, :) = avg_diff;
    t_map_concat(l, :) = t_map;
    sig_map_concat(l, :) = sig_map;

end

%% (3) Save whole-brain maps
cii_template.cdata = avg_net_diff';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);  % Number of layers
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);  % Number of voxels
cii_template.diminfo{1,1}.models(3:end) = [];  % Clean extra models if any
ciftisavereset(cii_template, fullfile(outdir, ['net_diff_' layer_str '.dtseries.nii']), 'wb_command');

cii_template.cdata = avg_ratio_diff';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);  % Number of layers
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);  % Number of voxels
cii_template.diminfo{1,1}.models(3:end) = [];  % Clean extra models if any
ciftisavereset(cii_template, fullfile(outdir, ['ratio_diff_' layer_str '.dtseries.nii']), 'wb_command');

cii_template.cdata = t_map_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);  % Number of layers
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);  % Number of voxels
cii_template.diminfo{1,1}.models(3:end) = [];  % Clean extra models if any
ciftisavereset(cii_template, fullfile(outdir, ['tmap_' layer_str '.dtseries.nii']), 'wb_command');

cii_template.cdata = sig_map_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);  % Number of layers
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);  % Number of voxels
cii_template.diminfo{1,1}.models(3:end) = [];  % Clean extra models if any
ciftisavereset(cii_template, fullfile(outdir, ['fdr_pmap_' layer_str '.dtseries.nii']), 'wb_command');
%% (4) Save visual ROI-only maps (masked)
% cii_template.cdata = (avg_diff .* roi_mask)';
% ciftisavereset(cii_template, fullfile(outdir, ['avg_diff_' layer_str '_ROI.dtseries.nii']), 'wb_command');
% 
% cii_template.cdata = (t_map .* roi_mask)';
% ciftisavereset(cii_template, fullfile(outdir, ['tmap_' layer_str '_ROI.dtseries.nii']), 'wb_command');
% 
% cii_template.cdata = (p_map .* roi_mask)';
% ciftisavereset(cii_template, fullfile(outdir, ['pmap_' layer_str '_ROI.dtseries.nii']), 'wb_command');
% 
% fprintf('Saved all maps for %s (whole-brain & ROI)\n', layer_str);