clc; clear;

% Add paths
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Set paths
basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result_orig_mk_stimulus/';
% basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/';

layer_type = 'layer4';  % or 'layer4'
pred_dir = fullfile(basedir, layer_type, 'E');
pe_dir   = fullfile(basedir, layer_type, 'Ahat');

subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
n_subjects = length(subjects);
n_layers = 4;

% Load one subject to get voxel count
sample = load(fullfile(pred_dir, subjects{1}, [subjects{1} '_avgcorr_lambda5.mat']));
n_voxels = size(sample.concat_corr, 2);

% Initialize arrays
diff_net_all = zeros(n_subjects, n_layers, n_voxels);
diff_ratio_all = zeros(n_subjects, n_layers, n_voxels);

% Load difference (Prediction - PE) for all subjects/layers
for s = 1:n_subjects
    subject_id = subjects{s};
    fprintf('Loading %s...\n', subject_id);
    pred_corr = load(fullfile(pred_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    pe_corr   = load(fullfile(pe_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    
    
    diff_net_all(s, :, :) = pred_corr - pe_corr;
    
    safe_eps = 1e-3;
    diff_ratio_all(s, :, :) = log((pred_corr + safe_eps) ./ (pe_corr + safe_eps));
end

% Load template for saving
cii_template = ciftiopen('/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii', 'wb_command');

% Output directory
outdir = fullfile(basedir, layer_type, 'Prediction_vs_PE_perlayer');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% Preallocate concat arrays
avg_net_diff_concat = zeros(n_layers, n_voxels);
t_map_net_concat = zeros(n_layers, n_voxels);
sig_map_net_concat = zeros(n_layers, n_voxels);

avg_ratio_diff_concat = zeros(n_layers, n_voxels);
t_map_ratio_concat = zeros(n_layers, n_voxels);
sig_map_ratio_concat = zeros(n_layers, n_voxels);

% Process each layer
for l = 1:n_layers
    layer_str = sprintf('Layer%d', l);
    fprintf('\n--- Processing %s ---\n', layer_str);

    net_diff = squeeze(diff_net_all(:, l, :));  % [subjects x voxels]
    ratio_diff = squeeze(diff_ratio_all(:, l, :));  % [subjects x voxels]

    %% Net diff stats
    [~, p_map_net, ~, stats_net] = ttest(net_diff);
    t_map_net = stats_net.tstat;
    avg_net_diff = mean(net_diff, 1);

    p_fdr_net = mafdr(p_map_net', 'BHFDR', true);
    sig_mask_net = p_fdr_net < 0.05;
    sig_map_net = avg_net_diff(:) .* sig_mask_net(:);

    % Save to concat arrays
    avg_net_diff_concat(l, :) = avg_net_diff;
    t_map_net_concat(l, :) = t_map_net;
    sig_map_net_concat(l, :) = sig_map_net;

    %% Ratio diff stats
    [~, p_map_ratio, ~, stats_ratio] = ttest(ratio_diff);
    t_map_ratio = stats_ratio.tstat;
    avg_ratio_diff = mean(ratio_diff, 1);

    p_fdr_ratio = mafdr(p_map_ratio', 'BHFDR', true);
    sig_mask_ratio = p_fdr_ratio < 0.05;
    sig_map_ratio = avg_ratio_diff(:) .* sig_mask_ratio(:);

    % Save to concat arrays
    avg_ratio_diff_concat(l, :) = avg_ratio_diff;
    t_map_ratio_concat(l, :) = t_map_ratio;
    sig_map_ratio_concat(l, :) = sig_map_ratio;
end
%% (4) Save Visual ROI masked version (P/PE ratio version)

% Load ROI mask
label_32k = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii', 'wb_command');
vertex = label_32k.cdata;
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

%% Save ROI-masked ratio maps
roi_mask_mat = repmat(roi_mask', [1, n_layers]); % (n_voxels x n_layers)

cii_template.cdata = (avg_net_diff_concat' .* roi_mask_mat);
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['net_diff_' layer_type '_ROI.dtseries.nii']), 'wb_command');

cii_template.cdata = (avg_ratio_diff_concat' .* roi_mask_mat);
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['ratio_diff_' layer_type '_ROI.dtseries.nii']), 'wb_command');

cii_template.cdata = (t_map_ratio_concat' .* roi_mask_mat);
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['tmap_ratio_' layer_type '_ROI.dtseries.nii']), 'wb_command');

cii_template.cdata = (sig_map_ratio_concat' .* roi_mask_mat);
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['fdr_pmap_ratio_' layer_type '_ROI.dtseries.nii']), 'wb_command');

fprintf('? Visual ROI ratio maps saved (wholebrain + ROI)\n');
