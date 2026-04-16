clc; clear;

% Add paths
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Set paths
% basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result_orig_mk_stimulus/';
basedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/';

layer_type = 'layer7';
pred_dir = fullfile(basedir, layer_type, 'E');
pe_dir   = fullfile(basedir, layer_type, 'Ahat');
c_dir = fullfile(basedir, layer_type, 'C'); 

subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
n_subjects = length(subjects);
n_layers = 7;

% Load one subject to get voxel count
sample = load(fullfile(pred_dir, subjects{1}, [subjects{1} '_avgcorr_lambda5.mat']));
n_voxels = size(sample.concat_corr, 2);


%% Initialize arrays for P-C / PE-C
diff_P_C_all  = zeros(n_subjects, n_layers, n_voxels);
diff_PE_C_all = zeros(n_subjects, n_layers, n_voxels);

%% Load all subjects P, PE, C and compute P-C / PE-C
for s = 1:n_subjects
    subject_id = subjects{s};
    fprintf('Loading %s...\n', subject_id);
    
    % Load
    pred_corr = load(fullfile(pred_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    pe_corr   = load(fullfile(pe_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    % C layer 
    C_corr = load(fullfile(c_dir, subject_id, [subject_id '_avgcorr_lambda5.mat'])).concat_corr;
    
    % Compute
    diff_P_C_all(s,:,:)  = pred_corr - C_corr;
    diff_PE_C_all(s,:,:) = pe_corr   - C_corr;
end

% Output directory
outdir = fullfile(basedir, layer_type, 'Prediction_vs_PE_perlayer');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% Load template for saving
cii_template = ciftiopen('/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii', 'wb_command');

%% Preallocate concat arrays for P-C / PE-C
avg_P_C_concat    = zeros(n_layers, n_voxels);
t_map_P_C_concat  = zeros(n_layers, n_voxels);
sig_map_P_C_concat= zeros(n_layers, n_voxels);

avg_PE_C_concat    = zeros(n_layers, n_voxels);
t_map_PE_C_concat  = zeros(n_layers, n_voxels);
sig_map_PE_C_concat= zeros(n_layers, n_voxels);

%% Process each layer
for l = 1:n_layers
    layer_str = sprintf('Layer%d', l);
    fprintf('\n--- Processing P-C / PE-C %s ---\n', layer_str);
    
    P_C_diff  = squeeze(diff_P_C_all(:, l, :));   % [subjects x voxels]
    PE_C_diff = squeeze(diff_PE_C_all(:, l, :));  % [subjects x voxels]
    
    % ----- P-C -----
    [~, p_map_P_C, ~, stats_P_C] = ttest(P_C_diff);
    t_map_P_C = stats_P_C.tstat;
    avg_P_C_diff = mean(P_C_diff, 1);
    
    p_fdr_P_C = mafdr(p_map_P_C', 'BHFDR', true);
    sig_mask_P_C = p_fdr_P_C < 0.05;
    sig_map_P_C = avg_P_C_diff(:) .* sig_mask_P_C(:);
    
    avg_P_C_concat(l,:)    = avg_P_C_diff;
    t_map_P_C_concat(l,:)  = t_map_P_C;
    sig_map_P_C_concat(l,:)= sig_map_P_C;
    
    % ----- PE-C -----
    [~, p_map_PE_C, ~, stats_PE_C] = ttest(PE_C_diff);
    t_map_PE_C = stats_PE_C.tstat;
    avg_PE_C_diff = mean(PE_C_diff, 1);
    
    p_fdr_PE_C = mafdr(p_map_PE_C', 'BHFDR', true);
    sig_mask_PE_C = p_fdr_PE_C < 0.05;
    sig_map_PE_C = avg_PE_C_diff(:) .* sig_mask_PE_C(:);
    
    avg_PE_C_concat(l,:)    = avg_PE_C_diff;
    t_map_PE_C_concat(l,:)  = t_map_PE_C;
    sig_map_PE_C_concat(l,:)= sig_map_PE_C;
end

%% Save P-C maps
cii_template.cdata = avg_P_C_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['P_minus_C_' layer_type '.dtseries.nii']), 'wb_command');

cii_template.cdata = t_map_P_C_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['tmap_P_minus_C_' layer_type '.dtseries.nii']), 'wb_command');

cii_template.cdata = sig_map_P_C_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['fdr_pmap_P_minus_C_' layer_type '.dtseries.nii']), 'wb_command');

%% Save PE-C maps
cii_template.cdata = avg_PE_C_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['PE_minus_C_' layer_type '.dtseries.nii']), 'wb_command');

cii_template.cdata = t_map_PE_C_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['tmap_PE_minus_C_' layer_type '.dtseries.nii']), 'wb_command');

cii_template.cdata = sig_map_PE_C_concat';
cii_template.diminfo{1,1}.length = size(cii_template.cdata, 1);
cii_template.diminfo{1,2}.length = size(cii_template.cdata, 2);
cii_template.diminfo{1,1}.models(3:end) = [];
ciftisavereset(cii_template, fullfile(outdir, ['fdr_pmap_PE_minus_C_' layer_type '.dtseries.nii']), 'wb_command');

fprintf(' P-C / PE-C maps saved in %s\n', outdir);
