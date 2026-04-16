%% Per-subject PCA across layers: normalize W, take subject PC1, then average across subjects
clc;clear;
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc; 

%% Layers to include (order matters for PCA)
% layername = {'/C0'; '/C1'; '/C2'; '/C3'};
% layername = {'/C0'; '/C1'; '/C2'; '/C3'; '/C4'; '/C5'; '/C6'};

layername = {'/E0'; '/E1'; '/E2'; '/E3'};  
% layername = {'/E0'; '/E1'; '/E2'; '/E3'; '/E4'; '/E5'; '/E6'};

% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6'}; 

%% Subjects
num_subjects = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06', ...
                'sub-09','sub-10','sub-14','sub-15','sub-16','sub-17', ...
                'sub-18','sub-19','sub-20'};

%% Layers (example: Ahat0~Ahat3)
n_sub = numel(num_subjects);
n_lyr = numel(layername);

% Resolve layer set (for paths)
if n_lyr == 4
    layer_name = 'layer4';
elseif n_lyr == 7
    layer_name = 'layer7';
else
    error('layername must have 4 or 7 entries.');
end

% Extract family prefix from '/Ahat0'
tok = regexp(layername{1}, '^/([A-Za-z]+)\d+$', 'tokens','once');
assert(~isempty(tok), 'layername entries must look like ''/Ahat0'', ''/E3'', etc.');
type = tok{1};

%% Paths
dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/', ...
            'Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result/', ...
            layer_name, '/', type, '/'];

if ~exist(dataroot,'dir'); mkdir(dataroot); end

%% Helper
normalize_vec = @(v) local_minmax_pm100(v);

%% Probe shape
probe_sid = num_subjects{1};
probe_lyr = layername{1}(2:end);
probe_file = fullfile(dataroot, probe_sid, ['W_lambda5_' probe_lyr '.mat']);
assert(exist(probe_file,'file')==2, 'Missing: %s', probe_file);
tmp = load(probe_file, 'W');
[~, n_vox] = size(tmp.W);
clear tmp;
%% --------- PASS 1: per subject  layer: PCA(Features->Voxels), save PC1 ----------
for si = 1:14
    sid = num_subjects{si};
    fprintf('Subject %s\n', sid); 
    subj_outdir = fullfile(dataroot, sid);
    if ~exist(subj_outdir,'dir'), mkdir(subj_outdir); end

    for li = 1:n_lyr
        current_layer = layername{li}(2:end);
        f = fullfile(dataroot, sid, ['W_lambda5_' current_layer '.mat']);
        if ~exist(f, 'file')
            warning('Missing (skip layer): %s', f);
            continue;
        end

        S = load(f, 'W');  W = S.W;  W(~isfinite(W)) = NaN;

        % PCA across features (obs = features, vars = voxels)
        [coeff, score, latent, tsq, explained] = pca(W, 'Algorithm','svd', 'Centered', true);

        % PC1 loading across voxels
        pc1_loading = coeff(:,1)';  % [1 x nVox]

        % Save per subject - layer
        save_file = fullfile(subj_outdir, sprintf('W_%s_PC1_norm.mat', current_layer));
        save(save_file, 'pc1_loading','-v7.3');
    end 
end
      
%% pc1 beta normalization
for li = 1:n_lyr
    current_layer = layername{li}(2:end);
    fprintf('Layer: %s\n', current_layer);

    for si = 1:n_sub
        sid = num_subjects{si};
        subj_outdir = fullfile(dataroot, sid);
        fname = fullfile(subj_outdir, sprintf('W_%s_PC1_norm.mat', current_layer)); 

        if ~exist(fname, 'file')
            warning('Missing file (skip): %s', fname);
            continue;
        end

        % --- load pc1 loading (accept both 'pc1_loading' or 'pc1')
        S = load(fname);
        if isfield(S, 'pc1_loading')
            pc1_loading = S.pc1_loading;
        else
            warning('No pc1_loading/pc1 in %s (skip)', fname);
            continue;
        end
        pc1_loading = pc1_loading(:)';     % row vector
        pc1_loading(~isfinite(pc1_loading)) = NaN;

        if exist(fname, 'file')
            save(fname, 'pc1_loading', '-v7.3');
        else
            continue;
        end

        % --- 1) max-abs scaling to [-100, 100] (sign preserved)
        s = max(abs(pc1_loading(isfinite(pc1_loading))));
        if isempty(s) || ~isfinite(s) || s <= 0
            pc1_norm_maxabs = zeros(size(pc1_loading));
        else
            pc1_norm_maxabs = 100 * (pc1_loading / s);
            % optional strict clip
            pc1_norm_maxabs(pc1_norm_maxabs > 100)  = 100;
            pc1_norm_maxabs(pc1_norm_maxabs < -100) = -100;
        end

        % --- 2) z-score across voxels (per subject)
        m  = mean(pc1_loading, 'omitnan');
        sd = std(pc1_loading, 0, 'omitnan');
        if ~isfinite(sd) || sd == 0
            pc1_norm_zscore = zeros(size(pc1_loading));
        else
            pc1_norm_zscore = (pc1_loading - m) ./ sd;
        end

        % --- append into the SAME mat file
        norm_info = struct('scale', 'maxabs_[-100,100]_and_zscore', ...
                           'timestamp', datestr(now, 30));
        save(fname, 'pc1_norm_maxabs', 'pc1_norm_zscore', '-v7.3', '-append');

        fprintf('Appended normals  %s\n', fname);
    end
end

%% --------- PASS 2: group average per layer over subjects ----------
% Template CIFTI (for saving)
template_path = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
template_file = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
cii = ciftiopen([template_path, template_file], 'wb_command');

layer_wise_group_beta_maxabs = nan(n_lyr, n_vox);
layer_wise_group_beta_zscore = nan(n_lyr, n_vox);

for li = 1:n_lyr
    current_layer = layername{li}(2:end);
    fprintf('Aggregating layer: %s\n', current_layer);

    all_beta_maxabs = nan(n_sub, n_vox);
    all_beta_zscore = nan(n_sub, n_vox);

    for si = 1:n_sub
        sid = num_subjects{si};
        subj_outdir = fullfile(dataroot, sid);
        fname = fullfile(subj_outdir, sprintf('W_%s_PC1_norm.mat', current_layer));
        if ~exist(fname,'file')
            warning('Missing result (skip in mean): %s', fname);
            continue;
        end
        S1 = load(fname, 'pc1_norm_maxabs');
        S2 = load(fname, 'pc1_norm_zscore');
        if isfield(S1,'pc1_norm_maxabs')
            all_beta_maxabs(si, :) = S1.pc1_norm_maxabs;
        end
        if isfield(S2,'pc1_norm_zscore')
            all_beta_zscore(si, :) = S2.pc1_norm_zscore;
        end
    end
    % maxabs
    group_beta_map_maxabs = nanmean(all_beta_maxabs, 1);         % [1 x nVox]
    layer_wise_group_beta_maxabs(li, :) = group_beta_map_maxabs; % [nLayer x nVox]

    % maxabs
    group_beta_map_zscore = nanmean(all_beta_zscore, 1);         % [1 x nVox]
    layer_wise_group_beta_zscore(li, :) = group_beta_map_zscore; % [nLayer x nVox]
end

% save group avg beta .mat
save(fullfile(dataroot, ['group_beta_' type '_all_layers_PCA_maxabs.mat']), 'group_beta_map_maxabs', '-v7.3');
save(fullfile(dataroot, ['group_beta_' type '_all_layers_PCA_zscore.mat']), 'group_beta_map_zscore', '-v7.3');

% maxabs 
% Save as dtseries (voxels x layers)
new_cii = cii;
new_cii.cdata = layer_wise_group_beta_maxabs';  % [nVox x nLayer]
new_cii.diminfo{1,1}.length = size(new_cii.cdata, 1);
new_cii.diminfo{1,2}.length = size(new_cii.cdata, 2);
new_cii.diminfo{1,1}.models(3:end) = [];

outfile = fullfile(dataroot, ['group_beta_' type '_all_layers_PCA_maxabs.dtseries.nii']);
ciftisavereset(new_cii, outfile, 'wb_command');
fprintf('Saved: %s\n', outfile);

% zscore
% Save as dtseries (voxels x layers)
new_cii = cii;
new_cii.cdata = layer_wise_group_beta_zscore';  % [nVox x nLayer]
new_cii.diminfo{1,1}.length = size(new_cii.cdata, 1);
new_cii.diminfo{1,2}.length = size(new_cii.cdata, 2);
new_cii.diminfo{1,1}.models(3:end) = [];

outfile = fullfile(dataroot, ['group_beta_' type '_all_layers_PCA_zscore.dtseries.nii']);
ciftisavereset(new_cii, outfile, 'wb_command');
fprintf('Saved: %s\n', outfile);

