function eval_fdr_pipeline()
% End-to-end: run-wise r -> subject-wise Fisher zbar -> group t-test -> BH-FDR -> ROI selection

clc;

%% =================== USER CONFIG (best-practice defaults) ===================
subjects  = {'sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20'};
% layername = {'/C0'; '/C1'; '/C2'; '/C3'};
% layername = {'/C0'; '/C1'; '/C2'; '/C3'; '/C4'; '/C5'; '/C6'};

layername = {'/E0'; '/E1'; '/E2'; '/E3'};  
% layername = {'/E0'; '/E1'; '/E2'; '/E3'; '/E4'; '/E5'; '/E6'};

% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};
% layername = {'/Ahat 0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6'}; 


n_lyr = numel(layername);

% Resolve layer set (for paths)
if n_lyr == 4 
    layer_tag = 'layer4';
elseif n_lyr == 7
    layer_tag = 'layer7';
else
    error('layername must have 4 or 7 entries.');
end

% Derive "type" (e.g., 'Ahat' from '/Ahat0', 'E' from '/E3')
tok = regexp(layername{1}, '^/([A-Za-z]+)\d+$', 'tokens','once');
assert(~isempty(tok), 'layername entries must look like ''/Ahat0'', ''/E3'', etc.');
type = tok{1};  % e.g., 'Ahat' or 'E'

runs      = 5:8;                                        % test runs
q_level   = 0.05;                                       % BH-FDR
roi_keep_threshold = 0.10;                              % keep ROI if >=10% vertices significant

base_dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/' ...
                 'Titanic_trained/GUMP_test/encoding_analysis/original_prednet/result/' ...
                 layer_tag, '/', type, '/'];

% CIFTI helpers
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Geometry template (any dtseries with correct geometry)
tmpl_path = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
tmpl_file = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
cii_tmpl  = ciftiopen(fullfile(tmpl_path, tmpl_file), 'wb_command');

% Glasser dlabel for ROI mapping (cortex)
glasser = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii','wb_command');
labels  = glasser.cdata(:);
roi_ids = unique(labels(labels > 0));

%% =================== 1) WITHIN-SUBJECT: combine runs via Fisher z ===================
fprintf('\n[1/3] Building subject-wise Fisher zbar (run)...\n');
for s = 1:numel(subjects)
    sub    = subjects{s};
    subdir = fullfile(base_dataroot, sub);
    if ~exist(subdir,'dir'), mkdir(subdir); end

    all_zbar = cell(numel(layername),1);
    all_rbar = cell(numel(layername),1);

    for L = 1:numel(layername)
        r_mat = []; % [V x R]
        for i = 1:numel(runs)
            corrfile = fullfile(subdir, sprintf('%s_corr_lambda5_run_%d.mat', sub, runs(i)));
            if ~exist(corrfile, 'file')
                error('Missing correlation file: %s', corrfile);
            end
            S = load(corrfile, 'concat_corr');  % [nLayer x V]
            r_vec = S.concat_corr(L,:).';       % [V x 1]
            r_mat = [r_mat, r_vec];             % [V x R]
        end

        % numeric stability
        r_mat = max(min(r_mat, 0.999999), -0.999999);

        % handle NaN per run/voxel + fisher z
        z_mat = atanh(r_mat);  % fisher z transform
        zbar = nanmean(z_mat, 2);  % [Vx1]: ignore NaN and mean across run (within the same layer)
        zbar(~isfinite(zbar)) = 0;   % if NaN all vector then make it as 0 or remove later

        rbar = tanh(zbar);

        all_zbar{L} = zbar.';                       % 1 x V
        all_rbar{L} = rbar.';                       % 1 x V
    end

    concat_zbar = cat(1, all_zbar{:});              % [nLayer x V]  <-- group statistics input
    concat_rbar = cat(1, all_rbar{:});              % [nLayer x V]  (reporting/plots)

    save(fullfile(subdir, [sub,'_avgcorr_lambda5_FisherZ.mat']), 'concat_zbar', 'concat_rbar', '-v7.3');
    fprintf('  %s: saved FisherZ/R mats\n', sub);
end

%% =================== 2) GROUP: one-sample t-test per layer -> BH-FDR ===================
fprintf('\n[2/3] Group-level one-sample t-test and BH-FDR (per layer)...\n');
fprintf(['layer type is ', type]);

out_dir = fullfile(base_dataroot, 'group_stats');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

V = 59412;
Sig_all =  false(numel(layername), V); 
all_ROIs = cell(numel(layername),1);


for L = 1:numel(layername)
    fprintf('  Layer %s ...\n', layername{L}(2:end));

    % Stack subjects' zbar: Z [S x V]
    Z = [];
    for s = 1:numel(subjects)
        sub = subjects{s};
        S   = load(fullfile(base_dataroot, sub, [sub,'_avgcorr_lambda5_FisherZ.mat']), 'concat_zbar');
        zbar = S.concat_zbar(L,:);                 % 1 x V
        Z = [Z; zbar];                              % S x V
    end

    % one-sample t-test against 0 (two-sided)
    Ssub = size(Z,1);              % subjects
    valid = isfinite(Z);           % S x V

    n_eff = sum(valid, 1);         % 1 x V
    m     = nanmean(Z, 1);         % 1 x V
    sd    = nanstd(Z, 0, 1);       % 1 x V
    se    = sd ./ sqrt(max(n_eff, 1));   % avoid sqrt(0)

    % t-values and df per voxel
    tvals = m ./ max(se, eps);     % guard divide-by-zero
    df_v  = max(n_eff - 1, 1);     % df <= 0 -> 1 as a safe fallback

    % p-values (voxels with n_eff <= 1 -> p=1)
    p = 2 * tcdf(-abs(tvals), df_v);
    p(n_eff <= 1) = 1;
    p(~isfinite(p)) = 1;

    % BH-FDR (per layer)
    [~, ~, ~, qvals] = fdr_bh(p(:), q_level, 'pdep', 'no');
    qmap = reshape(qvals, size(p));         % 1 x V
    sig  = (qmap <= q_level) & isfinite(qmap);
    Sig_all(L,:) = sig; 

    % group mean r = tanh(mean zbar)
    rbar_group = tanh(m);

    % outputs: -log10(q), mask, thresholded r
    neglogq = -log10(qmap); neglogq(~isfinite(neglogq)) = 0;
    rbar_thr = rbar_group .* sig;

    
    % Save dtseries (each is 1 x V row)
    avg_cdata = cii_tmpl;

    avg_cdata.cdata = double(sig)';
    avg_cdata.diminfo{1, 1}.length = size(avg_cdata.cdata, 1);
    avg_cdata.diminfo{1, 2}.length = size(avg_cdata.cdata, 2);
    avg_cdata.diminfo{1, 1}.models(3:end) = [];
    save_file_name =  fullfile(out_dir, sprintf('Group_FDR_mask_%s.dtseries.nii', layername{L}(2:end)));
    ciftisavereset(avg_cdata, (save_file_name), 'wb_command');

    % avg_cdata.cdata = neglogq';
    % avg_cdata.diminfo{1, 1}.length = size(avg_cdata.cdata, 1);
    % avg_cdata.diminfo{1, 2}.length = size(avg_cdata.cdata, 2);
    % avg_cdata.diminfo{1, 1}.models(3:end) = [];
    % save_file_name =  fullfile(out_dir, sprintf('Group_FDR_neglogq_%s.dtseries.nii', layername{L}(2:end)));
    % ciftisavereset(avg_cdata, (save_file_name), 'wb_command');

    % avg_cdata.cdata = rbar_thr';
    % avg_cdata.diminfo{1, 1}.length = size(avg_cdata.cdata, 1);
    % avg_cdata.diminfo{1, 2}.length = size(avg_cdata.cdata, 2);
    % avg_cdata.diminfo{1, 1}.models(3:end) = [];
    % save_file_name =  fullfile(out_dir, sprintf('Group_FDR_rbar_thr_%s.dtseries.nii', layername{L}(2:end)));
    % ciftisavereset(avg_cdata, (save_file_name), 'wb_command');

    % save_dtseries_like(cii_tmpl, neglogq, fullfile(out_dir, sprintf('Group_FDR_neglogq_%s.dtseries.nii', layername{L}(2:end))));
    % save_dtseries_like(cii_tmpl, double(sig), fullfile(out_dir, sprintf('Group_FDR_mask_%s.dtseries.nii',     layername{L}(2:end))));
    % save_dtseries_like(cii_tmpl, rbar_thr,    fullfile(out_dir, sprintf('Group_FDR_rbar_thr_%s.dtseries.nii', layername{L}(2:end))));

    %% =================== 3) ROI selection by significant proportion ===================
    sig_any = sig(:);  % 1xV logical
    keep_rois = [];
    roi_stats = zeros(numel(roi_ids), 3); % [ROI_ID, N_sig, Prop_sig]

    for k = 1:numel(roi_ids)
        rid = roi_ids(k);
        idx = (labels == rid);
        n_all = sum(idx);
        n_sig = sum(sig_any(idx));
        prop  = n_sig / max(n_all,1);
        roi_stats(k,:) = [rid, n_sig, prop];
        if prop >= roi_keep_threshold
            keep_rois(end+1) = rid; %#ok<AGROW>
        end
    end
  
    all_ROIs{L} = keep_rois;

    % T = array2table(roi_stats, 'VariableNames', {'ROI_ID','N_sig','Prop_sig'});
    % writetable(T, fullfile(out_dir, sprintf('Group_FDR_ROIstats_%s.csv', layername{L}(2:end))));
    % save(fullfile(out_dir, sprintf('Group_FDR_keepROIs_%s.mat', layername{L}(2:end))), 'keep_rois')
end
% save ROIS across layer 
fprintf('    kept %d ROIs (>= %.0f%% significant)\n', numel(keep_rois), roi_keep_threshold*100);
save(fullfile(out_dir, sprintf('Group_ANY_FDR_keepROIs.mat', layername{L}(2:end))), 'all_ROIs');

fprintf('\n[Done] Group FDR + ROI selection completed.\n');

% ---- After the loop: build combined masks ----
mask_any  = any(Sig_all, 1);
mask_all  = all(Sig_all, 1);
layer_cnt = sum(Sig_all, 1);

% ---- Save union/intersection maps ----
avg_cdata = cii_tmpl;
avg_cdata.cdata = double(mask_any)';
avg_cdata.diminfo{1, 1}.length = size(avg_cdata.cdata, 1);
avg_cdata.diminfo{1, 2}.length = size(avg_cdata.cdata, 2);
avg_cdata.diminfo{1, 1}.models(3:end) = [];
ciftisavereset(avg_cdata, fullfile(out_dir, 'Group_FDR_mask_ANYlayer.dtseries.nii'), 'wb_command');

avg_cdata = cii_tmpl;
avg_cdata.cdata = double(mask_all)';
avg_cdata.diminfo{1, 1}.length = size(avg_cdata.cdata, 1);
avg_cdata.diminfo{1, 2}.length = size(avg_cdata.cdata, 2);
avg_cdata.diminfo{1, 1}.models(3:end) = [];
ciftisavereset(avg_cdata, fullfile(out_dir, 'Group_FDR_mask_ALLlayers.dtseries.nii'), 'wb_command');


% % Layer-count map (0..num_layers)
% avg_cdata.cdata = double(layer_cnt);         % 1 x V
% avg_cdata.diminfo{1,1}.length = 1;
% avg_cdata.diminfo{1,2}.length = size(avg_cdata.cdata, 2);
% avg_cdata.diminfo{1,1}.models(3:end) = [];
% ciftisavereset(avg_cdata, fullfile(out_dir, 'Group_FDR_layerCount.dtseries.nii'), 'wb_command');
% 
% % (Optional) Stack per-layer masks as a multi-frame dtseries [L x V]
% avg_cdata.cdata = double(Sig_all);           % nLayer x V
% avg_cdata.diminfo{1,1}.length = size(Sig_all, 1);  % frames = nLayer
% avg_cdata.diminfo{1,2}.length = size(Sig_all, 2);  % V
% avg_cdata.diminfo{1,1}.models(3:end) = [];
% ciftisavereset(avg_cdata, fullfile(out_dir, 'Group_FDR_mask_perLayer.dtseries.nii'), 'wb_command');

end

%% =================== Helpers ===================
function save_dtseries_like(cii_template, row1xV, out_path)
% row1xV: 1 x V numeric row
out = cii_template;
out.cdata = row1xV;
out.diminfo{1,1}.length = size(out.cdata,1);
out.diminfo{1,2}.length = size(out.cdata,2);
out.diminfo{1,1}.models(3:end) = [];
ciftisavereset(out, out_path, 'wb_command');
end

function [h, crit_p, adj_p, qvals] = fdr_bh(pvals,q,method,report)
% Minimal BH-FDR (Benjamini & Hochberg)
if nargin < 2 || isempty(q), q = 0.05; end
if nargin < 3 || isempty(method), method = 'pdep'; end
if nargin < 4, report = 'no'; end

p = pvals(:);
valid = isfinite(p); p(~valid) = 1;
[ps, idx] = sort(p);
V = numel(p); I = (1:V)';

if strcmpi(method,'pdep')
    thresh = (I./V) * q;
elseif strcmpi(method,'dep')
    cV = sum(1./(1:V));
    thresh = (I./V) * (q/cV);
else
    error('method must be ''pdep'' or ''dep''.');
end

w = find(ps <= thresh);
if isempty(w)
    crit_p = 0; h = false(V,1);
else
    crit_p = ps(max(w)); h = p <= crit_p;
end

adj_ps = ps .* V ./ I;
for i = V-1:-1:1
    adj_ps(i) = min(adj_ps(i), adj_ps(i+1));
end
adj_p = zeros(V,1); adj_p(idx) = adj_ps;
qvals = adj_p;

% restore NaNs (optional): leave as 1 / non-significant
h(~valid) = false; qvals(~valid) = 1;

if strcmpi(report,'yes')
    fprintf('FDR-BH: %d/%d significant (q=%.3f, crit=%.4g)\n', sum(h), V, q, crit_p);
end
end
